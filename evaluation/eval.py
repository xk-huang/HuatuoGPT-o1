import argparse
from tqdm import tqdm
import argparse
import openai
from jinja2 import Template
import os
import json
from transformers import AutoTokenizer
from jinja2 import Template
from scorer import get_results
from pathlib import Path

def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template',type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=16)    
    parser.add_argument("--only_get_results", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--format", type=str, default="huatuo")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--force_think", action="store_true")
    parser.add_argument("--think_str", type=str, default="<|im_start|>think")
    parser.add_argument("--answer_str", type=str, default="<|im_start|>answer")
    parser.add_argument("--overlength_str", type=str, default="\nFinal Answer:")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_answer_length", type=int, default=100)
    args = parser.parse_args()

    if args.only_get_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        task_name = os.path.split(args.model_name)[-1]

        task_name = task_name + os.path.basename(args.eval_file).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
        save_path = output_dir / f'{task_name}.json'
        get_results(save_path)
        return


    print(f"Using local API server at port {args.port}")
    client = openai.Client(
    base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template = Template(tokenizer.chat_template)

    def call_model(prompts, model, max_new_tokens=50, print_example =False):
        if print_example:
            print("Example:")
            print(prompts[0])
        preds = []
        if args.use_chat_template:
            prompts = [template.render(messages=[{"role": "user", "content": prom}],bos_token= tokenizer.bos_token,add_generation_prompt=True) for prom in prompts]
        
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt,add_special_tokens= False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt[-args.max_tokens:])
            prompts = new_prompts

        stop = None
        if args.force_think:
            # https://github.com/simplescaling/s1
            prompts = [i+args.think_str for i in prompts]
            stop = ["<|im_end|>", "<|im_start|>"]

        response = client.completions.create(
            model="default",
            prompt=prompts,
            temperature=0.1, 
            # top_p=0.9,
            max_tokens=max_new_tokens - args.final_answer_length,
            stop=stop,
            seed=args.seed,
        )
        preds = [x.text for x in response.choices]
        finish_reasons = [x.finish_reason for x in response.choices]
        pred_lengths = []
        if args.force_think:
            # https://github.com/simplescaling/s1
            # https://github.com/simplescaling/s1/blob/main/eval/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
            final_prompts = []
            for prompt, pred, finish_reason in zip(prompts, preds, finish_reasons):
                final_prompt = prompt + pred
                pred_length = len(tokenizer.encode(pred, add_special_tokens=False))

                if not final_prompt.endswith("\n"):
                    final_prompt += "\n"
                    pred_length += len(tokenizer.encode("\n", add_special_tokens=False))

                final_prompt += args.answer_str
                pred_length += len(tokenizer.encode(args.answer_str, add_special_tokens=False))

                if finish_reason == "length":
                    final_prompt += args.overlength_str
                    pred_length += len(tokenizer.encode(args.overlength_str, add_special_tokens=False))

                final_prompts.append(final_prompt)
                pred_lengths.append(pred_length)

            response = client.completions.create(
                model="default",
                prompt=final_prompts,
                temperature=0.1, 
                # top_p=0.9, 
                max_tokens=max_new_tokens - max(pred_lengths),
                stop=stop,
                seed=args.seed,
            )
            preds = [pred + x.text for pred, x in zip(preds, response.choices)]
        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds

    input_data = load_file(args.eval_file)
    model = None

    if args.limit > 0:
        print(f"limit: {args.limit}")
        input_data = input_data[:args.limit]
 
    final_results = []
    if args.format == 'huatuo':
        if args.strict_prompt:
            query_prompt = "Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
        else:
            query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"
    elif args.format == 'box':
        # https://github.com/open-thoughts/open-thoughts/blob/e3ad6c98b0e03ce7f42e7ee8064ab1c5a59f18cf/open_thoughts/math/reason.py#L16
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}. Return your final response within \\boxed{{}}"
    elif args.format == 'answer':
        # https://github.com/openai/simple-evals/blob/0a6e8f62e52bc5ae915f752466be3af596caf392/mgsm_eval.py#L33
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}. Finish with: 'Final Answer: {{answer}}' where [answer] is just the option"
    else:
        raise ValueError(f"unknown query_prompt: {query_prompt}")
    print(f"query_prompt: {query_prompt}")


    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if len(batch) == 0:
            break

        for item in batch:
            item['option_str'] = '\n'.join([ f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [ item["input_str"] for item in batch]
    
        if idx == 0:
            print_example = True
        else:
            print_example = False
        
        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, print_example=print_example)

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = os.path.split(args.model_name)[-1]

    task_name = task_name + os.path.basename(args.eval_file).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
    save_path = output_dir / f'{task_name}.json'

    with open(save_path,'w') as fw:
        json.dump(final_results,fw,ensure_ascii=False,indent=2)

    # get results
    get_results(save_path)


if __name__ == "__main__":
    main()
