# EVAL

```bash
set -a
source .env
set +a
echo "HF_HOME=$HF_HOME"
```

```bash
log_num=0
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28${log_num}35
# CUDA_VISIBLE_DEVICES=9
# --dp 1
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 8 --tp 1 
#  > sglang${log_num}.log 2>&1 &


log_num=0
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \

# --strict_prompt


bash evaluation/kill_sglang_server.sh
```