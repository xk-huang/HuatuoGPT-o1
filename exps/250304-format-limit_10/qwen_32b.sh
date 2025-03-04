
log_num=0
model_name="Qwen/Qwen2.5-32B-Instruct" # Path to the model you are deploying
port=28${log_num}35
# CUDA_VISIBLE_DEVICES=9
# --dp 1
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 2 --tp 4 \
# > sglang${log_num}.log 2>&1 &

exit


output_dir=outputs/250304-format-limit_10/qwen2_5_32b_instruct
log_num=0
model_name="Qwen/Qwen2.5-32B-Instruct" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=huatuo \
--output_dir=$output_dir/huatuo

python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=huatuo \
--strict_prompt \
--output_dir=$output_dir/huatuo-strict

for format in "box" "answer"; do
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=$format \
--output_dir=$output_dir/$format
done

exit


bash evaluation/kill_sglang_server.sh