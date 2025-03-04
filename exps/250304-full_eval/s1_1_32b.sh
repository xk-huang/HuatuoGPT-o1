
log_num=0
model_name="simplescaling/s1.1-32B" # Path to the model you are deploying
port=28${log_num}35
# CUDA_VISIBLE_DEVICES=9
# --dp 1
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 2 --tp 4 \
# > sglang${log_num}.log 2>&1 &

exit


output_dir=outputs/250304-full_eval/s1_1_32b-max_tokens_512
log_num=0
model_name="simplescaling/s1.1-32B" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--format=huatuo \
--force_think \
--max_new_tokens 512 \
--output_dir=$output_dir/huatuo

output_dir=outputs/250304-full_eval/s1_1_32b-max_tokens_1024
log_num=0
model_name="simplescaling/s1.1-32B" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--format=huatuo \
--force_think \
--max_new_tokens 1024 \
--output_dir=$output_dir/huatuo

output_dir=outputs/250304-full_eval/s1_1_32b-max_tokens_2048
log_num=0
model_name="simplescaling/s1.1-32B" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--format=huatuo \
--force_think \
--max_new_tokens 2048 \
--output_dir=$output_dir/huatuo

exit

bash evaluation/kill_sglang_server.sh