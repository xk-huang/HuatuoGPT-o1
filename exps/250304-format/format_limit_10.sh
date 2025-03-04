
log_num=0
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28${log_num}35
# CUDA_VISIBLE_DEVICES=9
# --dp 1
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 8 --tp 1 > sglang${log_num}.log 2>&1 &

exit


output_dir=outputs/250304-format/
log_num=0
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28${log_num}35
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=huatuo \
--output_dir=$output_dir/HuatuoGPT-o1-8B-huatuo

python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=huatuo \
--strict_prompt \
--output_dir=$output_dir/HuatuoGPT-o1-8B-huatuo-strict

for format in "box" "answer"; do
python evaluation/eval.py --model_name $model_name \
--eval_file evaluation/data/eval_data.json \
--port $port \
--limit=10 \
--format=$format \
--output_dir=$output_dir/HuatuoGPT-o1-8B-$format
done

# --strict_prompt


bash evaluation/kill_sglang_server.sh