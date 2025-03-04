# ENV

`.env` file
```bash
HF_HOME=cache/
```


```bash
# Install python
conda create -y -n xiaoke-huatuo_gpt_o1 python=3.10
source activate
conda activate xiaoke-huatuo_gpt_o1
which python
# should be `/insomnia001/depts/5sigma/users/xh2689/xiaoke/misc/miniconda/envs/xiaoke-methylformer/bin/python`

# Install pytorch 2.4.0
# https://pytorch.org/get-started/locally/
# We use cuda 12.4
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# [BUG] AttributeError: 'PyNcclCommunicator' object has no attribute 'device' https://github.com/vllm-project/vllm/issues/8420

# pip install -r requirements.txt
# xformer is too low

# https://docs.sglang.ai/start/install.html
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```


## Sync to other nodes

```bash
# sync code
output_files=()
for server_id in $(seq -f "%02g" 1 12); do
    target_node="ucsc-vlaa-${server_id}"
    temp_file=$(mktemp)

    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; rsync -avP -e 'ssh -o StrictHostKeyChecking=no' /data1/xhuan192/codes/HuatuoGPT-o1/ $target_node:/data1/xhuan192/codes/HuatuoGPT-o1/ >> $temp_file) &

    sleep 0.1
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"
```

Other commands

```bash
# check storage
output_files=()
for server_id in $(seq -f "%02g" 1 12); do
    target_node="tailscale-ucsc-vlaa-${server_id}"
    temp_file=$(mktemp)

    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node "mkdir -p /data1/xhuan192/codes/ && df -h | grep data1" >> $temp_file) &

    sleep 0.1
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"


# sync ssh config
curl -s https://gist.githubusercontent.com/xk-?/?/raw/?/? -o ~/.ssh/config
chmod 600 ~/.ssh/config
chmod 700 ~/.ssh

output_files=()
for server_id in $(seq -f "%02g" 1 12); do
    target_node="tailscale-ucsc-vlaa-${server_id}"
    temp_file=$(mktemp)

    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node "curl -s https://gist.githubusercontent.com/xk-?/?/raw/?/? -o ~/.ssh/config && 
    chmod 600 ~/.ssh/config && 
    chmod 700 ~/.ssh" >> $temp_file) &

    sleep 0.1
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"


# install env
output_files=()
for server_id in $(seq -f "%02g" 1 12); do
    target_node="ucsc-vlaa-${server_id}"
    temp_file=$(mktemp)

    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node '
export PATH="/data1/xhuan192/misc/miniconda3/bin/:/$PATH" 
echo $PATH
set -e 
curl -s https://gist.githubusercontent.com/xk-huang/02b5bd8c81327d9d960de7b066148aa6/raw/ae7988a6f6e03cc608d07582e45bdf945d07eb54/250304-ucsc-conda-env.sh | bash
' >> $temp_file) &

    sleep 0.1
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"
```