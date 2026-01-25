# matchminer-ai-training serial summarization branch
Code for training the MatchMiner-AI pipeline.
If you have access to a Linux machine with H100 x 8 GPUs and about a week to spare, you can replicate training by:

Making sure your machine can compile CUDA things:
```
sudo apt update
sudo apt install build-essential python3-dev nvidia-cuda-toolkit
```

Installing uv (https://docs.astral.sh/uv/getting-started/installation/)


Making and activating a venv:
```
uv venv mmai --python=3.12
source mmai/bin/activate
```


Pulling this code and installing dependencies:
```
git clone https://github.com/kenlkehl/matchminer-ai-training
cd matchminer-ai-training
uv pip install -r requirements.txt
```


4. Running the train_all.sh script:
```
bash train_all.sh
```

Note: Training makes heavy use of multiple instances of vllm for efficient parallelized inference. This sometimes causes errors related to race conditions on compile. We try to mitigate this by pre-compiling these at the beginning of the script, but errors during training may still occur and require restarting the script from the last completed step.

Also note: At the time of release (December 2025), we were encountering challenges with gibberish output from gpt-oss-120b when run with vllm on more than one RTX PRO 6000 GPU. Inference on a single RTX PRO 6000 seemed to work well, though.

Original framework and logic were implemented manually; parallelization was vibe-coded with Gemini 2.5 pro and Claude 4.5 Sonnet.
