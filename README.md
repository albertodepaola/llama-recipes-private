# Llama Guard standalone sample

## Requirements
Access to the Llama Guard models

1. Clone the repo and `cd`` into the folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run a prompt through the model with this command: `python llamaguard_demo/test_safety_check.py  --ckpt_dir <path_to_model>  --tokenizer_path <path_to_tokenizer>  --max_seq_len 2048 --max_batch_size 6 --prompt "User: <prompt>"`