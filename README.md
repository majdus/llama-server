torchrun --nproc_per_node 1 llama_chat.py

torchrun --nproc_per_node 1 llama_chat_params.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6