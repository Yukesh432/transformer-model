class Config:
    # Data Paths
    data_path = "data/training_data_10000.json"
    output_dir = "/home/boltzman/Desktop/finetune-gpt/bertgpt-enc-dec/output"

    # Model Names
    encoder_model = "bert-base-uncased"
    decoder_model = "gpt2"

    # Training Hyperparameters
    max_length = 256
    train_split = 0.9
    seed = 42

    batch_size = 4
    eval_batch_size = 4
    num_train_epochs = 3
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_steps = 500
    gradient_accumulation_steps = 1
    fp16 = True  # Enable for faster training on compatible GPUs

    # Generation Config
    decoder_start_token_id = None
    eos_token_id = None
    pad_token_id = None

    # Evaluation Metrics
    compute_wer = True
    compute_rouge = True
    compute_bertscore = True

    # Logging & Saving
    logging_dir = "logs/bert-gpt2-finetune-v1/"
    save_steps = 500
    eval_steps = 500
    logging_steps = 100
    save_total_limit = 2
    logging_first_step = True
    load_best_model_at_end = True
    early_stopping_patience = 3

    # Misc
    experiment_name = "bert-gpt2-finetune-v1"



from transformers import AutoTokenizer

if __name__ == "__main__":
    # Load tokenizer to set decoder tokens properly
    tokenizer = AutoTokenizer.from_pretrained(Config.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have pad_token by default

    # Set token IDs in Config dynamically
    Config.decoder_start_token_id = tokenizer.bos_token_id
    Config.eos_token_id = tokenizer.eos_token_id
    Config.pad_token_id = tokenizer.pad_token_id

    # Sanity check
    print(f"Experiment Name: {Config.experiment_name}")
    print(f"Encoder Model: {Config.encoder_model}")
    print(f"Decoder Model: {Config.decoder_model}")
    print(f"Max Length: {Config.max_length}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Decoder Start Token ID: {Config.decoder_start_token_id}")
    print(f"EOS Token ID: {Config.eos_token_id}")
    print(f"PAD Token ID: {Config.pad_token_id}")
    print(f"Compute Metrics -> WER: {Config.compute_wer}, ROUGE: {Config.compute_rouge}, BERTScore: {Config.compute_bertscore}")
