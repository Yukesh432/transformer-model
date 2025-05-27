# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
# )
# from config import Config
# from dataset import prepare_datasets


# def main():
#     train_ds, val_ds, tokenizer = prepare_datasets()

#     model = AutoModelForCausalLM.from_pretrained(Config.model_name)
#     model.resize_token_embeddings(len(tokenizer))

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False
#     )

#     args = TrainingArguments(
#         output_dir=str(Config.output_dir),
#         overwrite_output_dir=True,
#         num_train_epochs=Config.epochs,
#         per_device_train_batch_size=Config.batch_size,
#         per_device_eval_batch_size=Config.batch_size,
#         learning_rate=Config.learning_rate,
#         weight_decay=Config.weight_decay,
#         logging_steps=Config.logging_steps,
#         do_eval= True,
#         eval_steps=Config.eval_steps,
#         save_steps=Config.save_steps,
#         gradient_accumulation_steps=Config.gradient_accumulation_steps,
#         seed=Config.seed,
#         fp16=torch.cuda.is_available(),
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#     )

#     trainer.train()
#     trainer.save_model(str(Config.output_dir))
#     tokenizer.save_pretrained(str(Config.output_dir))


# if __name__ == '__main__':
#     main()

import torch
from transformers import (
    EncoderDecoderModel,
    BertTokenizer,
    GPT2Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from dataset import prepare_datasets
from config import Config

def main():
    # Load tokenizers
    encoder_tokenizer = BertTokenizer.from_pretrained(Config.encoder_model)
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(Config.decoder_model)
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    # Load datasets
    train_ds, val_ds = prepare_datasets(encoder_tokenizer, decoder_tokenizer)

    # Initialize model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        Config.encoder_model, Config.decoder_model
    )
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.eos_token_id = decoder_tokenizer.eos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = Config.max_length
    model.config.no_repeat_ngram_size = 3

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=decoder_tokenizer, model=model, padding=True
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.num_train_epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        logging_steps=Config.logging_steps,
        # evaluation_strategy="steps",
        eval_steps=Config.eval_steps,
        save_steps=Config.save_steps,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        seed=Config.seed,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=decoder_tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(Config.output_dir)
    decoder_tokenizer.save_pretrained(Config.output_dir)

if __name__ == "__main__":
    main()
