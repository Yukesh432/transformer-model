import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
torch.cuda.empty_cache()

def train_gpt2():
    # Ensure CUDA is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model, and send the model to the device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model= GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2).to(device)

    # texts= ['I am feeling good', 'this is not a coke', 'i watch old movies']
    # labels= [1,0,1]

    # dataset= load_dataset('text', data_files='v0.1_combined_transcripts.txt')['train']
    dataset= load_dataset('imdb', split='train')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

    # dataset= Dataset.from_dict({'text': texts, 'labels': labels})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    # tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define Data Collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # for evaluation data
    eval_data= load_dataset('imdb', split='test')
    eval_tokenized_data= dataset.map(tokenize_function, batched=True)
    eval_tokenized_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=2,  # Gradient accumulation
        logging_dir='./logs',
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_tokenized_data
        # data_collator=data_collator,
    )

    # Train and save the model
    trainer.train()
    model.save_pretrained('./sentiment_trained_model')

if __name__ == "__main__":
    
    train_gpt2()
