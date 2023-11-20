from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
torch.cuda.empty_cache()

def train_gpt2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    dataset = load_dataset('text', data_files='C:/Users/AIXI/OneDrive/Desktop/projects/AGENTS/transcripts/combined_transcripts.txt')['train']
    # Split the dataset into train and evaluation sets
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1).values()

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    tokenized_eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./Presults',
        num_train_epochs=2,
        per_device_train_batch_size=1, 
        logging_dir='./Plogs',
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained('./Ptrained_model')

if __name__ == "__main__":
    train_gpt2()
