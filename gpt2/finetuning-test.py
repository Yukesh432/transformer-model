import random

# Synthetic dataset creation
def generate_synthetic_data(num_samples=100):
    positive_phrases = ["I love this!", "Absolutely fantastic.", "So happy with this.", "Incredible experience.", "Very satisfying."]
    negative_phrases = ["I hate this.", "Absolutely terrible.", "So disappointed.", "Horrible experience.", "Very unsatisfying."]
    
    data = []
    for _ in range(num_samples):
        if random.random() < 0.5:  # Randomly choose between positive and negative
            data.append((random.choice(positive_phrases), 1))  # Positive sentiment
        else:
            data.append((random.choice(negative_phrases), 0))  # Negative sentiment
    return data

# Generate and save the dataset
synthetic_dataset = generate_synthetic_data()
print(synthetic_dataset)


# finetune.py
# finetune.py

# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import Dataset

# def train_gpt2():
#     # Ensure CUDA is available and set the device
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load tokenizer and model, and send the model to the device
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.pad_token = tokenizer.eos_token
#     model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

#     # Prepare the dataset
#     synthetic_dataset = generate_synthetic_data()  # Assuming generate_synthetic_data() is defined elsewhere
#     dataset = Dataset.from_dict({'text': [item[0] for item in synthetic_dataset]})

#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     tokenized_dataset = tokenized_dataset.remove_columns(['text'])
#     tokenized_dataset.set_format('torch', columns=['input_ids'])

#     # Define Data Collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=3,
#         per_device_train_batch_size=1,  # Reduced batch size
#         gradient_accumulation_steps=2,  # Gradient accumulation
#         logging_dir='./logs',
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         data_collator=data_collator,
#     )

#     # Train and save the model
#     trainer.train()
#     model.save_pretrained('./trained_model')

# if __name__ == "__main__":
#     train_gpt2()
