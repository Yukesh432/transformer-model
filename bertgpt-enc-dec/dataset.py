import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from config import Config

def load_records(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    records = []
    for entry in raw.values():
        title = entry['title']
        abstract = entry['abstract']
        records.append({"title": title, "abstract": abstract})
    return records


def prepare_datasets(encoder_tokenizer, decoder_tokenizer):
    records = load_records(Config.data_path)
    train_samples, val_samples = train_test_split(
        records,
        train_size=Config.train_split,
        random_state=Config.seed
    )

    def tokenize_fn(batch):
        # Title as input to Encoder
        inputs = encoder_tokenizer(
            batch['title'],
            truncation=True,
            max_length=Config.max_length,
            padding='max_length'
        )

        # Abstract as target labels for Decoder
        targets = decoder_tokenizer(
            batch['abstract'],
            truncation=True,
            max_length=Config.max_length,
            padding='max_length'
        )

        inputs['labels'] = targets['input_ids']
        return inputs

    train_ds = Dataset.from_list(train_samples).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_list(val_samples).map(tokenize_fn, batched=True)

    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_ds, val_ds



if __name__ == '__main__':
    train_ds, val_ds, tokenizer = prepare_datasets()
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Show a tokenized example
    example = train_ds[0]
    print(example)
    print(f"Input IDs length: {len(example['input_ids'])}")
    print(f"Labels length: {len(example['labels'])}")
