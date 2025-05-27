# import sys
# import os
# import json
# import csv
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from config import Config

# # Define the list of temperatures to experiment with
# TEMPERATURES = [0.7, 1.0, 1.3]


# def generate_abstract(
#     title: str,
#     max_length: int = 150,
#     temperature: float = 1.0,
#     top_p: float = 0.95,
#     num_return_sequences: int = 1
# ):
#     """
#     Generate abstract(s) for a given title using the fine-tuned model,
#     with specified sampling temperature and nucleus sampling (top_p).
#     """
#     prompt = f"Title: {title} \n Abstract:"
#     tokenizer = AutoTokenizer.from_pretrained(str(Config.output_dir))
#     model = AutoModelForCausalLM.from_pretrained(str(Config.output_dir))
#     model.eval()

#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs.input_ids.to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             max_length=input_ids.shape[-1] + max_length,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             num_return_sequences=num_return_sequences,
#         )

#     abstracts = []
#     for out in outputs:
#         generated = out[input_ids.shape[-1]:]
#         text = tokenizer.decode(generated, skip_special_tokens=True)
#         abstracts.append(text.strip())
#     return abstracts


# def main():
#     args = sys.argv[1:]
#     if not args:
#         print(
#             "Usage: python inference.py \"Your title here\"",
#             "or python inference.py titles.txt or titles.json"
#         )
#         sys.exit(1)

#     # Load titles from file or CLI arguments
#     if len(args) == 1 and os.path.isfile(args[0]):
#         path = args[0]
#         if path.endswith('.json'):
#             with open(path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             if isinstance(data, list):
#                 titles = data
#             elif isinstance(data, dict):
#                 # assume dict of id->title
#                 titles = list(data.values())
#             else:
#                 raise ValueError("JSON file must contain a list or dict of titles.")
#         else:
#             with open(path, 'r', encoding='utf-8') as f:
#                 titles = [line.strip() for line in f if line.strip()]
#     else:
#         titles = args

#     # Collect results across temperatures
#     results = []
#     for title in titles:
#         for temp in TEMPERATURES:
#             abs_list = generate_abstract(
#                 title,
#                 max_length=Config.max_length,
#                 temperature=temp,
#                 top_p=0.95,
#                 num_return_sequences=1
#             )
#             for abs_text in abs_list:
#                 results.append({
#                     'title': title,
#                     'temperature': temp,
#                     'abstract': abs_text
#                 })

#     # Save to CSV
#     output_file = 'generated_output.csv'
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['title', 'temperature', 'abstract']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in results:
#             writer.writerow(row)

#     print(f"Generated abstracts (with temperatures {TEMPERATURES}) saved to {output_file}")

# if __name__ == '__main__':
#     main()

import sys
import os
import json
import csv
import random
import torch
from evaluate import load as load_metric
from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer
from config import Config
from dataset import prepare_datasets

TEMPERATURES = [0.7, 1.0, 1.3]
SAMPLE_RATIO = 0.001  # Evaluate on 10% subset

# Load Encoder-Decoder Model
model = EncoderDecoderModel.from_pretrained(Config.output_dir)
model.eval()

# Device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Tokenizers
encoder_tokenizer = BertTokenizer.from_pretrained(Config.encoder_model)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(Config.decoder_model)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
breakpoint()
# Update model config for decoder tokens
model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
model.config.eos_token_id = decoder_tokenizer.eos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id

# Load Metrics
rouge = load_metric("rouge")
wer_metric = load_metric("wer")
bertscore = load_metric("bertscore")

def generate_abstract(title, temperature=1.0, max_length=Config.max_length):
    inputs = encoder_tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
        )

    decoded = decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded[0].strip()

def evaluate_model():
    # Load val dataset
    _, val_ds = prepare_datasets(encoder_tokenizer, decoder_tokenizer)

    # Subsample val dataset
    subset_size = int(len(val_ds) * SAMPLE_RATIO)
    indices = random.sample(range(len(val_ds)), subset_size)
    val_ds = val_ds.select(indices)
    print(f"Evaluating on {subset_size} samples out of {len(val_ds)} total.")

    results = []
    preds = []
    refs = []

    for sample in val_ds:
        title_ids = sample['input_ids']
        title_text = encoder_tokenizer.decode(title_ids, skip_special_tokens=True)

        reference = decoder_tokenizer.decode(sample['labels'], skip_special_tokens=True)

        for temp in TEMPERATURES:
            prediction = generate_abstract(title_text, temperature=temp)
            results.append({
                'title': title_text,
                'reference_abstract': reference,
                'temperature': temp,
                'generated_abstract': prediction
            })

            # For metrics (we're only evaluating temp=1.0 for metrics simplicity)
            if temp == 1.0:
                preds.append(prediction)
                refs.append(reference)

    # Compute Metrics
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    wer_score = wer_metric.compute(predictions=preds, references=refs)
    bertscore_scores = bertscore.compute(predictions=preds, references=refs, lang="en")

    print("\nEvaluation Metrics:")
    print("WER:", wer_score)
    print("ROUGE:", rouge_scores)
    print("BERTScore:", bertscore_scores)

    # Save Generated Abstracts CSV
    output_file = "evaluation_results.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'reference_abstract', 'temperature', 'generated_abstract']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Generated Abstracts saved to {output_file}")

    # Save Evaluation Metrics CSV (Summary)
    metrics_file = "evaluation_scores.csv"
    with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'metric': 'WER', 'value': wer_score})

        for key, score in rouge_scores.items():
            writer.writerow({'metric': f'ROUGE-{key}', 'value': score.mid.fmeasure})

        writer.writerow({'metric': 'BERTScore (Precision)', 'value': sum(bertscore_scores['precision']) / len(bertscore_scores['precision'])})
        writer.writerow({'metric': 'BERTScore (Recall)', 'value': sum(bertscore_scores['recall']) / len(bertscore_scores['recall'])})
        writer.writerow({'metric': 'BERTScore (F1)', 'value': sum(bertscore_scores['f1']) / len(bertscore_scores['f1'])})

    print(f"Evaluation Metrics saved to {metrics_file}")

if __name__ == "__main__":
    evaluate_model()
