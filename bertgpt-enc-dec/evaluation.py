# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import evaluate
# from dataset import prepare_datasets, load_records
# from sklearn.model_selection import train_test_split
# from config import Config
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import evaluate
# from dataset import prepare_datasets
# from config import Config



# # To speed up ROUGE, limit the number of validation samples processed
# MAX_ROUGE_SAMPLES = 10

# def evaluate_perplexity(batch_size: int = Config.batch_size):
#     """
#     Compute perplexity on the validation dataset.
#     """
#     _, val_ds, tokenizer = prepare_datasets()
#     ckpt = Config.output_dir / "checkpoint-470"
#     model = AutoModelForCausalLM.from_pretrained(str(ckpt))
#     model.eval()

#     loader = DataLoader(val_ds, batch_size=batch_size)
#     total_loss = 0.0
#     total_tokens = 0

#     for batch in loader:
#         input_ids = batch['input_ids']
#         with torch.no_grad():
#             outputs = model(input_ids, labels=input_ids)
#         loss = outputs.loss * input_ids.numel()
#         total_loss += loss.item()
#         total_tokens += input_ids.numel()

#     ppl = torch.exp(torch.tensor(total_loss / total_tokens))
#     print(f"Validation Perplexity: {ppl:.2f}")


# def evaluate_rouge():
#     """
#     Generate abstracts on the validation set and compute ROUGE scores.
#     """

#     rouge = evaluate.load('rouge')
#     model = AutoModelForCausalLM.from_pretrained(str(Config.output_dir))
#     tokenizer = AutoTokenizer.from_pretrained(str(Config.output_dir))
#     # Load raw text records and split to get validation samples
#     raw_records = load_records(Config.data_path)
#     _, val_samples = train_test_split(
#         raw_records,
#         train_size=Config.train_split,
#         random_state=Config.seed
#     )
#     # Limit samples to speed up evaluation
#     val_samples = val_samples[:MAX_ROUGE_SAMPLES]

#     preds, refs = [], []
#     for sample in val_samples:
#         text = sample['text']
#         prompt, reference = text.split('\n\nAbstract:')
#         inputs = tokenizer(prompt + '\n\nAbstract:', return_tensors='pt')
#         with torch.no_grad():
#             gen = model.generate(
#                 inputs.input_ids,
#                 max_length=Config.max_length,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         pred = tokenizer.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
#         preds.append(pred.strip())
#         refs.append(reference.strip())

#     scores = rouge.compute(predictions=preds, references=refs)
#     # breakpoint()
#     print(f"Processed {len(val_samples)} samples for ROUGE")
#     for k, v in scores.items():
#         print(f"{k}: {v:.4f}")
        

# if __name__ == '__main__':
#     print("=== Perplexity Evaluation ===")
#     evaluate_perplexity()
#     print("\n=== ROUGE Evaluation ===")
#     evaluate_rouge()

import os
import csv
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jiwer
import evaluate
import matplotlib.pyplot as plt

from dataset import load_records
from config import Config

MAX_EVAL_SAMPLES = 1000  # Limit for faster eval
import os
import csv
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jiwer
import evaluate
import matplotlib.pyplot as plt

from dataset import load_records
from config import Config

MAX_EVAL_SAMPLES = 500
TEMPERATURES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model():
    model = AutoModelForCausalLM.from_pretrained(str(Config.output_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(Config.output_dir))
    model.eval()

    raw_records = load_records(Config.data_path)
    _, val_samples = train_test_split(
        raw_records,
        train_size=Config.train_split,
        random_state=Config.seed
    )
    val_samples = val_samples[:MAX_EVAL_SAMPLES]

    rouge = evaluate.load('rouge')
    smoother = SmoothingFunction()

    all_rows = []
    summary_rows = []

    # Prepare output directory
    os.makedirs("evaluation_scores/plots", exist_ok=True)

    for temp in TEMPERATURES:
        print(f"\n🔥 Evaluating at temperature={temp}")
        preds, refs = [], []
        bleu_scores, meteor_scores, rougeL_scores, wer_scores = [], [], [], []

        for sample in tqdm(val_samples, desc=f"Temp={temp}"):
            prompt = sample['title'].strip()
            reference = sample['abstract'].strip()
            input_text = prompt + '\n\nAbstract:'
            
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            with torch.no_grad():
                gen = model.generate(
                    inputs.input_ids,
                    max_length=Config.max_length,
                    do_sample=True,
                    temperature=temp,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

            ref_tokens = reference.strip().split()
            gen_tokens = generated.strip().split()
            meteor = meteor_score([ref_tokens], gen_tokens)
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoother.method1)
            rougeL = rouge.compute(predictions=[generated], references=[reference])['rougeL']
            wer = jiwer.wer(reference, generated)

            preds.append(generated)
            refs.append(reference)

            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            rougeL_scores.append(rougeL)
            wer_scores.append(wer)

            all_rows.append({
                "temperature": temp,
                "title": prompt,
                "reference": reference,
                "generated": generated,
                "bleu": round(bleu, 4),
                "meteor": round(meteor, 4),
                "rougeL": round(rougeL, 4),
                "wer": round(wer, 4)
            })

        summary_rows.append({
            "temperature": temp,
            "bleu": sum(bleu_scores)/len(bleu_scores),
            "meteor": sum(meteor_scores)/len(meteor_scores),
            "rougeL": sum(rougeL_scores)/len(rougeL_scores),
            "wer": sum(wer_scores)/len(wer_scores)
        })

    # Save all individual results
    csv_path = "evaluation_scores/bertgpt_results.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["temperature", "title", "reference", "generated", "bleu", "meteor", "rougeL", "wer"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n📁 Saved detailed results to {csv_path}")

    # Save summary
    summary_path = "evaluation_scores/bertgpt_summary.csv"
    with open(summary_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["temperature", "bleu", "meteor", "rougeL", "wer"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"📁 Saved summary to {summary_path}")

    # Plotting function
    def plot_metric_from_summary(name):
        x = [row["temperature"] for row in summary_rows]
        y = [row[name] for row in summary_rows]
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.title(f"{name.upper()} vs Temperature")
        plt.xlabel("Temperature")
        plt.ylabel(name.upper())
        plt.grid(True)
        plt.savefig(f"evaluation_scores/plots/{name.lower()}_vs_temperature.png")
        plt.close()

    for metric in ["bleu", "meteor", "rougeL", "wer"]:
        plot_metric_from_summary(metric)

    print("📊 Plots saved in evaluation_scores/plots/")
    print("\n=== Evaluation Complete ===")
    for row in summary_rows:
        print(f"T={row['temperature']:.1f} | BLEU={row['bleu']:.4f} | METEOR={row['meteor']:.4f} | ROUGE-L={row['rougeL']:.4f} | WER={row['wer']:.4f}")

if __name__ == '__main__':
    evaluate_model()
