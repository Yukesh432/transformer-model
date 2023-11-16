from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_path = './trained_model'
    model, tokenizer = load_model(model_path)

    test_prompt = "once i saw that ghost, then "
    generated_text = generate_text(test_prompt, model, tokenizer)
    print(generated_text)
