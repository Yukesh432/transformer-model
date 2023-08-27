# Transformer Architecture

Transformer Architecture incorporate the main three parts:

a. The encoder-decoder framework

b. Attention Mechanism

c. Transfer Learning

The encoder module encodes the input sequence , preserving the positional information into n-dimensional vector called embeddings. The decoder decodes that very input embeddings to text using different decoding strategies(i.e Greedy search, Beam search, etc.)

## Different Types of Transformer architecture

- Encoder-Decoder Architecture
- Encoder-only Architecture
- Decoder-only Architecture

**Encoder Architecture:** The function of encoder is to represent the input(typically textual input, but can be others like images, audio, vidoe, sensor signals ,etc.) into contextual numerical n-dimensional vector representation. Encoder-only architecture is used for those task which requires the understanding of the overall context in the given input sentences.
The major operation performed in Encoder architecture is:

a. Positional Encoding

b. Multiheaded- Attention

c. Layer Normalization

d. Feed Forward Connection   

*Usecases:* Sentence Classification, Sentiment Analysis, Question Answering, etc. 

*Example:* BERT, DISTILBERT, ELECTRA etc. 

**Decoder Architecture:** The function of decoder architecture is to generate the output text(eg. text-generation). The pretraining objective for these models are mainly autoregressive language modeling and masked language modeling.

*Example:* Chatgpt, gpt-2, gpt-3 , PaLM, etc. 

## Emergent Cababilities in LLMS:
Models abilities that are not present in small scale LM, but on scaling, emerges. first was seen on gpt2-xl model. 
Eg. 3-digit addition and subtraction on GPT3-13B model.

#### Major Emergent Capabilities includes:

**a. In-context Learning:** LLM's ability to learn task without required to finetune on that specific task. Earlier models before gpt3 (eg. gpt1, gpt2) required finetuning on downstream task such as summarization, question-answering, NER etc. But here we provide the sample task with few-shot demonstration along with the prompt.

**b. Instruction Following:** LLM's gained the ability to follow instruction better to perform specific task. Task that are given to LLMs with specific instruction tends to perform better than without the proper instructions.

**c. Step By Step Reasoning:** This prompting method is usefoul for task that requires multi-step reasoning. Model performance imporves as we go with step by step reasoning process. Rather than interacting with straight Question and Answer approach, if we break down the task into sub task, the model tends to perform better.
Eg. Chain-of-Thought(CoT) prompting improved the arithmetic reasoning benchmarks


## Reference papers:

- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)
- [Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682.pdf)
- [A Survey on In-context Learning](https://arxiv.org/pdf/2301.00234.pdf)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)