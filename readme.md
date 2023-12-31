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
It is mainly used for NLU tasks in NLP.
The major operation performed in Encoder architecture is:

a. Positional Encoding

b. Multiheaded- Attention

c. Layer Normalization

d. Feed Forward Connection   

*Usecases:* Sentence Classification, Sentiment Analysis, Question Answering, etc. 

*Example:* BERT, DISTILBERT, ELECTRA etc. 

**Decoder Architecture:** The function of decoder architecture is to generate the output text(eg. text-generation). The pretraining objective for these models are mainly autoregressive language modeling and masked language modeling.

Example of Autoregressive LM: 
- The sun rise in the ...
- One upon a time there was ...

Example of Masked LM :
- The [MASK] brown [MASK] jump over a [MASK] dog.
- She was having a pepperoni [MASK] tonight.

*Decoder-only architecture example:* Chatgpt, gpt-2, gpt-3 , PaLM, etc. 

## Emergent Cababilities in LLMS:

Models abilities that are not present in small scale LM, but on scaling, emerges. first was seen on gpt2-xl model. 
Eg. 3-digit addition and subtraction on GPT3-13B model.

### Major Emergent Capabilities includes:

**a. In-Context Learning:** LLM's ability to learn task without required to finetune on that specific task. Earlier models before gpt3 (eg. gpt1, gpt2) required finetuning on downstream task such as summarization, question-answering, NER etc. But here we provide the sample task with few-shot demonstration along with the prompt.

The following approach is related to how at the **pre-training level**, can we leverage the ICL capabilities by using different techniques.

#### Supervised and Self-Supervised In-Context Training:
- Includes different supervised finetuning techniques by constructing ICL training dataset.eg. MetaICL, PICL
- However these approach does well to some extent, but quickly reach the saturation performance as training data increases.

Apart from pretraining, the another area of work focuses on the ICL capabilities during **inference time**. Using different techniques for demonstrating in-context task helps LLMs to generate desired output,i.e. different prompting strategies

#### Demonstration Designing:

It includes:

- **Demonstration Organization:** When we have set of example data, demonstration organization involves picking some examples and deciding their order. In demonstration organization , we have demonstration selection and ordering. While considering ICL capabilites, it is important to consider which examples are good enough examples for incontext learning scenarios. Different supervised method and unsupervised methods are used to select the appropriate task examples for ICL. Once we find out the relevent demonstration examples, the ordering of these examples also matters to get the desired output from LLMs while using ICL. The experiment related to demonstration organization was performed on GPT-2 and GPT-3 model and its varients.

**Experimentation Task:**
Task and their corresponding dataset:
- text classification: **SST-2** dataset, 6-way question classification **TREC**, textual entailment **CB** and binary **RTE** SuperGLUE, **AGNews**, **DBPedia**.  
- Fact retrieval: **LAMA**
- Information Extraction: **ATIS**, **MIT Movies** trivia10k13


Findings:

a. GPT-3 accuracy varies across different training examples, permutations, and prompt formats.

Why there is variations?

**---Majority Label Bias**
- GPT3 is biased towards answers that are frequent in the prompt. Eg. when a text classification prompt has class imbalance in few shot setting, i.e. if we have something like :

PROMPT:


*Review: Thie pizza is good enough for me*

*Sentiment: Positive*

*Review: I dont like that movie much*

*Sentiment: Negative*

*Review: This is a very congested place to live in*

*Sentiment: Negative*

*Review: Your joke doesnt make me laugh. It just made me feel sad.*

*Sentiment: Negative*

*Review: The weather is nice. We're going hike.*

*Sentiment:*

Here in the last line , the probable output sentiment is "Negative". Why?--> Due to imbalance class

**--- Recency Bias:**

**--- Common Token Bias:**

*Contextual Calibration Helps to improve the variation seen in this setting.*


- **Demonstration Formatting:** Demonstration formatting deals with how we present the task sample for few-shot learning. A common way is to concatenate the user query and task sample with a template. Differenet prompt engineering techniques deals with how do we demonstrate the task. We can portays the dample data in a form of yes/no question, or chain-of-thought reasoning format, etc.

Example:


**b. Instruction Following:** LLM's gained the ability to follow instruction better to perform specific task. Task that are given to LLMs with specific instruction tends to perform better than without the proper instructions. This ability is derived from in-context learning(ICL) capabilites of LLMs. Training LLM on Instruction tuned dataset with task description improves the in-context learning abilities.

**c. Step By Step Reasoning:** This prompting method is useful for task that requires multi-step reasoning. Model performance imporves as we go with step by step reasoning process. Rather than interacting with straight Question and Answer approach, if we break down the task into sub task, the model tends to perform better.
Eg. Chain-of-Thought(CoT) prompting improved the arithmetic reasoning benchmarks

### Why In-context Learning Works?

Studies have shown that, the ability to perform in-context learning in transformer models without changing the weights depends on various factors. Experimenting the workings and ICL abilitis of Transformer architecture on a controlled setup has yield promising results about the theoritical understanding of why such thing works, yet no concrete theory exists. However, following experiments and studies have helped us to understand a bit more about the ICL properties. 

- In-context learning capabilities in transformer models depends upon training data distributional properties.

- When pretraining distribution contains the sufficient amount of compositional structure of linguistics. 

### Model Compression Techniques:

**a. Knowledge Distillation**

- Standard Knowledge Distillation

- Emergent Abilities(EA)-based Knowledge Distillation

**b. Prunning**

- Structured Prunning

- Unstructured Prunning


**c. Quantization**

- Quantization-aware Training

- Quatization-aware finetuning

- Post-training Quantization

**d. Low-rank Factorization**

### Different Attention Mechanishm in LLMs

- Self-Attention
- Masked Self-Attention
- Multiheaded- Attention
- Cross-attention
- Flash attention
- Sparse Attention


### Different Positional Encoding Scheme

- Absolute Positional Encoding
- Relative Positional Encoding
- Rotatory Positional Encoding

### Evaluation metrics for LLMs:

- Perplexity: It is a measure of how well the language model good at predicting the next word. Lower the perplexity score , the better is the model's ability to predict the next word accurately.
Mathematically it is given as:

        Perplexity= 2^H

- Human Evaluation
- BLEU score: This metric is used for standard machine translation task

- ROGUE Score: Recall-Oriented Understudy for Gisting Evaluation(ROGUE)
ROGUE metric is case insensative, i.e uppercase and lowercase letters are treated same way. 

### Benchmarking Datasets:

- GLUE
- SuperGLUE
- LAMBADA
- StrategyQA
- SQuAD
- BIG-Bench

### Formal Lingustic Competence and Functional Linguistic Competence

**a. Formal Lingustic Capabilities**
- It includes morphology, phonology, syntax, semantics, core lingustic knowledge

**b. Functional Linguistic Capabilities**
- Formal Reasoning, Situational modeling, communicative intents


#### Distinction between formal and functional linguistic Competence



## Reference papers:

- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)
- [Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682.pdf)
- [Data Distributional Properties Drive Emergent In-Context Learning in Transformers](https://arxiv.org/pdf/2205.05055.pdf)
- [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/pdf/2208.01066)
- [A Survey on In-context Learning](https://arxiv.org/pdf/2301.00234.pdf)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262.pdf)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/pdf/2012.14913.pdf)
- [A Theory of Emergent In-Context Learning as Implicit Structure Induction](https://arxiv.org/pdf/2303.07971.pdf)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities](https://arxiv.org/pdf/2308.12833.pdf)
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/pdf/2308.07633.pdf)
- [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/pdf/1905.00537v3.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165v4.pdf)
- [Optimal Brain Damange](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)
- [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2307.15217.pdf)
- [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/pdf/2308.11432.pdf)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
- [Model Compression](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
- [A Survey of Transformers](https://arxiv.org/pdf/2106.04554.pdf)
- [ATTENTION, PLEASE! A SURVEY OF NEURAL ATTENTION MODELS IN DEEP LEARNING](https://arxiv.org/pdf/2103.16775.pdf)
- [Large Language Models as General Pattern Machines](https://arxiv.org/pdf/2307.04721.pdf)
- [The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain](https://arxiv.org/abs/2305.07141)
- [Verbal Disputes](https://consc.net/papers/verbal.pdf)
- [The Debate Over Understanding in AI’s Large Language Models](https://arxiv.org/pdf/2210.13966.pdf)
- [Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks](https://arxiv.orgpdf/2307.02477v2.pdf)
- [GPT Can Solve Mathematical Problems Without a Calculator](https://arxiv.org/pdf/2309.03241.pdf)
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)
- [LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409.pdf)
- [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf)
- [A General Survey on Attention Mechanisms in Deep Learning](https://arxiv.org/pdf/2203.14263.pdf)
- [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
- [Parameter-Efficient Transfer Learning for NLP](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)
- [Secrets of RLHF in Large Language Models Part I: PPO]()
- [NOAM CHOMSKY - SYNTACTIC STRUCTURES (1957)](https://vdocument.in/noam-chomsky-syntactic-structures-1957.html?page=32)
- [Training language models to follow instructions with human feedback](https://browse.arxiv.org/pdf/2203.02155.pdf)
- [LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY](https://browse.arxiv.org/pdf/2303.03846.pdf)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://browse.arxiv.org/pdf/2202.12837.pdf)
- [Common arguments regarding emergent abilities](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities)
- [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/pdf/2206.04615.pdf)
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)
- [Are Emergent Abilities of Large Language Models a Mirage?]()
- [PaLM 2 Technical Report](https://arxiv.org/pdf/2305.10403v3.pdf)
- [The Reversal Curse: LLMs trained on “A is B” fail to learn “B is A”](https://arxiv.org/pdf/2309.12288v1.pdf)
- [Eight Things to Know about Large Language Models](https://arxiv.org/pdf/2304.00612.pdf)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)
- [PATHWAYS: ASYNCHRONOUS DISTRIBUTED DATAFLOW FOR ML](https://arxiv.org/pdf/2203.12533.pdf)
- [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221.pdf)
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/pdf/2102.09690.pdf)
- [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)
- [Modular Deep Learning](https://arxiv.org/pdf/2302.11529.pdf)
- [SEEING IS BELIEVING: BRAIN-INSPIRED MODULAR TRAINING FOR MECHANISTIC INTERPRETABILITY]()
- [Progress measures for grokking via mechanistic interpretability](https://www.semanticscholar.org/readerc9ef79d6d47c90722a10c32c64c752eb0343fd61)
- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://openreview.net/pdf?id=nZeVKeeFYf9)
- [Learning representations by backpropagating errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)
- [LLMs for Knowledge Graph Construction and Reasoning:Recent Capabilities and Future Opportunities](https://arxiv.org/pdf/2305.13168.pdf)
-[GPT-4 Doesn’t Know It’s Wrong: An Analysis of Iterative Prompting for Reasoning Problems](https://arxiv.org/pdf/2310.12397.pdf)
- [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232.pdf)
-[AI for Mathematics: A Cognitive Science Perspective](https://arxiv.org/pdf/2310.13021.pdf)
- [Levels of AGI: Operationalizing Progress on the Path to AGI]()
- [Gemini: A Family of Highly Capable Multimodal Models](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
- [A New Approach to Linear Filtering and Prediction Problems1](https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf)
