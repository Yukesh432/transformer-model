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

Encoder Architecture: Encoder-only architecture is used for those task which requires the understanding of the overall context in the given input sentences. The function of encoder is to represent the input(typically textual input, but can be others like images, audio, vidoe, sensor signals ,etc.) into contextual numerical n-dimensional vector representation.   
