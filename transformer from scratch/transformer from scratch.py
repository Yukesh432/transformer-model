# References: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://www.youtube.com/watch?v=ISNdQcPhsts&t=227s
import torch
import torch.nn as nn
import math

# original sentence.....>> input Ids..........>>Embedding of 512
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):

        """
        Initializing the InputEmbedding module.

        Args:
        d_model(int): The dimensionality of the embedding vector.
        vocab_size(int): The size of the vocabularry, i.e. the no. of unique tokens in the input.

        """

        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size

        # create an embedding layer to map input token IDs to dense vectors of size d_model
        self.embedding= nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)* math.sqrt(self.d_model)
    
# Now we build Positional Encoding

# define the class positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_len
        self.dropout= nn.Dropout(dropout)

        #create a matrix of shape(seq_len, d_model)
        pe= torch.zeros(seq_len, d_model)
        # first we create a vector called "position" that will represent the position inside the sentence
        #create a vector of shape(seq_len)
        position= torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_model,2).float()* (-math.log(10000.0)/d_model))
        # apply the sin to even position
        pe[:, 0::2]= torch.sin(position*div_term)
        # apply the cosine to odd position
        pe[:, 1::2]= torch.cos(position*div_term)

        pe= pe.unsqueeze(0)  #(1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x= x +(self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
