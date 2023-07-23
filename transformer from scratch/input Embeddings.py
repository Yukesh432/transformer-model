# original sentence.....>> input Ids..........>>Embedding of 512
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size

        #mapping betwn numbers to vector 
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