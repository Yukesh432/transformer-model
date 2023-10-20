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

        super().__init__()  # this calls the constructor of the base class( i.e. nn.Module)
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

        # create a matrix of shape(seq_len, d_model)
        pe= torch.zeros(seq_len, d_model)
        # first we create a vector called "position" that will represent the position inside the sentence
        # create a vector of shape(seq_len)
        position= torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # compute the div_term for positional encoding 
        div_term= torch.exp(torch.arange(0, d_model,2).float()* (-math.log(10000.0)/d_model))
        # apply the sin to even position
        pe[:, 0::2]= torch.sin(position*div_term)
        # apply the cosine to odd position
        pe[:, 1::2]= torch.cos(position*div_term)

        pe= pe.unsqueeze(0)  # Expand the matrixx to shape (1, seq_len, d_model)
        # register the positional encodings(pe) tensor as a buffer in the module's state
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Perform the forward pass of the PositionalEncoding module.

        Args:
        x(torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        #add the positiional encoding to the input tensor along the sequence length dimension
        x= x +(self.pe[:, :x.shape[1], :]).requires_grad(False)

        # apply the drropout for regularization
        return self.dropout(x)

# Layer Normalization

class LayerNormalization(nn.Module):

    def __init__(self, eps: float= 10 **-6):
        super().__init__()
        self.eps= eps
        self.alpha= nn.Parameter(torch.ones(1))  # multiplied
        self.bias= nn.Parameter(torch.zeros(1))   # added
        
    def forward(self, x):
        mean= x.mean(dim= -1, keepdim= True)
        std= x.std(dim= -1, keepdim= True)
        return self.alpha * (x- mean) / (std + self.eps) + self.bias
    

# feed forward class

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
        super().__init__()
        self.linear_1= nn.Linear(d_model, d_ff)  #w1 and B1
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        #We have a bathc seq of length (batch, sequence_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
# Multi-Headed Attention Block....

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float) ->None:
        super().__init__()
        self.d_model= d_model
        self.h= h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k= d_model // h
        self.w_q= nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)

        self.w_o= nn.Linear(d_model, d_model)
        self.dropout= nn.Dropout(dropout)

    # method to calculate attention

    @staticmethod
    def attention(query, key, value, mask, dropout= nn.Dropout):
        d_k= query.shape[-1]

        attention_scores= (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #masking
        if mask is not None:
            attention_scores.masked_fill_(mask ==0, -1e9)
        attention_scores= attention_scores.softmax(dim = -1)  #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores= dropout(attention_scores)

        return (attention_scores @ value), attention_scores 

    def forward(self, q, k, v, mask):
        query= self.w_q(q)
        key= self.w_k(k)
        value= self.w_v(v)

        query= query.view(query.shape[0], query.shape[1], self.d_k).transpose(1,2)
        key= key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value= value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores= MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch, h, seq_len, d_k)--->(batch, seq_len, h, d_k)-->(batch, seq_len, d_model)
        x= x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h* self.d_k)

        # (batch, seq_len, d_model)--->(batch, seq_len,  d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float)-> None:
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm= LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    
    




