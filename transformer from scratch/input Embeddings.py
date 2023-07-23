# original sentence.....>> input Ids..........>>Embedding of 512
import torch
import torch.nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size