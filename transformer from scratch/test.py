import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# # Define a linear layer
# linear_layer = nn.Linear(in_features=10, out_features=5)

# # Example input
# input_tensor = torch.randn(1, 10)  # A random tensor with shape (1, 10)
# print(input_tensor)

# # Forward pass through the linear layer
# output = linear_layer(input_tensor)
# print(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Number of examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled Dot-Product Attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention

# Example use
embed_size = 256
heads = 8
attention = MultiHeadAttention(embed_size, heads)

print(attention)

def plot_attention(attention, save_path="attention.png"):
    # Assuming attention is a square matrix
    plt.figure(figsize=(10,10))
    plt.matshow(attention, cmap='viridis')
    plt.colorbar()
    plt.savefig(save_path)
    plt.show()

# Generating a random tensor for demonstration
input_tensor = torch.rand(1, 10, embed_size)  # (batch_size, sequence_length, embed_size)

# Get output and attention
output, attention_matrix = attention(input_tensor, input_tensor, input_tensor, None)

# Visualize the first head's attention (for simplicity)
plot_attention(attention_matrix[0, 0].detach().numpy(), "first_head_attention.png")
