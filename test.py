import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import requests

# Hyperparameters
batch_size = 32 # independent sequences to run in parallel
block_size = 64 # max_context when making the predictions
max_iters = 3000
eval_interval = 300
learning_rate = 3e-3
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2


# -------------- Get the Data ------------
url = "https://raw.githubusercontent.com/karpathy/charrnn/master/data/tinyshakespeare/input.txt"
data_path = Path("data/input.txt")

def get_data(url, data_path):
    if data_path.is_file():
        print(f"The directory {data_path} already exists.")
    
    else:
        Path("data").mkdir(parents = True, exists_ok = True)
        with open(data_path, "wb") as f:
            request = requests.get(url)
            f.write(request.content)
            
    with open(data_path, "r") as f:
        text = f.read()
    return text

text = get_data(url, data_path)


# ----------------- Encoder and Decoder -------
characters = set(text)
vocab_size = len(characters)

# string to integer mapping
stoi = {ch:i for i,ch in characters in enumerate(characters)}
# integer to string mapping
itos = {i:ch for i,ch in characters in enumerate(characters)}

# take the integer and generate the integers encoding
encode = lambda s: [stoi[c] for c in s]
# take the encoded integers and return the string
decode = lambda l: "".join(itos[i] for i in l)


# ------------------- Train and Test Split ---------
# change the text to integers encoding and then to the tensor 
data = torch.tensor(encode(text), dtype = torch.long)

# 90% train and 10% test
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ------------- Creating a data loader -----------
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data in split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in idx])
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
    x , y = x.to(device), y.to(device)
    return x,y


# --------------- Creating a loss estimating function ------
def estimate_loss():
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, y = get_batch(split)
                logits, loss = model(X, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# ----------------- creating heads ---------------------
class Head(nn.Module):
    """ Definition for a single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        # to create a lower triangular matrix
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input_size (Batch, Time_step, Channels)
        # outout_size (Batch, Time_step, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute the attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5) # (B,T,hs) * (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("inf")) # (B,T,T)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        # Get the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)

        out = wei @ v # (B,T,T) * (B,T,hs) -> (B,T,hs)


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # project the output from each heads to output a tensor as same shape as the input
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # h(x) returns -> (B,T,hs) 
        out = torch.cat([h(x) for h in self.heads], dim = -1) # out has shape (B,T,hs*num_heads)
        out = self.dropout(self.proj(out)) # project (B,T,hs*num_heads) -> (B,T,n_embd)
        return out

# --------------- FeedForward layer ---------------
class FeedForward(nn.Module):
    """ Simple Linear Layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd) # the original paper has innerdimension 4 times dmodel
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Drop(dropout)
        )

    def forward(self, x):
        return self.nex(x)



# -------------------- Creating a Transformer Block ---------------
class Block(nn.Module):
    """ Transformer Block: Attention followed by FeedFroward 
    (communication followed by computation)"""

    def __init__(self,n_embd, n_heads):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LinearNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # communicate (normalize x) add resudial connection
        x = x + self.ffwd(self.ln2(x)) # compute(normalize(x_from_previous)) add resudial connection
        return x 



















