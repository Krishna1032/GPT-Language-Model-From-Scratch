import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200

# -------------- Get the Data --------------

# function to get the data from the url to the data_path
def get_data(url, data_path):
    if data_path.is_file():
        print(f"The file {data_path} already exists.")
    
    else:
        Path("data").mkdir(parents = True, exist_ok = True)
        with open(data_path, "wb") as f:
            request = requests.get(url)
            f.write(request.content)
    with open(data_path, "r") as f:
        data = f.read()
    return data

url = "https://raw.githubusercontent.com/karpathy/charrnn/master/data/tinyshakespeare/input.txt"
data_path = Path("data/input.txt")
text = get_data(url, data_path)

# the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ---------------- Encoder and Decoder -----------------

# create a mapping of character to integers(index) for encoder and decoder
encoder_mapping = {ch:i for i,ch in enumerate(chars)}
decoder_mapping = {i:ch for i,ch in enumerate(chars)}

# takes a string and outputs list of integers according to our mapping
encode = lambda s: [encoder_mapping[c] for c in s]

# takes a list of integers and maps it to string according to its mapping
decode = lambda l: "".join([decoder_mapping[i] for i in l]) 



# ---------------- Train and test splits ------------------
data = torch.tensor(encode(text), dtype=torch.long)

# first 90% will be train, rest val
n = int(0.9 * len(data))  
train_data = data[:n]
val_data = data[n:]


# ----------------- data loading ---------------------------
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    
    # random_idx from 0 to len(data)-blocksize because extreme value = last_idx + block_size (which equals to len)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ------------------ Loss function ---------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ------------------- Create a Simple BigramLanguageModel ----------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        # when generating we don't provide the target and so no loss calculation
        if targets is None:
            loss = None
            
        # when target is provided calculate the loss
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx)
            # focus only on the latest time step embedding
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ----- train the model ------
for iter in range(max_iters):
    # sample a batch of data
    xb, yb = get_batch("train")
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
     # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
