import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)


class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        
        self.gn1 = nn.GroupNorm(num_groups, n_embd)
        self.gn2 = nn.GroupNorm(num_groups, n_embd)

    def forward(self, x):
        x = x + self.sa(
            self.gn1(
                x.permute(0, 2, 1)    # (B,T,C) -> (B,C,T)
            ).permute(0, 2, 1))       # (B,C,T) -> (B,T,C)
        x = x + self.ffwd(
            self.gn2(
                x.permute(0, 2, 1)    # (B,T,C) -> (B,C,T)
            ).permute(0, 2, 1))       # (B,C,T) -> (B,T,C)
        return x


class BigramLanguageModel(nn.Module):
    """ Super simple bigram model """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.gn_f = nn.GroupNorm(num_groups, n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.gn_f(
            x.permute(0, 2, 1)   # (B,T,C) -> (B,C,T)
        ).permute(0, 2, 1)       # (B,C,T) -> (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def plot_losses(train_losses, val_losses, eval_interval=100, model_name="model"):
    plt.figure(figsize=(15, 5))

    plt.grid(True)
    plt.plot(train_losses, linewidth=0.5, color="orange", label="Training loss", alpha=0.8)
    val_steps = [i * eval_interval for i in range(len(val_losses))]
    plt.plot(val_steps, val_losses, linewidth=1, color="red")
    plt.scatter(val_steps, val_losses, s=3, color="red", label="Validation loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training and validation losses for {model_name}")
    plt.legend()
    
    plt.savefig(f"{dir_path}/loss_graphics/{model_name.split(' ')[0]}_{timestamp}.png")

def generate_with_context(model, max_new_tokens=2000, context_str=""):
    if len(context_str) % 2:
        context_str = '\t' + context_str
    context = torch.tensor((encode(context_str[ : len(context_str)//2]), 
                            encode(context_str[len(context_str)//2 : ])), 
                           dtype=torch.long, 
                           device=device) \
        if len(context_str) else \
              torch.zeros((1, 1), dtype=torch.long, device=device)
    return ("\n\nContext:\n " + "'" + context_str + "'" + "\n" + 
            "Encoded context:\n" + str(context) + "\n" +
            decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))

# Hyperparameters definition

batch_size = 256 # how many independent sequences will we process in parallel
block_size = 32 # the maximum context length for predictions
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
num_groups = 8
n_head = 4
n_layer = 4
dropout = 0.0

dir_path = os.path.dirname(__file__)
timestamp = datetime.now().strftime("%d-%H-%M-%S")

# Dataset processing

songs_file = dir_path + "/datasets/corpus_split_clean.txt"
with open(songs_file, 'r', encoding='utf-8') as all_texts:
    songs = all_texts.read()
print("Texts read from file", songs_file)

# gather the unique characters occurring in the text
vocab = sorted(list(set(songs)))
vocab_size = len(vocab)
# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(vocab) }
itos = { i:ch for i, ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[int(i)] for i in l]) # decoder: take a list of integers, output a string
print("Texts encoded, mappings created")

# Split dataset on train and val

data = torch.tensor(encode(songs), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Train model

# set model
model = BigramLanguageModel().to(device)
model_name = "BigramLanguageModel with attention"
print("The model includes", sum(param.numel() for param in model.parameters())/1e6, 'M parameters')

# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# set loss and checkpoint logging
loss_dynamics_train, loss_dynamics_val = [], []
best_val_loss = float('inf')
best_model_weights = None
checkpts_path = dir_path + f"/checkpoints/best_bigram_{timestamp}.pth"

print(f"Start model training on {device}...")
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if (not iter % eval_interval) or (iter == max_iters - 1):
        losses = estimate_loss()
        print(f"    Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        loss_dynamics_val.append(losses['val'])

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_weights = model.state_dict().copy()
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": iter,
                "best_val_loss": best_val_loss,
            }, checkpts_path)

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    loss_dynamics_train.append(loss.item())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training complete. Best val loss: {best_val_loss:.4f}")

# Show train results: vizualization + generation

# plot model's loss
if best_weights is not None:
    model.load_state_dict(best_weights)
plot_losses(loss_dynamics_train, loss_dynamics_val, model_name=model_name)

print("\nGeneration examples:")
generation_file = f"{dir_path}/generation_examples/{model_name.split(' ')[0]}_{timestamp}.txt"
context_strs = ["",
                "Не в чистом поле, не в пустой степи",
                "Выйду ночью в поле с конем",
                "Аааааа"]
with open(generation_file, 'w', encoding='utf-8') as generation:
    for context_str in context_strs:
        generation.write(generate_with_context(model, 2000, context_str))
        print("Generation for context '" + context_str + "' done")