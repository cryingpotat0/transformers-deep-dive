import requests as r
import torch as t
import torch.optim as op
import time

import torch.nn as nn

# Hyperparams
train_pct = 0.8
batch_size = 32
context_window = 8
embedding_size = 16
dropout = 0.0
num_blocks = 1
num_heads = 1
lr = 1e-2
debug = False
max_iters = 10000
report_iters = 1000

# Data loading
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data = r.get(dataset_url).text
unique_chars = sorted(list(set(data)))
i_to_c = {i: c for i, c in enumerate(unique_chars)}
c_to_i = {c: i for i, c in enumerate(unique_chars)}

def encode(chars):
    return [c_to_i[c] for c in chars]

def decode(ints):
    return "".join([i_to_c[i] for  i in ints])

data_tensor = t.LongTensor(encode(data))
data_size = len(data_tensor)
train_size = int(data_size * train_pct)
vocab_size = len(unique_chars)
train, test = data_tensor[:train_size], data_tensor[train_size:]

if debug:
    print("Train sample", decode(train[:100].tolist()))

def get_batch(split):
    to_batch = train if split == 'train' else test
    batch_starts = t.randint(0, batch_size - context_window, (batch_size, ))
    x = t.stack([to_batch[i:i+context_window] for i in batch_starts])
    y = t.stack([to_batch[i+1:i+1+context_window] for i in batch_starts])
    return x, y

# Model defn

class AttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.keys = nn.Linear(embedding_size, head_size, bias=False)
        self.queries = nn.Linear(embedding_size, head_size, bias=False)
        self.values=  nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', t.tril(t.ones((context_window, context_window))))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        wei = queries @ keys.transpose(-2, -1) / (self.head_size ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -t.inf) # type: ignore
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ values
        return out

class MultiAttentionHead(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn =  t.concat([h(x) for h in self.attention_heads], dim=-1)
        return self.dropout(self.proj(attn))

class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(embedding_size, embedding_size * 4),
                nn.ReLU(),
                nn.Linear(embedding_size * 4, embedding_size),
                nn.Dropout(dropout),
                )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.sa = MultiAttentionHead(head_size=embedding_size // num_heads, num_heads=num_heads)
        self.ff = Feedforward()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ff(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.pos_embeddings = nn.Embedding(context_window, embedding_size)
        self.blocks = nn.Sequential(*[Block(num_heads) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= context_window, f"can only pass in {context_window} tokens"
        x = self.tok_embeddings(x) + self.pos_embeddings(t.arange(T))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            pass
        else:
            loss = None

        return logits, loss

    def generate(self, x, max_num_tokens):
        for _ in range(max_num_tokens):
            logits, _ = self(x[:, -context_window:])
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_toks = t.multinomial(probs, num_samples=1)
            x = t.cat([x, next_toks], dim=-1)
        return x


model = LanguageModel()
optim = op.Adam(model.parameters(), lr=lr)

def compute_loss(model):
    model.eval()
    x, y= get_batch('test')
    _, loss = model.forward(x, y)
    model.train()
    return loss

tokens_to_generate = 1000
print(f"Initial decode: {decode(model.generate(t.zeros((1, context_window - 1), dtype=t.long), tokens_to_generate)[0].tolist())}")

start_time = time.time()
for i in range(max_iters):
    x, y = get_batch('train')
    _, loss = model.forward(x, y)
    if loss is None:
        raise ValueError("loss is none??")
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    if i % report_iters == 0:
        loss = compute_loss(model)
        print(f"[{time.time() - start_time}] {i}: Loss is {loss}")

print(f"Final decode: {decode(model.generate(t.zeros((1, context_window - 1), dtype=t.long), tokens_to_generate)[0].tolist())}")
