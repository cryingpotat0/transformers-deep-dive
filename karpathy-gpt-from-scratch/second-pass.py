import torch as t
import numpy as np
import torch.nn as nn
import torch.optim as optim
import requests as r

# Hyper params
train_test_split = 0.8
batch_size = 32
context_size = 8
embedding_size = 32
head_size = 16
iters = 50000
reporting_iters = 5000

# Data loading and encoding/ decoding
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data = r.get(dataset_url)
data = data.text

unique_chars = sorted(list(set(data)))
i_to_c = {i: c for i, c in enumerate(unique_chars)}
c_to_i = {c: i for i, c in enumerate(unique_chars)}

def encode_string(s):
    return [c_to_i[c] for c in s]

def decode(t):
    return "".join([i_to_c[i] for i in t])

# More data loading
data = t.LongTensor(encode_string(data))
num_train = int(len(data) * train_test_split)
train, test = data[:num_train], data[num_train:]

def get_batch(split):
    to_batch = train if split == "train" else test
    idxs =  t.randint(len(to_batch) - context_size, (batch_size, ))
    x = t.stack([to_batch[i:i+context_size] for i in idxs])
    y = t.stack([to_batch[i+1:i+context_size+1] for i in idxs])
    return x, y
    

# Model definition
language_size = len(unique_chars)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(language_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)
        self.lm_head =  nn.Linear(head_size, language_size)
        self.keys = nn.Linear(embedding_size, head_size)
        self.queries = nn.Linear(embedding_size, head_size)
        self.values =  nn.Linear(embedding_size, head_size)


    def forward(self, x, target=None):
        # x (B, T), target (B, T)
        B, T = x.shape
        token_embeddings = self.token_embedding(x) # (B, T, C)
        position_embeddings = self.position_embedding(t.arange(T)) # T, C
        embeddings = token_embeddings + position_embeddings # (B, T, C)
        keys = self.keys(embeddings) # (B, T, H)
        queries = self.queries(embeddings) # (B, T, H)
        # print(keys.size(), queries.size())
        wei = queries @ keys.transpose(-2, -1) / np.sqrt(head_size) # (T, T)
        trill = t.tril(t.ones(T, T))
        wei = wei.masked_fill(trill == 0, -t.inf)
        wei = nn.functional.softmax(wei, dim=1) # (B, T, T)
        pre_logits = wei @ self.values(embeddings) # (B, T, T)
        # print(pre_logits.shape)
        logits = self.lm_head(pre_logits)
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = nn.functional.cross_entropy(logits, target)
        else:
            loss = None
        return logits, loss

    def generate(self, idxs, max_num_tokens):
        for _ in range(max_num_tokens):
            logits, _ = self(idxs[:, -context_size:])
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
            idx_next = t.multinomial(probs, num_samples=1)
            idxs = t.cat((idxs, idx_next), dim=-1)
        return idxs

def compute_loss(model, loss):
    model.eval()
    x, y = get_batch('train')
    _, loss = model.forward(x, y)
    print(f"Loss is {loss}")
    model.train()

# Train
model = BigramLanguageModel()
optim = optim.Adam(model.parameters())

for i in range(iters):
    x, y = get_batch('train')
    _, loss = model.forward(x, y)
    if loss is None:
        raise ValueError("loss is none")
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if i % reporting_iters == 0:
        compute_loss(model, loss)

# Try on a random output
generate_tokens = 1000
print(decode(model.generate(t.zeros((1, context_size - 1), dtype=t.long), generate_tokens)[0].tolist()))


