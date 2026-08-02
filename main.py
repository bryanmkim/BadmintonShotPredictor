import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from data_cleaning.data import RallyDataset

rally_size = 70
device = 'cuda' if torch.cuda.is_available() else 'mps'

# ***** PARAMETERS ******
# how many rallies to compute in parallel.
batch_size = 32
n_embd = 64
n_head = 4
n_layer = 1
dropout = 0.4
weight_decay = 0.1
learning_rate = 3e-4
max_iters = 5000

eval_interval = 500
eval_iters = 200

torch.manual_seed(1330)


# Load the CSV file
train_df = pd.read_csv('train.csv')
codes, uniques = pd.factorize(train_df['type'])
train_df['type'] = codes + 1          # 0 reserved for padding
train_df['player'] = train_df['player'] + 1 # 0 reserved for padding

val_given = pd.read_csv('val_given.csv')
val_gt = pd.read_csv('val_gt.csv')
val_df = pd.concat([val_given, val_gt]).sort_values(['rally_id', 'ball_round'])
val_df['type'] = val_df['type'].apply(lambda x: list(uniques).index(x) + 1)
val_df['player'] = val_df['player'] + 1

train_data = RallyDataset(train_df)
val_data = RallyDataset(val_df)

def get_batch(split):
    # generate a small abtch of data of inputs x and targets y

    # indexing from data returns data for one rally with 8 arrays of length 70(but anything past n shots is padding of 0).
    # each array represents one property. Indexing from the array returns the value of that property at {index}th shot.
    # random rally indexes.

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))

    rallies = [data[i] for i in ix]
    
    inp_shot   = torch.stack([torch.tensor(r[0]) for r in rallies])  # (32, 70)
    inp_x      = torch.stack([torch.tensor(r[1], dtype=torch.float32) for r in rallies])  # (32, 70)
    inp_y      = torch.stack([torch.tensor(r[2], dtype=torch.float32) for r in rallies])  # (32, 70)
    inp_player = torch.stack([torch.tensor(r[3]) for r in rallies])  # (32, 70)
    tgt_shot   = torch.stack([torch.tensor(r[4]) for r in rallies])  # (32, 70)
    tgt_x      = torch.stack([torch.tensor(r[5], dtype=torch.float32) for r in rallies])  # (32, 70)
    tgt_y      = torch.stack([torch.tensor(r[6], dtype=torch.float32) for r in rallies])  # (32, 70)
    tgt_player = torch.stack([torch.tensor(r[7]) for r in rallies])  # (32, 70)
    n          = torch.tensor([r[8] for r in rallies])               # (32,)
    
    return (inp_shot.to(device), inp_x.to(device), inp_y.to(device), inp_player.to(device),
            tgt_shot.to(device), tgt_x.to(device), tgt_y.to(device), tgt_player.to(device), n.to(device))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        shot_losses = torch.zeros(eval_iters)
        area_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player, n = get_batch(split)
            shot_logits, area_out, loss = model(inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player)
            
            pad_mask = (inp_shot != 0)
            pad_mask[:, :3] = False
            shot_losses[k] = F.cross_entropy(shot_logits[pad_mask], tgt_shot[pad_mask]).item()
            gold_xy = torch.stack([tgt_x[pad_mask], tgt_y[pad_mask]], dim=-1)
            area_losses[k] = Gaussian2D_loss(area_out[pad_mask], gold_xy).item()
        out[split] = {'shot': shot_losses.mean(), 'area': area_losses.mean(), 'total': (shot_losses + area_losses).mean()}
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(rally_size, rally_size, dtype=torch.bool)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inp_shot):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        pad = (inp_shot != 0).unsqueeze(1)                    # (B, 1, T)
        mask = self.tril[:T, :T] & pad                        # (B, T, T)
        wei = wei.masked_fill(mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inp_shot):
        out = torch.cat([h(x, inp_shot) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, inp_shot):
        x = x + self.sa(self.ln1(x), inp_shot)
        x = x + self.ffwd(self.ln2(x))
        return x

class ShotPredictionModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.shot_embedding  = nn.Embedding(11, n_embd)     # 10 types + pad
        # self.area_linear     = nn.Linear(2, n_embd)          # continuous (x,y) — not an embedding
        self.position_embedding_table = nn.Embedding(rally_size, n_embd)
        self.player_embedding = nn.Embedding(36, n_embd)     # 35 players + pad
        self.area_linear = nn.Linear(2, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, 11)
        self.area_head = nn.Linear(n_embd, 5)


    def forward(self, inp_shot, inp_x, inp_y, inp_player, tgt_shot=None, tgt_x=None, tgt_y=None, tgt_player=None):
        B, T = inp_shot.shape
        shot_emb = self.shot_embedding(inp_shot)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        area_emb = F.relu(self.area_linear(torch.stack([inp_x, inp_y], dim=-1)))  # (B,T,2) → (B,T,C)
        player_emb = self.player_embedding(inp_player)

        x = shot_emb + area_emb + pos_emb + player_emb 
        for block in self.blocks:
            x = block(x, inp_shot) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        shot_logits = self.lm_head(x) # (B,T,vocab_size)
        area_out = self.area_head(x)

        if tgt_shot is None:
            loss = None
        else:
            pad_mask = (inp_shot != 0)
            pad_mask[:, :3] = False
            loss_shot = F.cross_entropy(shot_logits[pad_mask], tgt_shot[pad_mask])
            gold_xy = torch.stack([tgt_x[pad_mask], tgt_y[pad_mask]], dim=-1)
            loss_area = Gaussian2D_loss(area_out[pad_mask], gold_xy)
            loss = loss_shot + loss_area

        return shot_logits, area_out, loss
    
def Gaussian2D_loss(V_pred, V_trgt):
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]
    sx = torch.exp(V_pred[:, 2])
    sy = torch.exp(V_pred[:, 3])
    corr = torch.tanh(V_pred[:, 4])

    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * corr * normx * normy / sxsy
    neg_rho = 1 - corr ** 2

    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * 3.14159265 * sxsy * torch.sqrt(neg_rho)
    result = -torch.log(torch.clamp(result / denom, min=1e-20))
    return result.mean()
    
model = ShotPredictionModel()
model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        t, v = losses['train'], losses['val']
        print(f"step {iter}: train {t['shot']:.4f}(shot type)+{t['area']:.4f}(landing area)={t['total']:.4f} | val {v['shot']:.4f}+{v['area']:.4f}={v['total']:.4f}")
    # sample a batch of data
    inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player, n = get_batch('train')
    # evaluate the loss
    shot_logits, area_out, loss = model(inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# grab one batch from val
inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player, n = get_batch('val')
shot_logits, area_out, loss = model(inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player)

# for the first rally in the batch
rally_len = n[0].item()
preds = shot_logits[0].argmax(dim=-1)
area = area_out[0]  # (T, 5)

type_names = ['PAD'] + list(uniques)

print("pos | predicted        | actual           | ✓? | pred (x, y)     | actual (x, y)")
print("----|------------------|------------------|----|-----------------|----------------")
for t in range(rally_len):
    pred_name = type_names[preds[t].item()]
    true_name = type_names[tgt_shot[0][t].item()]
    match = "✓" if preds[t].item() == tgt_shot[0][t].item() else "✗"
    px, py = area[t][0].item(), area[t][1].item()  # mux, muy from Gaussian
    tx, ty = tgt_x[0][t].item(), tgt_y[0][t].item()
    print(f"  {t} | {pred_name:16s} | {true_name:16s} | {match}  | ({px:+.2f}, {py:+.2f})  | ({tx:+.2f}, {ty:+.2f})")

    correct, total = 0, 0
    
for _ in range(100):
    inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player, n = get_batch('val')
    shot_logits, area_out, loss = model(inp_shot, inp_x, inp_y, inp_player, tgt_shot, tgt_x, tgt_y, tgt_player)
    pad_mask = (inp_shot != 0)
    pad_mask[:, :3] = False
    preds = shot_logits[pad_mask].argmax(dim=-1)
    correct += (preds == tgt_shot[pad_mask]).sum().item()
    total += pad_mask.sum().item()
print(f"accuracy: {correct/total:.4f}")