#!/usr/bin/env python3
import os, math, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from momentfm import MOMENTPipeline

# =========================
# CONFIG
# =========================
MODEL_ID = "AutonLab/MOMENT-1-small"
HORIZON = 32
N_CHANNELS = 1
N_TOTAL_STEPS = 6000
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-4

# LoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EXCLUDE_KEYWORDS = ("head", "forecast", "classifier")  # generic: avoid touching heads

OUT_DIR = "moment_lora_adapter"
RESULTS_JSON = "results.json"

# =========================
# data
# =========================
def set_seed(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_series(T, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, T)
    ys = []
    for c in range(C):
        trend = (c+1)*0.8*t + 0.2*(t**2)
        season = 0.6*torch.sin(2*math.pi*(3+c)*t) + 0.2*torch.sin(2*math.pi*(13+2*c)*t)
        noise = 0.15*torch.randn(T, generator=g)
        y = trend + season + noise
        ys.append(y)
    y = torch.stack(ys, 0)  # [C,T]
    y = (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-6)
    return y

class WinDS(Dataset):
    def __init__(self, series_ct, start, end, L, H):
        self.s = series_ct
        self.L, self.H = L, H
        self.idxs = list(range(start, end - (L+H)))
        self.mask = torch.ones(L, dtype=torch.long)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        k = self.idxs[i]
        x = self.s[:, k:k+self.L]                 # [C,L]
        y = self.s[:, k+self.L:k+self.L+self.H]   # [C,H]
        return x, y, self.mask

def collate(b):
    x,y,m = zip(*b)
    return torch.stack(x), torch.stack(y), torch.stack(m)  # [B,C,L],[B,C,H],[B,L]

# =========================
# LoRA monkey-patch for nn.Linear
# =========================
def add_lora_to_linear(linear: torch.nn.Linear, r, alpha, dropout):
    # attach LoRA params on the linear module (no type replacement)
    linear.lora_r = r
    linear.lora_alpha = alpha
    linear.lora_scaling = alpha / float(r)
    linear.lora_dropout = torch.nn.Dropout(dropout)
    linear.lora_A = torch.nn.Parameter(torch.zeros(r, linear.in_features))
    linear.lora_B = torch.nn.Parameter(torch.zeros(linear.out_features, r))
    torch.nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
    torch.nn.init.zeros_(linear.lora_B)

    # freeze base weights
    linear.weight.requires_grad = False
    if linear.bias is not None:
        linear.bias.requires_grad = False

    # patch forward
    old_forward = linear.forward
    def lora_forward(x):
        out = old_forward(x)
        lora = (linear.lora_dropout(x) @ linear.lora_A.t()) @ linear.lora_B.t()
        return out + linear.lora_scaling * lora
    linear.forward = lora_forward

def inject_lora(model, r, alpha, dropout, exclude_keywords=()):
    replaced = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(k in name.lower() for k in exclude_keywords):
            add_lora_to_linear(mod, r=r, alpha=alpha, dropout=dropout)
            replaced += 1
    return replaced

def mark_only_lora_trainable(model):
    for n,p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)

def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn

def lora_state_dict(model):
    return {k: v.detach().cpu() for k,v in model.state_dict().items() if ("lora_A" in k) or ("lora_B" in k)}

# =========================
# eval/train
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum = mae_sum = 0.0
    n = 0
    for x,y,m in loader:
        x,y,m = x.to(device), y.to(device), m.to(device)
        B,C,L = x.shape
        x_in = x.reshape(B*C, 1, L)
        m_in = m.repeat_interleave(C, 0)
        out = model(x_enc=x_in, input_mask=m_in)        # MOMENT keyword-only
        yhat = out.forecast                              # [B*C,1,H] or [B*C, H]
        if yhat.ndim == 3: yhat = yhat[:,0,:]
        y_true = y.reshape(B*C, -1)
        mse_sum += F.mse_loss(yhat, y_true, reduction="sum").item()
        mae_sum += F.l1_loss(yhat, y_true, reduction="sum").item()
        n += y_true.numel()
    return {"mse": mse_sum/n, "mae": mae_sum/n}

def train_lora(model, train_loader, val_loader, device, epochs, lr):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best = float("inf"); best_sd = None
    for ep in range(1, epochs+1):
        model.train()
        for x,y,m in train_loader:
            x,y,m = x.to(device), y.to(device), m.to(device)
            B,C,L = x.shape
            x_in = x.reshape(B*C, 1, L)
            m_in = m.repeat_interleave(C, 0)
            out = model(x_enc=x_in, input_mask=m_in)
            yhat = out.forecast
            if yhat.ndim == 3: yhat = yhat[:,0,:]
            y_true = y.reshape(B*C, -1)
            loss = F.mse_loss(yhat, y_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate(model, val_loader, device)
        print(f"[epoch {ep:02d}] val_mse={val['mse']:.6f} val_mae={val['mae']:.6f}")
        if val["mse"] < best:
            best = val["mse"]
            best_sd = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)

# =========================
# main
# =========================
def main():
    set_seed(0)
    device = dev()
    print("device =", device)

    # load MOMENT forecasting head
    pipe = MOMENTPipeline.from_pretrained(
        MODEL_ID,
        model_kwargs={"task_name": "forecasting", "forecast_horizon": HORIZON, "n_channels": 1},
    )
    pipe.init()
    model = pipe.to(device)

    SEQ_LEN = pipe.config.seq_len

    # data
    series = make_series(N_TOTAL_STEPS, N_CHANNELS, seed=0)
    T = series.size(1)
    tr_end = int(TRAIN_FRAC*T)
    va_end = int((TRAIN_FRAC+VAL_FRAC)*T)

    train_ds = WinDS(series, 0, tr_end, SEQ_LEN, HORIZON)
    val_ds   = WinDS(series, tr_end, va_end, SEQ_LEN, HORIZON)
    test_ds  = WinDS(series, va_end, T, SEQ_LEN, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # baseline eval (freeze everything)
    for p in model.parameters(): p.requires_grad = False
    tot, trn = count_params(model)
    print(f"[baseline] total_params={tot:,} trainable_params={trn:,}")
    base = evaluate(model, test_loader, device)
    print(f"[baseline] test_mse={base['mse']:.6f} test_mae={base['mae']:.6f}")

    # inject LoRA (no deep MOMENT knowledge)
    nrep = inject_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, exclude_keywords=EXCLUDE_KEYWORDS)
    model = model.to(device)              # move newly-created LoRA params
    mark_only_lora_trainable(model)
    tot2, trn2 = count_params(model)
    print(f"[lora] replaced_linears={nrep}")
    print(f"[lora] total_params={tot2:,} trainable_params={trn2:,}")

    # train LoRA
    train_lora(model, train_loader, val_loader, device, EPOCHS, LR)

    # eval after LoRA
    after = evaluate(model, test_loader, device)
    print(f"[lora] test_mse={after['mse']:.6f} test_mae={after['mae']:.6f}")

    # save only adapters
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(lora_state_dict(model), os.path.join(OUT_DIR, "lora_adapters.pt"))
    with open(RESULTS_JSON, "w") as f:
        json.dump(
            {"baseline": base, "lora": after,
             "params": {"baseline_total": tot, "baseline_trainable": trn,
                        "lora_total": tot2, "lora_trainable": trn2},
             "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
             "seq_len": SEQ_LEN, "horizon": HORIZON},
            f, indent=2
        )
    print(f"saved adapters to {OUT_DIR}/lora_adapters.pt and results to {RESULTS_JSON}")

if __name__ == "__main__":
    main()

