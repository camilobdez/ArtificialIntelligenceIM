"""
run_all.py — Corre las verificaciones de los 4 ejercicios del Workshop 3.

Uso:
    cd Workshops/Workshop3/2026-1
    python run_all.py

Requiere: pip install torch matplotlib tokenizers
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import (
    ModelConfig, RMSNorm, SwiGLUFFN, MiniLLaMA,
    precompute_rope_freqs, apply_rope, GroupedQueryAttention,
)
from src.tokenizer import BPETokenizer
from src.data import get_corpus

PASS = "✓"
FAIL = "✗"

# ===========================================================================
# SECCIÓN 1 — RMSNorm y SwiGLU
# ===========================================================================

class MyRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


class MySwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        D, Dff = config.d_model, config.d_ff
        self.gate_proj = nn.Linear(D, Dff, bias=False)
        self.up_proj   = nn.Linear(D, Dff, bias=False)
        self.down_proj = nn.Linear(Dff, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)


def verify_section1():
    print("=" * 55)
    print("VERIFICACIÓN 1 — RMSNorm y SwiGLU")
    print("=" * 55)
    cfg = ModelConfig(vocab_size=256, d_model=32, n_heads=4,
                      n_kv_heads=2, d_ff=64, max_seq_len=16)
    x = torch.randn(2, 8, 32)

    ref_norm = RMSNorm(32)
    my_norm  = MyRMSNorm(32)
    my_norm.gamma = ref_norm.gamma
    norm_ok = torch.allclose(ref_norm(x), my_norm(x), atol=1e-5)
    print(f"  RMSNorm output match : {norm_ok}")

    ref_ffn = SwiGLUFFN(cfg)
    my_ffn  = MySwiGLUFFN(cfg)
    my_ffn.gate_proj.weight = ref_ffn.gate_proj.weight
    my_ffn.up_proj.weight   = ref_ffn.up_proj.weight
    my_ffn.down_proj.weight = ref_ffn.down_proj.weight
    ffn_ok = torch.allclose(ref_ffn(x), my_ffn(x), atol=1e-5)
    print(f"  SwiGLU output match  : {ffn_ok}")
    print(f"  SwiGLU output shape  : {list(my_ffn(x).shape)}  (esperado [2, 8, 32])")

    ok = norm_ok and ffn_ok
    print(f"\n  {PASS if ok else FAIL} Sección 1 {'correcta' if ok else 'FALLA'}\n")
    return ok


# ===========================================================================
# SECCIÓN 2 — Weight tying
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def build_model_no_tying(config: ModelConfig) -> MiniLLaMA:
    model = MiniLLaMA(config)
    model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
    return model


def verify_section2():
    print("=" * 55)
    print("VERIFICACIÓN 2 — Weight tying")
    print("=" * 55)
    cfg = ModelConfig(vocab_size=256, d_model=32, n_heads=4,
                      n_kv_heads=2, d_ff=64, max_seq_len=16)

    model_tied   = MiniLLaMA(cfg)
    model_no_tie = build_model_no_tying(cfg)

    params_tied   = count_parameters(model_tied)
    params_no_tie = count_parameters(model_no_tie)
    saved         = params_no_tie - params_tied
    expected_save = cfg.vocab_size * cfg.d_model

    print(f"  Params con tying    : {params_tied:,}")
    print(f"  Params sin tying    : {params_no_tie:,}")
    print(f"  Diferencia          : {saved:,}  (esperado {expected_save:,})")
    tying_ok  = model_tied.lm_head.weight.data_ptr() == model_tied.embed.weight.data_ptr()
    no_tie_ok = model_no_tie.lm_head.weight.data_ptr() != model_no_tie.embed.weight.data_ptr()
    print(f"  Mismo tensor (tied)      : {tying_ok}")
    print(f"  Tensor distinto (no tied): {no_tie_ok}")

    ok = (saved == expected_save and tying_ok and no_tie_ok)
    print(f"\n  {PASS if ok else FAIL} Sección 2 {'correcta' if ok else 'FALLA'}\n")
    return ok


# ===========================================================================
# SECCIÓN 3 — GQA vs MHA
# ===========================================================================

def build_attention(n_heads: int, n_kv_heads: int, d_model: int = 32) -> GroupedQueryAttention:
    cfg = ModelConfig(vocab_size=256, d_model=d_model, n_heads=n_heads,
                      n_kv_heads=n_kv_heads, d_ff=d_model * 2, max_seq_len=128)
    return GroupedQueryAttention(cfg)


def compare_attention_maps(attn_mha, attn_gqa, x, rope_freqs, mask):
    def _get_attn_weights(mod, x, rope_freqs, mask):
        B, T, D = x.shape
        Dh = mod.head_dim
        q = mod.Wq(x).reshape(B, T, mod.n_heads,    Dh)
        k = mod.Wk(x).reshape(B, T, mod.n_kv_heads, Dh)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)
        k = k.unsqueeze(3).expand(B, T, mod.n_kv_heads, mod.n_rep, Dh)\
             .reshape(B, T, mod.n_heads, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    return _get_attn_weights(attn_mha, x, rope_freqs, mask), \
           _get_attn_weights(attn_gqa, x, rope_freqs, mask)


def verify_section3():
    print("=" * 55)
    print("VERIFICACIÓN 3 — GQA vs MHA")
    print("=" * 55)
    D, Hq, Hkv, T = 32, 4, 2, 10
    mha = build_attention(n_heads=Hq, n_kv_heads=Hq,  d_model=D)
    gqa = build_attention(n_heads=Hq, n_kv_heads=Hkv, d_model=D)

    mha_kv = mha.Wk.weight.numel() + mha.Wv.weight.numel()
    gqa_kv = gqa.Wk.weight.numel() + gqa.Wv.weight.numel()
    print(f"  MHA  K+V params : {mha_kv:,}")
    print(f"  GQA  K+V params : {gqa_kv:,}")
    print(f"  Reducción       : {(1 - gqa_kv/mha_kv)*100:.0f}%  (esperado {(1-Hkv/Hq)*100:.0f}%)")

    x          = torch.randn(1, T, D)
    rope_freqs = precompute_rope_freqs(D // Hq, T)
    mask       = torch.tril(torch.ones(T, T))

    out_mha = build_attention(Hq, Hq,  D)(x, rope_freqs, mask)
    out_gqa = build_attention(Hq, Hkv, D)(x, rope_freqs, mask)
    print(f"  MHA output shape: {list(out_mha.shape)}")
    print(f"  GQA output shape: {list(out_gqa.shape)}")

    w_mha, w_gqa = compare_attention_maps(mha, gqa, x, rope_freqs, mask)
    print(f"  MHA attn shape  : {list(w_mha.shape)}")
    print(f"  GQA attn shape  : {list(w_gqa.shape)}")

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, w, title in zip(axes, [w_mha, w_gqa], ["MHA (head 0)", "GQA (head 0)"]):
        im = ax.imshow(w[0, 0].detach().numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Attention maps — MHA vs GQA")
    plt.tight_layout()
    plt.savefig("attention_maps.png", dpi=130)
    print("  Plot guardado en attention_maps.png")

    shapes_ok = out_mha.shape == out_gqa.shape == torch.Size([1, T, D])
    kv_ok     = gqa_kv == mha_kv * Hkv // Hq
    ok = shapes_ok and kv_ok
    print(f"\n  {PASS if ok else FAIL} Sección 3 {'correcta' if ok else 'FALLA'}\n")
    return ok


# ===========================================================================
# SECCIÓN 4 — RoPE ablation
# ===========================================================================

class AttentionNoRoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads
        self.head_dim   = config.d_model // config.n_heads
        D, Dh = config.d_model, self.head_dim
        self.Wq = nn.Linear(D, config.n_heads    * Dh, bias=False)
        self.Wk = nn.Linear(D, config.n_kv_heads * Dh, bias=False)
        self.Wv = nn.Linear(D, config.n_kv_heads * Dh, bias=False)
        self.Wo = nn.Linear(config.n_heads * Dh, D,    bias=False)

    def forward(self, x, rope_freqs, mask=None):
        B, T, D = x.shape
        Dh = self.head_dim
        q = self.Wq(x).reshape(B, T, self.n_heads,    Dh)
        k = self.Wk(x).reshape(B, T, self.n_kv_heads, Dh)
        v = self.Wv(x).reshape(B, T, self.n_kv_heads, Dh)
        # Sin RoPE — Q y K no son rotadas
        k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, Dh)\
             .reshape(B, T, self.n_heads, Dh)
        v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, Dh)\
             .reshape(B, T, self.n_heads, Dh)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)
        return self.Wo(out.transpose(1, 2).reshape(B, T, self.n_heads * Dh))


class TextDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int):
        self.seq_len = seq_len
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.n       = max(0, len(self.data) - seq_len - 1)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        return (self.data[idx : idx + self.seq_len],
                self.data[idx + 1 : idx + self.seq_len + 1])


def train_quick(use_rope: bool, epochs: int = 100) -> list:
    SEED, SEQ_LEN, BATCH_SIZE, LR = 42, 64, 8, 3e-3
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(__file__)
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(base, "checkpoints", "tokenizer.json"))
    corpus    = get_corpus(os.path.join(base, "corpus.txt"))
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)

    dataset    = TextDataset(token_ids, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=4,
                      n_kv_heads=2, d_ff=256, max_seq_len=SEQ_LEN, dropout=0.0)
    model = MiniLLaMA(cfg).to(device)

    if not use_rope:
        model.layer.attn = AttentionNoRoPE(cfg).to(device)
        torch.manual_seed(SEED)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = epochs * max(len(dataloader), 1)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    model.train()
    losses = []
    for _ in range(epochs):
        epoch_loss, n = 0.0, 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, loss = model(xb, targets=yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1
        losses.append(epoch_loss / max(n, 1))
    return losses


def verify_section4():
    print("=" * 55)
    print("VERIFICACIÓN 4 — RoPE ablation  (entrena 100 epochs)")
    print("=" * 55)

    losses_rope    = train_quick(use_rope=True,  epochs=100)
    losses_no_rope = train_quick(use_rope=False, epochs=100)

    final_rope    = losses_rope[-1]
    final_no_rope = losses_no_rope[-1]
    print(f"  Loss final con RoPE    : {final_rope:.4f}")
    print(f"  Loss final sin RoPE    : {final_no_rope:.4f}")
    print(f"  RoPE mejora el loss    : {final_rope < final_no_rope}")

    plt.figure(figsize=(7, 4))
    plt.plot(losses_rope,    label="con RoPE",  linewidth=1.5)
    plt.plot(losses_no_rope, label="sin RoPE",  linewidth=1.5, linestyle="--")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("RoPE ablation — curva de entrenamiento")
    plt.legend(); plt.tight_layout()
    plt.savefig("rope_ablation.png", dpi=130)
    print("  Plot guardado en rope_ablation.png")

    ok = final_rope < final_no_rope
    print(f"\n  {PASS if ok else '~'} Sección 4 {'correcta' if ok else 'resultado inesperado'}\n")
    return ok


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    results = {
        "Sección 1 — RMSNorm y SwiGLU" : verify_section1(),
        "Sección 2 — Weight tying"      : verify_section2(),
        "Sección 3 — GQA vs MHA"        : verify_section3(),
        "Sección 4 — RoPE ablation"     : verify_section4(),
    }

    print("=" * 55)
    print("RESUMEN")
    print("=" * 55)
    all_ok = True
    for name, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        all_ok = all_ok and ok
    print()
    if all_ok:
        print("  Todas las secciones correctas.")
    else:
        print("  Hay secciones que necesitan revisión.")
