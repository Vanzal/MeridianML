"""Ein GPT-Sprachmodell (Decoder-only Transformer), von Grund auf in PyTorch gebaut.

Aufbau (von unten nach oben):
    Token-Embedding + Positions-Embedding
    -> N x [LayerNorm -> Causal Self-Attention -> + Residual -> LayerNorm -> MLP -> + Residual]
    -> LayerNorm -> Linear (Vokabular-Grösse)   (Gewichte mit dem Token-Embedding geteilt)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 128      # Kontextlänge: so viele Tokens sieht das Modell auf einmal
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class KVCache:
    """Speicher für Keys und Values aller Schichten, damit beim Erzeugen jedes Token nur einmal gerechnet wird.

    Ohne Cache rechnet das Modell für jedes neue Token den ganzen bisherigen Text noch einmal durch (Aufwand wächst
    quadratisch). Mit Cache wird nur das neue Token gerechnet; Keys/Values der alten Tokens werden wiederverwendet.
    Das ändert am Ergebnis nichts (verlustfrei), es spart nur Rechenzeit.
    """

    def __init__(self, cfg: GPTConfig, batch: int, device, dtype=torch.float32):
        hd = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch, cfg.n_head, cfg.block_size, hd)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0                         # so viele Positionen sind schon gefüllt
        self.capacity = cfg.block_size

    def reset(self) -> None:
        self.length = 0

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd muss durch n_head teilbar sein"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False, cache: KVCache | None = None, layer: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:                   # neue Keys/Values ablegen, dann auf alle bisherigen schauen
            start = cache.length
            cache.k[layer, :B, :, start:start + T] = k
            cache.v[layer, :B, :, start:start + T] = v
            if start == 0:                      # Prefill: mit den frischen (zusammenhängenden) Tensoren rechnen –
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # schneller als über den Cache
            elif T == 1:                        # ein Token darf alles Vorherige sehen: keine Maske nötig
                k, v = cache.k[layer, :B, :, :start + 1], cache.v[layer, :B, :, :start + 1]
                y = F.scaled_dot_product_attention(q, k, v)
            else:                               # mehrere neue Tokens hinter schon gecachten
                k, v = cache.k[layer, :B, :, :start + T], cache.v[layer, :B, :, :start + T]
                mask = torch.ones(T, start + T, dtype=torch.bool, device=x.device).tril(diagonal=start)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y), None
        if return_attn:   # langsame Variante, die die Aufmerksamkeits-Matrix verrät (für die Visualisierung)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = self.drop(att) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout_p if self.training else 0.0)
            att = None
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y)), att


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn: bool = False, cache: KVCache | None = None, layer: int = 0):
        a, att = self.attn(self.ln1(x), return_attn, cache, layer)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight Tying: spart Parameter und hilft beim Lernen
        self.apply(self._init_weights)
        for name, p in self.named_parameters():    # GPT-2-Trick: Residual-Projektionen kleiner initialisieren
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False,
                cache: KVCache | None = None):
        """``cache``: optionaler KVCache. Dann sind ``idx`` die *neuen* Tokens ab Position ``cache.length``;
        danach ist ``cache.length`` um ``T`` gewachsen."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        assert start + T <= self.cfg.block_size, f"Sequenz ({start + T}) länger als Kontext ({self.cfg.block_size})"
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        atts = []
        for i, block in enumerate(self.blocks):
            x, att = block(x, return_attn and cache is None, cache, i)
            atts.append(att)
        if cache is not None:
            cache.length += T
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, atts
        return logits, loss

    def new_cache(self, batch: int = 1) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, batch, p.device, p.dtype)

    # ------------------------------------------------------------- Hilfsfunktionen
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Parameter-Anzahl pro Bauteil – für das Diagramm in der App."""
        def n(mod): return sum(p.numel() for p in mod.parameters())
        out = {
            "Token-Embedding (= Ausgabe-Schicht)": n(self.tok_emb),
            "Positions-Embedding": n(self.pos_emb),
            "Attention (alle Schichten)": sum(n(b.attn) for b in self.blocks),
            "MLP (alle Schichten)": sum(n(b.mlp) for b in self.blocks),
            "LayerNorms": sum(n(b.ln1) + n(b.ln2) for b in self.blocks) + n(self.ln_f),
        }
        return out

    @staticmethod
    def estimate_params(cfg: GPTConfig) -> int:
        """Parameter-Zahl berechnen, ohne das Modell zu bauen (für die Vorschau beim Einstellen)."""
        d, L = cfg.n_embd, cfg.n_layer
        attn = d * 3 * d + 3 * d + d * d + d
        mlp = d * 4 * d + 4 * d + 4 * d * d + d
        ln = 2 * 2 * d
        return cfg.vocab_size * d + cfg.block_size * d + L * (attn + mlp + ln) + 2 * d

    def layer_stats(self) -> list[dict]:
        """Gewichts-Statistiken (Mittel, Std, Norm) pro Parameter-Matrix."""
        stats = []
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                d = p.detach()
                stats.append({"name": name, "mean": d.mean().item(), "std": d.std().item(),
                              "norm": d.norm().item(), "n": p.numel()})
        return stats

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        on_cuda = next(self.parameters()).is_cuda     # "fused" = ein einziger GPU-Kernel für den ganzen Update-Schritt
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True if on_cuda else None)
