import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

class MeanPooler(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # [B,T,1]
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom


class ReviewMTL(nn.Module):
    def __init__(self, encoder_name: str, n_quality: int = 3, n_relevance: int = 2, proj_dim: int = 256, dropout: float = 0.1, meta_dim: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pool = MeanPooler()
        H = self.encoder.config.hidden_size

        # Projection head for contrastive space
        self.proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, proj_dim)
        )

        # Optional meta pathway (concat features after pooling)
        self.meta_bn = nn.BatchNorm1d(meta_dim) if meta_dim > 0 else None
        head_in = H + (meta_dim if meta_dim > 0 else 0)

        # Two classifier heads
        self.head_quality = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_quality))
        self.head_relev  = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_relevance))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, meta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = self.pool(out.last_hidden_state, attention_mask)  # [B,H]
        z = nn.functional.normalize(self.proj(emb), p=2, dim=-1)  # [B,D]
        if meta is not None and meta.numel() > 0:
            if self.meta_bn is not None:
                meta = self.meta_bn(meta)
            emb = torch.cat([emb, meta], dim=1)
        q_logits = self.head_quality(emb)
        r_logits = self.head_relev(emb)
        return z, q_logits, r_logits
      
@dataclass
class TrainConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_len: int = 128
    classes_per_batch: int = 6
    samples_per_class: int = 4  # batch = C × S
    epochs_contrastive: int = 2
    epochs_joint: int = 4
    lr_encoder: float = 2e-5
    lr_heads: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.05
    lambda_contrastive: float = 0.1
    temperature: float = 0.07
    use_class_weights: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ReviewMTLPredictor:
    def __init__(self, model, tokenizer, cfg, labels, thresholds, device):
        self.model = model.eval()
        self.tok = tokenizer
        self.cfg = cfg
        self.labels = labels
        self.thresh = thresholds
        self.device = device

    @classmethod
    def from_dir(cls, model_dir: str, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load cfg + tokenizer
        with open(os.path.join(model_dir, "cfg.json")) as f:
            cfg = TrainConfig(**json.load(f))
        tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), use_fast=True)

        # rebuild model skeleton (no meta features in your setup → meta_dim=0)
        m = ReviewMTL(cfg.encoder_name, n_quality=3, n_relevance=2, meta_dim=0).to(device)
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
        m.load_state_dict(state)

        # labels & thresholds
        with open(os.path.join(model_dir, "labels.json")) as f:
            labels = json.load(f)
        with open(os.path.join(model_dir, "thresholds.json")) as f:
            thresholds = json.load(f)

        return cls(m, tok, cfg, labels, thresholds, device)

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 64) -> pd.DataFrame:
        q_probs_all, r_probs_all = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(
                batch, truncation=True, padding=True,
                max_length=self.cfg.max_len, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            _, q, r = self.model(enc["input_ids"], enc["attention_mask"])
            q_probs_all.append(q.softmax(-1).cpu().numpy())
            r_probs_all.append(r.softmax(-1).cpu().numpy())

        q_probs = np.vstack(q_probs_all)      # [N,3]
        r_probs = np.vstack(r_probs_all)      # [N,2]

        q_ids = q_probs.argmax(1)
        r_ids = (r_probs[:,1] >= float(self.thresh.get("relevance_threshold", 0.5))).astype(int)

        # map to strings
        q_map = self.labels["quality"]       # {"0":"Low",...} or {0:"Low",...}
        r_map = self.labels["relevance"]
        # ensure int keys
        q_str = np.array([q_map[str(i)] if str(i) in q_map else q_map[i] for i in q_ids])
        r_str = np.array([r_map[str(i)] if str(i) in r_map else r_map[i] for i in r_ids])

        # simple usefulness score for ranking
        uvs = r_probs[:,1] * (0.7*q_probs[:,2] + 0.3*q_probs[:,1])

        return pd.DataFrame({
            "pred_quality": q_str,
            "pred_quality_p": q_probs.max(1).round(3),
            "pred_relevance": r_str,
            "pred_relevance_p": r_probs[:,1].round(3),
            "uvs": uvs.round(3),
            "p_quality_low":  q_probs[:,0],
            "p_quality_medium": q_probs[:,1],
            "p_quality_good": q_probs[:,2],
            "p_relevance_irrel": r_probs[:,0],
            "p_relevance_rel":   r_probs[:,1],
        })