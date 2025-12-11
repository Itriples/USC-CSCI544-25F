import math
import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model

from models.GPT2_arch import AccustumGPT2Model
from models.NCL import TimeDataAugment, NCL as NCLHead


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Expects x: (B, S, d_model), adds PE along S.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        S = x.size(1)
        return x + self.pe[:, :S, :]


class Time_Embedding(nn.Module):
    """
    Normalize along time (L), then map L -> hidden_dim and treat M as "sequence length".
    Input:  x: (B, L, M)
    Output: (B, M, hidden_dim), plus means, stdev for de-normalization.
    """

    def __init__(self, input_dim, hidden_dim=768):
        super(Time_Embedding, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, L, M)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = rearrange(x, 'b l m -> b m l')  # (B, M, L)
        x = self.linear(x)                  # (B, M, hidden_dim)
        return x, means, stdev


class EncoderTimeOnly(nn.Module):
    """
    Simple encoder over the "sequence" dimension M (e.g., channels/variables).
    Input:  (B, M, d_model)
    Output: (B, M, d_model)
    """

    def __init__(self, hidden_dim=768, num_heads=12, num_encoder_layers=1, max_len=512):
        super(EncoderTimeOnly, self).__init__()
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        # x: (B, M, d_model)
        x = self.pos_enc(x)                            # (B, M, d_model)
        x = self.transformer_encoder(x.transpose(0, 1))  # (M, B, d_model)
        x = x.transpose(0, 1)                          # (B, M, d_model)
        return x


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )

        # Single GPT-2 temporal branch
        self.gpt2 = AccustumGPT2Model.from_pretrained(
            'gpt2', output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        # Freeze everything except LN, WPE, and LoRA params
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Time embedding (normalize + project)
        self.embd = Time_Embedding(configs.seq_len, hidden_dim=configs.d_model)

        # Time-only encoder (instead of PCA/prototypes)
        self.in_layer = EncoderTimeOnly(
            hidden_dim=configs.d_model,
            num_heads=getattr(configs, "num_heads", 12),
            num_encoder_layers=getattr(configs, "num_encoder_layers", 1),
            max_len=getattr(configs, "enc_in", configs.enc_in),
        )

        # Project intermediate GPT-2 states
        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False)
             for _ in range(configs.gpt_layers + 1)]
        )

        # Output heads
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name in ('imputation', 'anomaly_detection'):
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        # NCL + augmentation
        self.augment = TimeDataAugment(
            jitter_std=getattr(configs, "ncl_jitter_std", 0.01),
            mask_ratio=getattr(configs, "ncl_mask_ratio", 0.0),
        )
        self.ncl_head = NCLHead(
            d_model=configs.d_model,
            proj_dim=getattr(configs, "ncl_proj_dim", 256),
            temperature=getattr(configs, "ncl_temperature", 0.07),
            k=getattr(configs, "ncl_k", 1),
            pool=True,
        )

        for layer in (self.gpt2, self.embd, self.in_layer, self.out_layer, self.time_proj):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    # -----------------------
    # TASK: Forecast
    # -----------------------
    def forecast(self, x, owner_rows=None):
        """
        x: (B, L, M)
        """
        B, L, M = x.shape

        # Normalize + project time dimension
        x_emb, means, stdev = self.embd(x)      # (B, M, d_model)

        # Time-only encoder with positional encoding
        enc = self.in_layer(x_emb)              # (B, M, d_model)

        # GPT-2 over "sequence" M
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=enc)
        # Residual
        outputs_time = outputs_time + enc       # (B, M, d_model)

        # Project intermediate features
        intermidiate_feat_time = tuple(
            self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))
        )

        # NCL: use the last layer hidden states
        z_time = intermidiate_feat_time[-1]     # shape ~ (B, M, d_model)
        v1 = self.augment(z_time)
        v2 = self.augment(z_time)
        ncl_loss = self.ncl_head(v1, v2, owner_rows=owner_rows)

        # Prediction: use all M positions
        outputs_time = self.out_layer(outputs_time)         # (B, M, pred_len)
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')  # (B, pred_len, M)

        # De-normalize
        outputs_time = outputs_time * stdev + means

        return {
            'time_embd_v1': x_emb,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'ncl_loss': ncl_loss,
        }

    # -----------------------
    # TASK: Classification
    # -----------------------
    def classification(self, x):
        """
        x: (B, L, M)
        """
        B, L, M = x.shape

        x_emb, _, _ = self.embd(x)             # (B, M, d_model)
        enc = self.in_layer(x_emb)             # (B, M, d_model)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=enc)
        outputs_time = outputs_time + enc      # (B, M, d_model)

        intermidiate_feat_time = tuple(
            self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))
        )

        outputs_time = outputs_time.reshape(B, -1)  # (B, M*d_model)
        outputs_time = self.out_layer(outputs_time)  # (B, num_class)

        return {
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
        }

    # -----------------------
    # TASK: Imputation
    # -----------------------
    def imputation(self, x, mask):
        """
        x: (B, L, M)
        mask: (B, L, M) 1 for observed, 0 for missing
        """
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x_masked = x.masked_fill(mask == 0, 0.0)
        stdev = torch.sqrt(
            torch.sum(x_masked ** 2, dim=1) / (torch.sum(mask == 1, dim=1) + 1e-5)
        ).unsqueeze(1).detach()
        x_norm = x_masked / (stdev + 1e-5)

        x_norm = rearrange(x_norm, 'b l m -> b m l')  # (B, M, L)
        x_norm = self.embd.linear(x_norm)             # (B, M, d_model)

        enc = self.in_layer(x_norm)                   # (B, M, d_model)
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=enc)
        outputs_time = outputs_time + enc

        intermidiate_feat_time = tuple(
            self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))
        )

        outputs_time = self.out_layer(outputs_time)            # (B, M, seq_len)
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')  # (B, L, M)

        outputs_time = outputs_time * stdev + means

        return {
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
        }

    # -----------------------
    # TASK: Anomaly detection
    # -----------------------
    def anomaly_detection(self, x):
        """
        x: (B, L, M)
        """
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = x / stdev

        x_norm = rearrange(x_norm, 'b l m -> b m l')  # (B, M, L)
        x_norm = self.embd.linear(x_norm)             # (B, M, d_model)

        enc = self.in_layer(x_norm)                   # (B, M, d_model)
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=enc)
        outputs_time = outputs_time + enc

        intermidiate_feat_time = tuple(
            self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))
        )

        outputs_time = self.out_layer(outputs_time)            # (B, M, seq_len)
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')  # (B, L, M)

        outputs_time = outputs_time * stdev + means

        return {
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
        }

    # -----------------------
    # MAIN FORWARD
    # -----------------------
    def forward(self, x, mask=None, owner_rows=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x, owner_rows=owner_rows)
        if self.task_name == 'classification':
            return self.classification(x)
        if self.task_name == "imputation":
            return self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x)
        raise ValueError(f"Unknown task_name: {self.task_name}")

