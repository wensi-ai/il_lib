import math
import torch
import torch.nn as nn
from il_lib.nn.common import get_activation, MLP
from il_lib.optim import default_optimizer_groups
from typing import Callable


class MLPResNetDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int,
        diffusion_step_embed_dim: int,
        num_blocks: int,
        hidden_dim: int,
        hidden_expand_ratio: int = 4,
        cond_encoder_hidden_expand_ratio: int = 2,
        cond_encoder_hidden_depth: int = 2,
        dropout_rate: None | float = None,
        use_layernorm: bool = False,
        activation: str | Callable = "swish",
    ):
        super().__init__()
        self.time_embd = LearnableFourierPosEmb(diffusion_step_embed_dim)
        self.cond_encoder = MLP(
            diffusion_step_embed_dim,
            hidden_dim=cond_encoder_hidden_expand_ratio * diffusion_step_embed_dim,
            output_dim=diffusion_step_embed_dim,
            hidden_depth=cond_encoder_hidden_depth,
            activation=activation,
        )
        self.reverse_network = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim + diffusion_step_embed_dim,
            hidden_dim=hidden_dim,
            hidden_expand_ratio=hidden_expand_ratio,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, diffusion_t: torch.Tensor):
        """
        x: (..., D), where D = input_dim
        diffusion_t: (..., 1), i.e., integer timesteps

        output: (..., D_output) where D_output = output_dim
        """
        assert diffusion_t.shape[-1] == 1
        t_ff = self.time_embd(
            diffusion_t
        )  # (..., D) where D = diffusion_step_embed_dim
        cond_enc = self.cond_encoder(t_ff)  # (..., D)

        reverse_input = torch.cat([x, cond_enc], dim=-1)  # (..., D_diff + D_input)
        return self.reverse_network(reverse_input)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "time_embd.*",
                "cond_encoder.*",
                "reverse_network.*",
            ],
        )
        return pg, pid


class LearnableFourierPosEmb(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        time_dim: int = 1,
    ):
        super().__init__()
        self.w = nn.Parameter(torch.randn(embd_dim // 2, time_dim))
        nn.init.normal_(self.w, std=0.2)

    def forward(self, t: torch.Tensor):
        """
        t: (..., time_dim)

        output: (..., embd_dim)
        """
        f = 2 * math.pi * t.unsqueeze(-2) @ self.w.T  # (..., 1, embd_dim // 2)
        f = f.squeeze(-2)  # (..., embd_dim // 2)
        rtn = torch.concat([torch.cos(f), torch.sin(f)], dim=-1)  # (..., embd_dim)
        return rtn


class MLPResNet(nn.Module):
    def __init__(
        self,
        *,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        hidden_expand_ratio: int = 4,
        output_dim: int,
        dropout_rate: None | float = None,
        use_layernorm: bool = False,
        activation: str | Callable = "swish",
    ):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    input_dim=hidden_dim,
                    hidden_expand_ratio=hidden_expand_ratio,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    use_layernorm=use_layernorm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_act = get_activation(activation)()
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_act(x)
        x = self.output_proj(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_expand_ratio: int = 4,
        activation: str | Callable,
        dropout_rate: None | float = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.dropout = None
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = None
        if use_layernorm:
            self.layernorm = nn.LayerNorm(input_dim)
        self.dense1 = nn.Linear(input_dim, input_dim * hidden_expand_ratio)
        self.act_layer = get_activation(activation)()
        self.dense2 = nn.Linear(input_dim * hidden_expand_ratio, input_dim)

    def forward(self, x):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        x = self.dense1(x)
        x = self.act_layer(x)
        x = self.dense2(x)

        return x + residual