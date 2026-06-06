import torch
import torch.nn as nn



class LayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.eps = float(eps)

    def forward(self, x):
        # x: B,H,W,C
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


def _valid_groups(channels: int, groups: int) -> int:
    g = max(1, int(groups))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def make_norm2d(channels: int, norm_type: str = "layernorm", norm_groups: int = 8, norm_eps: float = 1e-6) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "layernorm":
        return LayerNorm2d(channels, eps=float(norm_eps))
    if kind == "groupnorm":
        return nn.GroupNorm(_valid_groups(channels, norm_groups), channels, eps=float(norm_eps))
    raise ValueError(f"Unsupported norm_type={norm_type}. Use 'layernorm' or 'groupnorm'.")


def _normalize_convnext_dilations(dilations, depth: int) -> list[int]:
    depth = int(depth)
    if dilations is None:
        return [1] * depth
    values = [int(d) for d in dilations]
    if not values:
        raise ValueError("ConvNeXt dilations must contain at least one value.")
    if any(d <= 0 for d in values):
        raise ValueError(f"ConvNeXt dilations must be positive integers, got {values}.")
    if len(values) < depth:
        reps = (depth + len(values) - 1) // len(values)
        values = (values * reps)[:depth]
    elif len(values) > depth:
        values = values[:depth]
    return values


class ConvNeXtDenseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size: int = 7,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        use_reflect_padding: bool = True,
        use_grn: bool = True,
    ):
        super().__init__()
        self.use_grn = bool(use_grn)
        self.dilation = int(dilation)
        pad = (int(kernel_size) // 2) * self.dilation
        if use_reflect_padding:
            self.dwconv = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=0,
                    dilation=self.dilation,
                    groups=channels,
                ),
            )
        else:
            self.dwconv = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=self.dilation,
                groups=channels,
            )
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, expansion * channels)
        self.act = nn.GELU()
        self.grn = GRN(expansion * channels) if self.use_grn else nn.Identity()
        self.pw2 = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtDenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
        head_bias: bool = True,
        dilations=None,
        use_grn: bool = True,
        stem_norm: bool = True,
    ):
        super().__init__()
        depth = int(depth)
        dilations = _normalize_convnext_dilations(dilations, depth)
        self.dilations = tuple(dilations)

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1) if use_reflect_padding else nn.Identity(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0 if use_reflect_padding else 1),
            LayerNorm2d(hidden_channels) if stem_norm else nn.Identity(),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ConvNeXtDenseBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    use_reflect_padding=use_reflect_padding,
                    use_grn=use_grn,
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=head_bias)
        if not final_norm:
            self.final_norm = nn.Identity()
        else:
            ntype = str(final_norm_type).lower()
            if ntype == "batchnorm":
                self.final_norm = nn.BatchNorm2d(latent_channels, track_running_stats=False)
            elif ntype in ("layernorm", ""):
                self.final_norm = LayerNorm2d(latent_channels)
            else:
                raise ValueError(
                    f"Unsupported final_norm_type={final_norm_type}. "
                    "Use 'layernorm' or 'batchnorm'."
                )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


class EscnnC4PyramidEncoder(nn.Module):
    """
    C4 rotation-equivariant pyramid encoder using escnn.

    Input is a normal BCHW tensor with the same channel contract as
    convnext_dense_pyramid: per-scale CDD channels concatenated with per-scale
    mask-token channels. escnn lifts those trivial input fields into regular
    C4 fields, applies equivariant R2Conv blocks, then group-pools to return a
    standard invariant BCHW tensor.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
    ):
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as exc:
            raise ImportError(
                "escnn_c4_pyramid requires the optional dependency 'escnn'. "
                "Install it with: pip install escnn"
            ) from exc

        depth = int(depth)
        hidden_channels = int(hidden_channels)
        latent_channels = int(latent_channels)
        kernel_size = int(kernel_size)
        padding = kernel_size // 2

        self.enn = enn
        self.r2_act = gspaces.rot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.r2_act, int(in_channels) * [self.r2_act.trivial_repr])
        self.hidden_type = enn.FieldType(self.r2_act, hidden_channels * [self.r2_act.regular_repr])
        self.out_type = enn.FieldType(self.r2_act, latent_channels * [self.r2_act.regular_repr])

        self.lift = enn.SequentialModule(
            enn.R2Conv(self.in_type, self.hidden_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.hidden_type),
            enn.ReLU(self.hidden_type, inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                enn.SequentialModule(
                    enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=kernel_size, padding=padding, bias=False),
                    enn.InnerBatchNorm(self.hidden_type),
                    enn.ReLU(self.hidden_type, inplace=True),
                )
                for _ in range(depth)
            ]
        )
        self.head = enn.R2Conv(self.hidden_type, self.out_type, kernel_size=1, padding=0, bias=True)
        self.gpool = enn.GroupPooling(self.out_type)
        if not final_norm:
            self.final_norm = nn.Identity()
        else:
            ntype = str(final_norm_type).lower()
            if ntype == "batchnorm":
                self.final_norm = nn.BatchNorm2d(latent_channels, track_running_stats=False)
            elif ntype in ("layernorm", ""):
                self.final_norm = LayerNorm2d(latent_channels)
            else:
                raise ValueError(
                    f"Unsupported final_norm_type={final_norm_type}. "
                    "Use 'layernorm' or 'batchnorm'."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.enn.GeometricTensor(x, self.in_type)
        gx = self.lift(gx)
        for block in self.blocks:
            gx = gx + block(gx)
        gx = self.head(gx)
        gx = self.gpool(gx)
        return self.final_norm(gx.tensor)
