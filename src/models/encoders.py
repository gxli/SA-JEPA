import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



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


class D4InvariantWrapper(nn.Module):
    """
    Enforce exact 4-way rotational invariance by shared-weight test-time group pooling.

    For each k in {0,1,2,3}:
      1) rotate input(s) by 90*k
      2) run base encoder
      3) inverse-rotate output by -90*k
    Then pool across the 4 responses (max or mean).
    """

    def __init__(self, base_encoder: nn.Module, pool: str = "mean"):
        super().__init__()
        self.base_encoder = base_encoder
        self.pool = str(pool).lower()
        if self.pool not in ("max", "mean"):
            raise ValueError(f"Unsupported D4 pool={pool}. Use 'max' or 'mean'.")

    @staticmethod
    def _rot(x: torch.Tensor, k: int) -> torch.Tensor:
        return torch.rot90(x, k=k, dims=(-2, -1))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        outs = []
        for k in range(4):
            xk = self._rot(x, k)
            kw = {}
            for name, val in kwargs.items():
                if torch.is_tensor(val) and val.ndim >= 3 and val.shape[-2:] == x.shape[-2:]:
                    kw[name] = self._rot(val, k)
                else:
                    kw[name] = val
            yk = self.base_encoder(xk, **kw)
            outs.append(self._rot(yk, -k))
        y = torch.stack(outs, dim=2)  # B,C,4,H,W
        if self.pool == "max":
            y, _ = torch.max(y, dim=2)
            return y
        return torch.mean(y, dim=2)


class ResCNNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: Optional[int] = None,
        norm_type: str = "groupnorm",
        norm_groups: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if hidden is None:
            hidden = channels
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=0),
            make_norm2d(hidden, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=0),
            make_norm2d(channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
        )
    def forward(self, x):
        # Keep the residual stream linear to preserve signal propagation.
        return x + self.net(x)


class ResCNNDenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 6,
        final_norm: bool = True,
        norm_type: str = "groupnorm",
        norm_groups: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0),
            make_norm2d(hidden_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResCNNBlock(
                    hidden_channels,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = (
            make_norm2d(latent_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps)
            if final_norm
            else nn.Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


def _init_scale_aware_state(
    module: nn.Module,
    *,
    scales,
    scale_feat_channels: int,
    fusion_type: str,
    cdd_append_last_residual: bool,
) -> None:
    module.scales = tuple(float(s) for s in scales)
    module.num_scales = len(module.scales)
    module.scale_feat_channels = int(scale_feat_channels)
    module.fusion_type = str(fusion_type).lower()
    module.cdd_append_last_residual = bool(cdd_append_last_residual)
    if module.fusion_type not in ("concat", "topdown"):
        raise ValueError(f"Unsupported fusion_type={fusion_type}. Use 'concat' or 'topdown'.")

    logs = torch.log(torch.tensor(module.scales, dtype=torch.float32))
    if logs.numel() > 1:
        logs = (logs - logs.mean()) / logs.std(unbiased=False).clamp_min(1e-6)
    else:
        logs = logs * 0.0
    module.register_buffer("scale_codes", logs.view(1, module.num_scales, 1, 1), persistent=False)
    if module.fusion_type == "topdown":
        module.fusion_proj = nn.ModuleList(
            [nn.Conv2d(module.scale_feat_channels, module.scale_feat_channels, kernel_size=1) for _ in range(module.num_scales)]
        )


def _scale_aware_features(module: nn.Module, fields: torch.Tensor, mask_tokens=None) -> torch.Tensor:
    if fields.ndim != 4:
        raise ValueError(f"Expected fields B,S,H,W, got {tuple(fields.shape)}")
    b, s, h, w = fields.shape
    if mask_tokens is None:
        mask_tokens = torch.zeros_like(fields)

    if s > module.num_scales:
        n_extra = s - module.num_scales
        if module.cdd_append_last_residual:
            base = fields[:, :module.num_scales, :, :]
            extra = fields[:, module.num_scales:, :, :]
            last = base[:, -1:, :, :] + extra.sum(dim=1, keepdim=True)
            fields = torch.cat([base[:, :-1, :, :], last], dim=1)
        else:
            fields = fields[:, :module.num_scales, :, :]
        print(
            f"[{module.__class__.__name__}] WARNING: Truncated {n_extra} extra channel(s) "
            f"(append_last_residual={module.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
        )
    elif s < module.num_scales:
        n_missing = module.num_scales - s
        if module.cdd_append_last_residual:
            n_split = n_missing + 1
            split = fields[:, -1:, :, :] / float(n_split)
            fields = torch.cat([fields[:, :-1, :, :], split.expand(-1, n_split, -1, -1)], dim=1)
            mask_tokens = torch.cat([mask_tokens[:, :-1, :, :], mask_tokens[:, -1:, :, :].expand(-1, n_split, -1, -1)], dim=1)
        else:
            zeros = torch.zeros(b, n_missing, h, w, dtype=fields.dtype, device=fields.device)
            fields = torch.cat([fields, zeros], dim=1)
            mask_tokens = torch.cat([mask_tokens, zeros], dim=1)
        print(
            f"[{module.__class__.__name__}] WARNING: Padded {n_missing} missing channel(s) "
            f"(append_last_residual={module.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
        )

    s = module.num_scales
    mask_tokens = mask_tokens[:, :s, :, :]
    if mask_tokens.shape != fields.shape:
        raise ValueError(f"mask_tokens shape must match fields shape. fields={tuple(fields.shape)} mask={tuple(mask_tokens.shape)}")

    scale_maps = module.scale_codes.to(dtype=fields.dtype, device=fields.device).expand(b, s, h, w)
    feats = [
        module.adapter(torch.stack([fields[:, i], mask_tokens[:, i], scale_maps[:, i]], dim=1))
        for i in range(s)
    ]
    if module.fusion_type == "topdown":
        fused = [None] * s
        running = None
        for rev_i, feat in enumerate(reversed(feats)):
            idx = s - 1 - rev_i
            if running is None:
                running = feat
            else:
                if running.shape[-2:] != feat.shape[-2:]:
                    running = F.interpolate(running, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                running = feat + running
            fused[idx] = module.fusion_proj[idx](running)
        feats = fused
    return torch.cat(feats, dim=1)


def _make_convnext_scale_adapter(
    scale_feat_channels: int,
    kernel_size: int,
    *,
    use_reflect_padding: bool,
    adapter_norm: bool,
) -> nn.Sequential:
    pad = int(kernel_size) // 2
    layers = []
    if use_reflect_padding and pad > 0:
        layers.append(nn.ReflectionPad2d(pad))
        pad = 0
    layers.append(nn.Conv2d(3, scale_feat_channels, kernel_size=int(kernel_size), padding=pad))
    if adapter_norm:
        layers.append(LayerNorm2d(scale_feat_channels))
    layers += [nn.GELU(), nn.Conv2d(scale_feat_channels, scale_feat_channels, kernel_size=1)]
    if adapter_norm:
        layers.append(LayerNorm2d(scale_feat_channels))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)


class CDDScaleAwareConvNeXtEncoder(nn.Module):
    """
    Scale-aware CDD pyramid encoder.

    Input:
      fields:      B x S x H x W
      mask_tokens: B x S x H x W

    Per scale:
      [field_s, mask_s, normalized_log_sigma_s] -> shared adapter
    Then concatenate scale features and feed ConvNeXt dense encoder.
    """

    def __init__(
        self,
        scales,
        hidden_channels: int,
        latent_channels: int,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        scale_feat_channels: int = 8,
        adapter_kernel_size: int = 3,
        fusion_type: str = "concat",
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
        head_bias: bool = True,
        cdd_append_last_residual: bool = True,
        adapter_norm: bool = True,
        use_grn: bool = True,
        stem_norm: bool = True,
        dilations=None,
    ):
        super().__init__()
        _init_scale_aware_state(
            self,
            scales=scales,
            scale_feat_channels=scale_feat_channels,
            fusion_type=fusion_type,
            cdd_append_last_residual=cdd_append_last_residual,
        )
        self.adapter = _make_convnext_scale_adapter(
            self.scale_feat_channels,
            adapter_kernel_size,
            use_reflect_padding=use_reflect_padding,
            adapter_norm=adapter_norm,
        )

        print(
            f"[CDDScaleAwareConvNeXt] depth={depth}, dilations={dilations}, "
            f"stem_norm={stem_norm}, adapter_norm={adapter_norm}, "
            f"final_norm={final_norm}({final_norm_type}), grn={use_grn}"
        )
        self.convnext = ConvNeXtDenseEncoder(
            in_channels=self.num_scales * self.scale_feat_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            depth=depth,
            kernel_size=kernel_size,
            expansion=expansion,
            use_reflect_padding=use_reflect_padding,
            final_norm=final_norm,
            final_norm_type=final_norm_type,
            head_bias=head_bias,
            use_grn=use_grn,
            stem_norm=stem_norm,
            dilations=dilations,
        )
    def forward(self, fields: torch.Tensor, mask_tokens=None) -> torch.Tensor:
        return self.convnext(_scale_aware_features(self, fields, mask_tokens))


class CDDScaleAwareResCNNEncoder(nn.Module):
    """
    Scale-aware CDD pyramid encoder using ResCNN blocks.

    Input:
      fields:      B x S x H x W
      mask_tokens: B x S x H x W

    Per scale:
      [field_s, mask_s, normalized_log_sigma_s] -> shared adapter
    Then fuse scale features and feed ResCNN dense encoder.
    """

    def __init__(
        self,
        scales,
        hidden_channels: int,
        latent_channels: int,
        depth: int = 4,
        scale_feat_channels: int = 8,
        adapter_kernel_size: int = 3,
        fusion_type: str = "concat",
        final_norm: bool = True,
        norm_type: str = "groupnorm",
        norm_groups: int = 1,
        norm_eps: float = 1e-5,
        cdd_append_last_residual: bool = True,
    ):
        super().__init__()
        _init_scale_aware_state(
            self,
            scales=scales,
            scale_feat_channels=scale_feat_channels,
            fusion_type=fusion_type,
            cdd_append_last_residual=cdd_append_last_residual,
        )

        pad = int(adapter_kernel_size) // 2
        self.adapter = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(3, self.scale_feat_channels, kernel_size=int(adapter_kernel_size), padding=0),
            make_norm2d(self.scale_feat_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(
                self.scale_feat_channels,
                self.scale_feat_channels,
                kernel_size=int(adapter_kernel_size),
                padding=0,
            ),
            make_norm2d(self.scale_feat_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
        )
        self.rescnn = ResCNNDenseEncoder(
            in_channels=self.num_scales * self.scale_feat_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            depth=depth,
            final_norm=final_norm,
            norm_type=norm_type,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
        )

    def forward(self, fields: torch.Tensor, mask_tokens=None) -> torch.Tensor:
        return self.rescnn(_scale_aware_features(self, fields, mask_tokens))
