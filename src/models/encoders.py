import torch
import torch.nn as nn
import torch.nn.functional as F



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


def _require_odd_kernel_size(kernel_size: int, name: str) -> int:
    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer for dense phase alignment, got {kernel_size!r}.")
    return k


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
        kernel_size = _require_odd_kernel_size(kernel_size, "ConvNeXtDenseBlock.kernel_size")
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
        kernel_size = _require_odd_kernel_size(kernel_size, "ConvNeXtDenseEncoder.kernel_size")
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


class CDDScaleAwareConvNeXtEncoder(nn.Module):
    """
    Scale-aware CDD pyramid encoder.

    Input:
      fields:      B x S x H x W
      mask_tokens: B x S x H x W

    Per scale:
      [field_s, mask_s, normalized_log_sigma_s] -> shared adapter.
    Then concatenate scale features and feed a dense ConvNeXt encoder.
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
        kernel_size = _require_odd_kernel_size(kernel_size, "CDDScaleAwareConvNeXtEncoder.kernel_size")
        adapter_kernel_size = _require_odd_kernel_size(
            adapter_kernel_size,
            "CDDScaleAwareConvNeXtEncoder.adapter_kernel_size",
        )
        self.scales = tuple(float(s) for s in scales)
        self.num_scales = len(self.scales)
        self.scale_feat_channels = int(scale_feat_channels)
        self.fusion_type = str(fusion_type).lower()
        self.cdd_append_last_residual = bool(cdd_append_last_residual)
        if self.fusion_type not in ("concat", "topdown"):
            raise ValueError(f"Unsupported fusion_type={fusion_type}. Use 'concat' or 'topdown'.")

        logs = torch.log(torch.tensor(self.scales, dtype=torch.float32))
        if logs.numel() > 1:
            logs = (logs - logs.mean()) / logs.std(unbiased=False).clamp_min(1e-6)
        else:
            logs = logs * 0.0
        self.register_buffer("scale_codes", logs.view(1, self.num_scales, 1, 1), persistent=False)

        pad = int(adapter_kernel_size) // 2
        if use_reflect_padding and pad > 0:
            adapter_layers = [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(3, self.scale_feat_channels, kernel_size=int(adapter_kernel_size), padding=0),
            ]
        else:
            adapter_layers = [
                nn.Conv2d(3, self.scale_feat_channels, kernel_size=int(adapter_kernel_size), padding=pad),
            ]
        if adapter_norm:
            adapter_layers.append(LayerNorm2d(self.scale_feat_channels))
        adapter_layers += [
            nn.GELU(),
            nn.Conv2d(self.scale_feat_channels, self.scale_feat_channels, kernel_size=1),
        ]
        if adapter_norm:
            adapter_layers.append(LayerNorm2d(self.scale_feat_channels))
        adapter_layers.append(nn.GELU())
        self.adapter = nn.Sequential(*adapter_layers)

        dil_list = _normalize_convnext_dilations(dilations, depth)
        block_fov = 1
        for d in dil_list:
            block_fov += (int(kernel_size) - 1) * int(d)
        print(
            f"[CDDScaleAwareConvNeXt] depth={depth}, dilations={dil_list}, "
            f"conv_footprint={block_fov}px, stem_norm={stem_norm}, adapter_norm={adapter_norm}, "
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
        if self.fusion_type == "topdown":
            self.fusion_proj = nn.ModuleList(
                [
                    nn.Conv2d(self.scale_feat_channels, self.scale_feat_channels, kernel_size=1)
                    for _ in range(self.num_scales)
                ]
            )

    def forward(self, fields: torch.Tensor, mask_tokens=None) -> torch.Tensor:
        if fields.ndim != 4:
            raise ValueError(f"Expected fields B,S,H,W, got {tuple(fields.shape)}")
        b, s, h, w = fields.shape
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        if s != self.num_scales:
            if s > self.num_scales:
                n_extra = s - self.num_scales
                if self.cdd_append_last_residual:
                    base = fields[:, : self.num_scales, :, :]
                    extra = fields[:, self.num_scales :, :, :]
                    last = base[:, self.num_scales - 1 : self.num_scales, :, :] + extra.sum(dim=1, keepdim=True)
                    fields = torch.cat([base[:, : self.num_scales - 1, :, :], last], dim=1)
                else:
                    fields = fields[:, : self.num_scales, :, :]
                print(
                    f"[{self.__class__.__name__}] WARNING: Truncated {n_extra} extra channel(s) "
                    f"(append_last_residual={self.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
                )
            else:
                n_missing = self.num_scales - s
                if self.cdd_append_last_residual:
                    residual = fields[:, -1:, :, :]
                    res_mask = mask_tokens[:, -1:, :, :]
                    n_split = n_missing + 1
                    split = residual / float(n_split)
                    fields = torch.cat([fields[:, :-1, :, :], split.expand(-1, n_split, -1, -1)], dim=1)
                    mask_tokens = torch.cat([mask_tokens[:, :-1, :, :], res_mask.expand(-1, n_split, -1, -1)], dim=1)
                else:
                    zeros = torch.zeros(b, n_missing, h, w, dtype=fields.dtype, device=fields.device)
                    fields = torch.cat([fields, zeros], dim=1)
                    mask_tokens = torch.cat([mask_tokens, zeros], dim=1)
                print(
                    f"[{self.__class__.__name__}] WARNING: Padded {n_missing} missing channel(s) "
                    f"(append_last_residual={self.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
                )
            s = self.num_scales

        mask_tokens = mask_tokens[:, :s, :, :]
        if mask_tokens.shape != fields.shape:
            raise ValueError(
                f"mask_tokens shape must match fields shape. fields={tuple(fields.shape)} mask={tuple(mask_tokens.shape)}"
            )

        scale_maps = self.scale_codes.to(dtype=fields.dtype, device=fields.device).expand(b, s, h, w)
        feats = []
        for i in range(s):
            xi = torch.stack([fields[:, i], mask_tokens[:, i], scale_maps[:, i]], dim=1)
            feats.append(self.adapter(xi))
        if self.fusion_type == "topdown":
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
                fused[idx] = self.fusion_proj[idx](running)
            feats = fused
        x = torch.cat(feats, dim=1)
        return self.convnext(x)
