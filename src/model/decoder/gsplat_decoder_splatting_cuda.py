from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import math

from ...dataset import DatasetCfg
from ..types import Gaussians
from .decoder import DepthRenderingMode
from gsplat.rendering import rasterization
from .decoder import Decoder, DecoderOutput


@dataclass
class GSplatDecoderSplattingCUDACfg:
    name: Literal["gsplat"]
    scale_invariant: bool
    use_covariances: bool | None = True


class GSplatDecoderSplattingCUDA(Decoder[GSplatDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GSplatDecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_radii: bool = False,
    ) -> DecoderOutput:
        height, width = image_shape
        means = gaussians.means  # [B, G, 3]
        # NOTE: rasterization does normalization internally
        quats = gaussians.rotations_unnorm  # [B, G, 4]
        scales = gaussians.scales  # [B, G, 3]
        if self.cfg.use_covariances:
            covars = gaussians.covariances  # [B, G, 3, 3]
        else:
            covars = None
        opacities = gaussians.opacities  # [B, G]
        colors = gaussians.harmonics.permute(0, 1, 3, 2)  # [B, G, d_sh, 3]
        sh_degree = int(math.sqrt(colors.shape[-2])) - 1  # d_sh = (degree + 1) ** 2
        viewmats = extrinsics.inverse()  # [B, V, 4, 4]
        intrinsics = intrinsics.clone()  # [B, V, 3, 3]
        # scale to the image shape
        intrinsics[:, :, 0] *= width
        intrinsics[:, :, 1] *= height

        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            near_plane=near[0, 0].item(),  # expect float
            far_plane=far[0, 0].item(),
            eps2d=0.1,
            rasterize_mode="antialiased",
            packed=True,
            absgrad=False,
            sparse_grad=False,
            render_mode="RGB+ED",
            covars=covars,
            # gsplat 1.x appends the depth background internally for RGB+ED
            # after validating against the RGB channel count, so a non-None
            # background trips its own shape assertion.  ReSplat/QUEEN use black
            # backgrounds in these diagnostics, matching gsplat's None default.
            backgrounds=None,
        )

        color = render_colors[..., :3].permute(0, 1, 4, 2, 3)  # [B, V, 3, H, W]
        depth = render_colors[..., -1]  # [B, V, H, W]

        return DecoderOutput(
            color,
            depth=depth,
            accumulated_alpha=render_alphas.squeeze(-1)  # [B, V, H, W]
        )
