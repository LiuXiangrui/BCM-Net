import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.BasicBlock import AtrousConv, ResBlock, LayerNorm, FeedForward, CrissCrossSelfAttention, BidirectionalCrossAttention, AffineTransform


class SymmetryBasedIntraSliceContextGeneration(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local_fusion = AtrousConv(in_channels=channels, out_channels=channels)

        self.affine_transform = AffineTransform(channels=channels)
        self.criss_cross_self_attn = CrissCrossSelfAttention(channels=channels)
        self.feed_forward = FeedForward(channels=channels)

        self.norm_before_attn = LayerNorm(channels=channels)
        self.norm_after_attn = LayerNorm(channels=channels)

        self.bottom = nn.Sequential(
            ResBlock(channels=channels),
            ResBlock(channels=channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.local_fusion(x)
        affine_grid, inv_affine_grid = self.affine_transform(y)
        y = F.grid_sample(y, grid=affine_grid, align_corners=False)
        y = self.feed_forward(torch.cat([y, self.norm_after_attn(self.criss_cross_self_attn(self.norm_before_attn(y)))], dim=1))
        y = F.grid_sample(y, grid=inv_affine_grid, align_corners=False)
        return x + self.bottom(y)


class BidirectionalInterSliceContextGeneration(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bidirectional_cross_attn = BidirectionalCrossAttention(channels=channels)
        self.feed_forward = FeedForward(channels=channels)

        self.norm_before_attn_x = LayerNorm(channels=channels)
        self.norm_before_attn_ref_1 = LayerNorm(channels=channels)
        self.norm_before_attn_ref_2 = LayerNorm(channels=channels)

        self.norm_after_attn = LayerNorm(channels=channels)

    def forward(self, x: torch.Tensor, ref_1: torch.Tensor, ref_2: torch.Tensor) -> torch.Tensor:
        y = self.bidirectional_cross_attn(self.norm_before_attn_x(x),
                                          ref_1=self.norm_before_attn_ref_1(ref_1),  ref_2=self.norm_before_attn_ref_2(ref_2))
        return x + self.feed_forward(torch.cat([x, self.norm_after_attn(y)], dim=1))


class IntraSliceAndInterSliceContextGeneration(nn.Module):
    def __init__(self, channels_F: int, channels_M: int, R: int) -> None:
        """
        Intra-slice and inter-slice context generation module.
        :param channels_F: channels of features.
        :param channels_M: channels of contexts.
        :param R: number of residual blocks.
        """
        super().__init__()

        self.lossy_res_blocks = nn.Sequential(*[ResBlock(channels=channels_F) for _ in range(R)])

        self.forward_res_blocks = nn.Sequential(*[ResBlock(channels=channels_F) for _ in range(R)])

        self.backward_res_blocks = nn.Sequential(*[ResBlock(channels=channels_F) for _ in range(R)])

        self.symmetry_based_intra_slice_ctx_generation = SymmetryBasedIntraSliceContextGeneration(channels=channels_F)

        self.bidirectional_inter_slice_ctx_generation = BidirectionalInterSliceContextGeneration(channels=channels_F)

        self.ctx_fusion_head_wo_ref = nn.Sequential(
            nn.Conv2d(in_channels=channels_F, out_channels=channels_M, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.ctx_fusion_head_with_ref = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels_F, out_channels=channels_M, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.ctx_fusion_body = nn.Sequential(*[ResBlock(channels=channels_M) for _ in range(R)])

    def forward(self, lossy_feats: torch.Tensor, forward_ref_feats: torch.Tensor = None, backward_ref_feats: torch.Tensor = None) -> torch.Tensor:
        intra_feats = self.lossy_res_blocks(lossy_feats)
        intra_ctx = self.symmetry_based_intra_slice_ctx_generation(intra_feats)

        if forward_ref_feats is not None and backward_ref_feats is not None:  # use both intra-slice and inter-slice context
            forward_ref_feats = self.forward_res_blocks(forward_ref_feats)
            backward_ref_feats = self.backward_res_blocks(backward_ref_feats)
            inter_ctx = self.bidirectional_inter_slice_ctx_generation(intra_feats, ref_1=forward_ref_feats, ref_2=backward_ref_feats)
            prior_feats = self.ctx_fusion_head_with_ref(torch.cat([intra_ctx, inter_ctx], dim=1))
        else:  # only use intra-slice context
            prior_feats = self.ctx_fusion_head_wo_ref(intra_ctx)
        prior_feats = self.ctx_fusion_body(prior_feats)

        return prior_feats
