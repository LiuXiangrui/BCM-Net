import torch
import torch.nn as nn

from Modules.EntropyModel import EntropyModel
from Modules.FeatsExtraction import FeatureExtractionModule
from Modules.I2CG import IntraSliceAndInterSliceContextGeneration


class Network(nn.Module):
    def __init__(self, bit_depth: int, channels_F: int = 64, channels_M: int = 120, channels_X: int = 1, K: int = 10, R: int = 8) -> None:
        """
        BCM-Net to losslessly compress and decompress residues.
        :param bit_depth: bit-depth of residues.
        :param channels_F: channels of features.
        :param channels_M: channels of contexts.
        :param channels_X: channels of residues.
        :param K: number of mixtures in the discrete logistic mixture model.
        :param R: number of residual blocks.
        """
        super().__init__()

        self.bit_depth = bit_depth

        self.i2cg = IntraSliceAndInterSliceContextGeneration(channels_F=channels_F, channels_M=channels_M, R=R)

        self.entropy_models = EntropyModel(K=K, channels_ctx=channels_M, channels_data=channels_X)

        self.feats_extract_forward_ref = FeatureExtractionModule(channels_X=channels_X, channels_F=channels_F, R=R)

        self.feats_extract_backward_ref = FeatureExtractionModule(channels_X=channels_X, channels_F=channels_F, R=R)

        self.feats_extract_lossy_rec = FeatureExtractionModule(channels_X=channels_X, channels_F=channels_F, R=R)

    @torch.no_grad()
    def compress(self, residues: torch.Tensor, x_tilde: torch.Tensor, x_min: int, x_max: int, ref_forward: torch.Tensor = None, ref_backward: torch.Tensor = None) -> list:
        """
        Compress residues.
        :param residues: residues to be compressed.
        :param x_tilde: lossy reconstructions.
        :param x_min: minimum value of residues.
        :param x_max: maximum value of residues.
        :param ref_forward: forward reference.
        :param ref_backward: backward reference.
        :return strings: bitstreams of residues.
        """
        if ref_backward is None and ref_forward is not None:
            ref_backward = ref_forward.clone().detach()  # duplicate the forward reference

        # normalize input data
        residues = residues * 1.
        x_tilde = x_tilde / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]
        if ref_forward is not None and ref_backward is not None:
            ref_forward = ref_forward / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]
            ref_backward = ref_backward / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]

        # extract reference features
        lossy_feats = self.feats_extract_lossy_rec(x_tilde)
        forward_ref_feats = self.feats_extract_forward_ref(ref_forward)
        backward_ref_feats = self.feats_extract_backward_ref(ref_backward)

        # extract prior features and estimate likelihoods
        prior_feats = self.i2cg(lossy_feats=lossy_feats, forward_ref_feats=forward_ref_feats, backward_ref_feats=backward_ref_feats)

        # estimate probability and compress residues
        strings = self.entropy_models.compress(residues, ctx=prior_feats, x_min=x_min, x_max=x_max)

        return strings

    @torch.no_grad()
    def decompress(self, strings: list, x_tilde: torch.Tensor, x_min: int, x_max: int, ref_forward: torch.Tensor = None, ref_backward: torch.Tensor = None) -> torch.Tensor:
        """
        Decompress residues.
        :param strings: bitstreams of residues.
        :param x_tilde: lossy reconstructions.
        :param x_min: minimum value of residues.
        :param x_max: maximum value of residues.
        :param ref_forward: forward reference.
        :param ref_backward: backward reference.
        :return residues: decompressed residues.
        """
        if ref_backward is None and ref_forward is not None:
            ref_backward = ref_forward.clone().detach()  # duplicate the forward reference

        # normalize input data
        x_tilde = x_tilde / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]
        if ref_forward is not None and ref_backward is not None:
            ref_forward = ref_forward / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]
            ref_backward = ref_backward / (2 ** self.bit_depth - 1) * 1.  # scale to [0, 1]

        # extract reference features
        lossy_feats = self.feats_extract_lossy_rec(x_tilde)
        forward_ref_feats = self.feats_extract_forward_ref(ref_forward)
        backward_ref_feats = self.feats_extract_backward_ref(ref_backward)

        # extract prior features and estimate likelihoods
        prior_feats = self.i2cg(lossy_feats=lossy_feats, forward_ref_feats=forward_ref_feats, backward_ref_feats=backward_ref_feats)

        # estimate probability and decompress residues
        residues = self.entropy_models.decompress(strings, ctx=prior_feats, x_min=x_min, x_max=x_max)

        return residues
