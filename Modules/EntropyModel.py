import torch
from einops import rearrange
from torch import nn as nn
from torchac import torchac

_NUM_PARAMS = 3
_NUM_SUB_IMG = 4
_LOG_SCALES_MIN = -7.
_BOUND_EPS = 0.001
_CDF_LOWER_BOUND = 1e-12
_BIN_WIDTH = 1

# if the level of cdf exceeds threshold, then split image to patches to calculate cdf to reduce memory consumption (useful when compressing 16-bit images)
_SEQUENTIAL_CDF_CAL_LEVEL_THRESHOLD = 1024
_MAX_PATCH_SIZE_CDF = 64


def to_data(symbols: torch.Tensor, x_min: int) -> torch.Tensor:
    data = symbols.to(torch.float32) * _BIN_WIDTH + x_min
    return data


def to_symbol(x: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
    symbols = torch.clamp(x, min=x_min, max=x_max)
    symbols = (symbols - x_min) / _BIN_WIDTH
    symbols = torch.round(symbols).long().to(torch.int16)
    return symbols


class DiscreteLogisticMixtureModel(nn.Module):
    def __init__(self, K: int) -> None:
        super().__init__()
        self.K = K

    @torch.no_grad()
    def cdf(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        """
        Calculate CDF of x.
        :param params: parameters with shape (B, _NUM_PARAMS * 3 * K, H, W).
        :param x_min: minimum value of x.
        :param x_max: maximum value of x.
        :return cdf: CDF of x with shape (B, C, H, W, L + 1).
        """
        L = x_max - x_min + 1
        if L > _SEQUENTIAL_CDF_CAL_LEVEL_THRESHOLD:  # split x into patches and sequentially calculate CDF of each patches to avoid OOM
            return self.sequential_cdf_calculate(params, x_min=x_min, x_max=x_max)
        else:
            return self.parallel_cdf_calculate(params, x_min=x_min, x_max=x_max)

    @torch.no_grad()
    def sequential_cdf_calculate(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        """
        Calculate CDF of each patches of x sequentially to avoid OOM.
        :param params: parameters with shape (B, _NUM_PARAMS * 3 * K, H, W).
        :param x_min: minimum value of x.
        :param x_max: maximum value of x.
        :return cdf: CDF of x with shape (B, C, H, W, L + 1).
        """
        L = x_max - x_min + 1

        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)
        B, _, C, _, H, W = params.shape

        weights_softmax = torch.softmax(params[:, 0, ...], dim=2).unsqueeze(dim=-1)  # shape (B, C, K, H, W, 1)
        means = params[:, 1, ...].unsqueeze(dim=-1)  # shape (B, C, K, H, W, 1)
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN).unsqueeze(dim=-1)  # shape (B, C, K, H, W, 1)
        inv_sigma = torch.exp(-log_scales)

        targets = torch.linspace(start=x_min - _BIN_WIDTH / 2, end=x_max + _BIN_WIDTH / 2, steps=L + 1, dtype=torch.float32, device=params.device)

        cdf = torch.zeros(B, C, H, W, L + 1, device=params.device)

        for i in range(H // _MAX_PATCH_SIZE_CDF):
            for j in range(W // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets *= inv_sigma[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets.sigmoid_()  # shape (B, C, K, _MAX_PATCH_SIZE_CDF, _MAX_PATCH_SIZE_CDF, L + 1)
                centered_targets *= weights_softmax[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                cdf[:, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF), :] = centered_targets.sum(dim=2)
        if H % _MAX_PATCH_SIZE_CDF != 0:
            start_idx = H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF
            for j in range(W // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :, start_idx:, j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets = centered_targets * inv_sigma[:, :, :, start_idx:, j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets.sigmoid_()  # shape (B, C, K, _MAX_PATCH_SIZE_CDF, _MAX_PATCH_SIZE_CDF, L + 1)
                centered_targets = centered_targets * weights_softmax[:, :, :, start_idx:, j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                cdf[:, :, start_idx:, j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)] = centered_targets.sum(dim=2)
        if W % _MAX_PATCH_SIZE_CDF != 0:
            start_idx = W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF
            for i in range(H // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), start_idx:]
                centered_targets = centered_targets * inv_sigma[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), start_idx:]
                centered_targets.sigmoid_()  # shape (B, C, K, _MAX_PATCH_SIZE_CDF, _MAX_PATCH_SIZE_CDF, L + 1)
                centered_targets = centered_targets * weights_softmax[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), start_idx:]
                cdf[:, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), start_idx:] = centered_targets.sum(dim=2)
        if H % _MAX_PATCH_SIZE_CDF != 0 and W % _MAX_PATCH_SIZE_CDF != 0:
            centered_targets = targets - means[:, :, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:, W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            centered_targets = centered_targets * inv_sigma[:, :, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:, W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            centered_targets.sigmoid_()  # shape (B, C, K, _MAX_PATCH_SIZE_CDF, _MAX_PATCH_SIZE_CDF, L + 1)
            centered_targets = centered_targets * weights_softmax[:, :, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:, W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            cdf[:, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:, W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:] = centered_targets.sum(dim=2)
        return cdf

    @torch.no_grad()
    def parallel_cdf_calculate(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        """
        Calculate CDF of x in parallel.
        :param params: parameters with shape (B, _NUM_PARAMS * 3 * K, H, W).
        :param x_min: minimum value of x.
        :param x_max: maximum value of x.
        :return cdf: CDF of x with shape (B, C, H, W, L + 1).
        """
        L = x_max - x_min + 1

        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)

        weights_softmax = torch.softmax(params[:, 0, ...], dim=2)
        means = params[:, 1, ...]
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN)
        inv_sigma = torch.exp(-log_scales).unsqueeze(dim=-1)

        targets = torch.linspace(start=x_min - _BIN_WIDTH / 2, end=x_max + _BIN_WIDTH / 2, steps=L + 1, dtype=torch.float32, device=params.device)

        centered_targets = targets - means.unsqueeze(dim=-1)  # shape (B, C, K, H, W, L + 1)
        centered_targets = centered_targets * inv_sigma
        centered_targets.sigmoid_()  # shape (B, C, K, H, W, L + 1)
        centered_targets = centered_targets * weights_softmax.unsqueeze(dim=-1)
        cdf = centered_targets.sum(dim=2)  # shape (B, C, H, W, L + 1)
        return cdf


class ParametersEstimatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AutoregressiveContextExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EntropyModel(nn.Module):
    def __init__(self, K: int, channels_ctx: int, channels_data: int) -> None:
        """
        Entropy model to compress and decompress data.
        :param K: number of mixtures in the discrete logistic mixture model.
        :param channels_ctx: channels of contexts.
        :param channels_data: channels of data.
        """
        super().__init__()
        self.auto_ctx_extraction = nn.ModuleList([
            AutoregressiveContextExtraction(in_channels=i * channels_data, out_channels=channels_ctx) for i in
            range(1, _NUM_SUB_IMG)
        ])

        self.params_estimator = nn.ModuleList([
            ParametersEstimatorBlock(in_channels=(1 + int(i > 0)) * channels_ctx, out_channels=channels_data * K * _NUM_PARAMS) for
            i in range(_NUM_SUB_IMG)
        ])

        self.discrete_logistic_mixture_model = DiscreteLogisticMixtureModel(K=K)

    @torch.no_grad()
    def compress(self, x: torch.Tensor, ctx: torch.Tensor, x_min: int, x_max: int) -> list:
        """
        Compress input data to bitstreams.
        """
        # split to four sub images
        sub_x = self.spatial_split(x)
        strings = []
        # autoregressive compress sub images
        for i in range(_NUM_SUB_IMG):
            # estimate parameters
            if i == 0:  # the first sub image
                params = self.params_estimator[i](ctx)
            else:
                auto_ctx = self.auto_ctx_extraction[i - 1](torch.cat(sub_x[:i], dim=1))  # extract autoregressive context
                params = self.params_estimator[i](torch.cat([ctx, auto_ctx], dim=1))
            # calculate cdf
            cdf = self.discrete_logistic_mixture_model.cdf(params, x_min=x_min, x_max=x_max).cpu()
            # convert to symbols
            symbols = to_symbol(sub_x[i], x_min=x_min, x_max=x_max).cpu()
            # compress to bitstreams
            strings.append(torchac.encode_float_cdf(cdf_float=cdf, sym=symbols))
        return strings

    @torch.no_grad()
    def decompress(self, strings: list, ctx: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        """
        Decompress input bitstreams to original data.
        """
        assert len(strings) == _NUM_SUB_IMG, f'Number of bitstreams {len(strings)} is not equal to {_NUM_SUB_IMG}.'
        sub_x = []
        # autoregressive decompress sub images
        for i in range(_NUM_SUB_IMG):
            # estimate parameters
            if i == 0:
                params = self.params_estimator[i](ctx)
            else:
                auto_ctx = self.auto_ctx_extraction[i - 1](torch.cat(sub_x[:i], dim=1))
                params = self.params_estimator[i](torch.cat([ctx, auto_ctx], dim=1))
            # estimate cdf
            cdf = self.discrete_logistic_mixture_model.cdf(params, x_min=x_min, x_max=x_max).cpu()
            # decompress from bitstreams
            symbols = torchac.decode_float_cdf(cdf_float=cdf, byte_stream=strings[i])
            # convert to data
            sub_x.append(to_data(symbols, x_min=x_min).to(ctx.device))
        return self.spatial_merge(sub_x)

    @staticmethod
    def spatial_split(x: torch.Tensor) -> tuple:
        """
        Split input image to four sub images.
        """
        upper_left = x[:, :, ::2, ::2]
        upper_right = x[:, :, ::2, 1::2]
        bottom_left = x[:, :, 1::2, ::2]
        bottom_right = x[:, :, 1::2, 1::2]
        return upper_left, bottom_right, upper_right, bottom_left

    @staticmethod
    def spatial_merge(sub_list: list) -> torch.Tensor:
        """
        Merge four sub images to one image.
        """
        assert len(sub_list) == _NUM_SUB_IMG
        upper_left, bottom_right, upper_right, bottom_left = sub_list
        B, C, H_half, W_half = upper_left.shape
        x = torch.zeros(B, C, H_half * 2, W_half * 2, device=upper_left.device)
        x[:, :, ::2, ::2] = upper_left
        x[:, :, ::2, 1::2] = upper_right
        x[:, :, 1::2, ::2] = bottom_left
        x[:, :, 1::2, 1::2] = bottom_right
        return x
