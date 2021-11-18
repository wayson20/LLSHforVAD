import torch
from torch import nn


class FeatureTransform(nn.Module):
    def __init__(self, in_dim: list):
        super().__init__()
        self.__in_dim = in_dim
        self.feature_norm = nn.LayerNorm(self.ddim, elementwise_affine=False)

    @property
    def ddim(self):
        dim = 1
        for _d in self.__in_dim:
            dim *= _d
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, H, W, C]
        x = x.flatten(1, -1)  # [batch, ddim]
        x = self.feature_norm(x)
        return x  # [batch, ddim]


class _HashLayer(nn.Module):
    def __init__(self, in_dim, len_hash_code):
        super().__init__()
        self.hash_layer = nn.Linear(in_dim, len_hash_code, bias=False)
        self.binary_func = nn.Sigmoid()

        nn.init.xavier_uniform_(self.hash_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ddim]
        _hash_code = self.binary_func(self.hash_layer(x))
        return _hash_code  # [batch, len_hash_code]


class HashEncoder(nn.Module):
    def __init__(self, in_dim, num_hash_layer, len_hash_code):
        super().__init__()
        self.hash_layers = nn.ModuleList([_HashLayer(in_dim, len_hash_code) for i in range(num_hash_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ddim]
        hash_codes = torch.stack([layer(x) for layer in self.hash_layers], 1)
        return hash_codes  # [batch, num_hash_layer, len_hash_code]


class HashNet(nn.Module):
    def __init__(self, feat_dim, num_hash_layer, len_hash_code):
        super().__init__()
        self.feature_transform = FeatureTransform(in_dim=feat_dim)
        self.hash_encoder = HashEncoder(self.feature_transform.ddim, num_hash_layer, len_hash_code)

        self.__num_hash_layer = num_hash_layer
        self.__len_hash_code = len_hash_code

    @property
    def num_hash_layer(self):
        return self.__num_hash_layer

    @property
    def len_hash_code(self):
        return self.__len_hash_code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, H, W, C]

        x = self.feature_transform(x)  # [batch, ddim]
        hash_codes: torch.Tensor = self.hash_encoder(x)  # [batch, num_hash_layer, len_hash_code]

        if self.training:
            long_hash_code = torch.flatten(hash_codes, 1)  # [batch, num_hash_layer * len_hash_code]
            return long_hash_code

        return hash_codes
