# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class Conv1DBiLSTMEncoder(nn.Module):
    """A simple 1D temporal convolutional stack followed by a
    bidirectional LSTM encoder.

    Args:
        num_features (int): Input feature dimension for each time step.
        conv_channels (list): List of output channels for successive Conv1d
            layers. Each Conv1d preserves sequence length using padding.
        kernel_size (int): Kernel width for Conv1d layers.
        lstm_hidden (int): Hidden size for the LSTM (per direction).
        lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout applied after LSTM.
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: Sequence[int] = (128, 128),
        kernel_size: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0
        self.num_features = num_features
        self.conv_channels = list(conv_channels)

        layers: list[nn.Module] = []
        in_ch = num_features
        for out_ch in self.conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_ch),
                ]
            )
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=self.conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, F = inputs.shape

        # Prepare for Conv1d: (N, C_in, L=T)
        x = inputs.permute(1, 2, 0)

        # Apply conv stack: (N, C_out, T)
        x = self.conv_net(x)

        # Prepare for LSTM: (T, N, C_out)
        x = x.permute(2, 0, 1)

        # LSTM -> (T, N, hidden*2)
        x, _ = self.lstm(x)

        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        T = x.shape[0]
        return x + self.pe[:T].unsqueeze(1)


class TimeSeriesTransformerEncoder(nn.Module):
    """An encoder-only Transformer for temporal EMG sequences.

    Inputs: (T, N, num_features) -> Outputs: (T, N, d_model)

    Notes:
    - This implementation does NOT downsample the time dimension. If you
      add subsampling, remember to update emission length handling in
      `TDSConvCTCModule._step`.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 10000,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.input_proj(inputs)  # (T, N, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (T, N, d_model)
        x = self.dropout(x)
        return x


class Conv1DTransformerEncoder(nn.Module):
    """A small Conv1D frontend followed by a Transformer encoder.

    Inputs: (T, N, num_features) -> Outputs: (T, N, d_model)
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: Sequence[int] = (128, 128),
        kernel_size: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 10000,
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0
        layers: list[nn.Module] = []
        in_ch = num_features
        for out_ch in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_ch),
                ]
            )
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)

        # Reuse the TimeSeriesTransformerEncoder for the transformer stack
        self.transformer = TimeSeriesTransformerEncoder(
            num_features=in_ch,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, F = inputs.shape

        # (T, N, F) -> (N, F, T) for Conv1d
        x = inputs.permute(1, 2, 0)
        x = self.conv_net(x)  # (N, C_out, T)

        # (N, C_out, T) -> (T, N, C_out) for transformer
        x = x.permute(2, 0, 1)

        # Transformer expects (T, N, C)
        x = self.transformer(x)  # (T, N, d_model)
        return x


class Conv1DGRUEncoder(nn.Module):
    """Conv1D frontend followed by a deep GRU encoder.

    Inputs: (T, N, num_features) -> Outputs: (T, N, out_dim)

    Args:
        num_features: input feature dim per timestep
        conv_channels: channels for stacked Conv1d layers
        kernel_size: conv kernel size (padding preserves length)
        gru_hidden: hidden size for GRU (per direction)
        gru_layers: number of GRU layers
        bidirectional: whether GRU is bidirectional
        dropout: dropout after GRU
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: Sequence[int] = (128, 128),
        kernel_size: int = 3,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0
        layers: list[nn.Module] = []
        in_ch = num_features
        for out_ch in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_ch),
                ]
            )
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_dim = gru_hidden * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, F = inputs.shape

        # (T, N, F) -> (N, F, T) for Conv1d
        x = inputs.permute(1, 2, 0)
        x = self.conv_net(x)  # (N, C_out, T)

        # (N, C_out, T) -> (T, N, C_out) for GRU
        x = x.permute(2, 0, 1)

        # GRU -> (T, N, out_dim)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x


class Conv1DGRUStackedEncoder(nn.Module):
    """Two stacked Conv1DGRUEncoders for increased modeling capacity.

    Inputs: (T, N, num_features) -> Outputs: (T, N, out_dim)

    Args:
        num_features: input feature dim per timestep
        mlp_features: list of hidden dims for the MLP before the Conv1DGRUEncoders
        block_channels: channels for stacked Conv1d layers in Conv1DGRUEncoders
        kernel_width: conv kernel size (padding preserves length)
        gru_hidden: hidden size for GRU (per direction)
        stack_GRU: number of stacked GRU
        gru_layers: number of GRU layers
        bidirectional: whether GRU is bidirectional
        dropout: dropout after GRU
    """
    
    def __init__(
        self,
        num_features: int,
        mlp_features: Sequence[int],
        conv_channels: Sequence[int] = (128, 128),
        conv_kernel: int = 3,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        numGRUStack: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0
        self.num_features = num_features
        self.conv_channels = list(conv_channels)

        layers: list[nn.Module] = []
        in_ch = num_features
        for out_ch in self.conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_ch),
                ]
            )
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)

        # Build two stacked Conv1D + GRU encoders. The second encoder's
        # input dimension equals the first encoder's output dimension.
        for i in range(numGRUStack):
            # dynamically create the number of GRU objects and name them 
            setattr(
                self,
                f"encoder_{i+1}",
                Conv1DGRUEncoder(
                    num_features=in_ch if i == 0 else gru_hidden * (2 if bidirectional else 1),
                    conv_channels=conv_channels,
                    kernel_size=conv_kernel,
                    gru_hidden=gru_hidden,
                    gru_layers=gru_layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, F = inputs.shape

        # (T, N, F) -> (N, F, T) for Conv1d
        x = inputs.permute(1, 2, 0)
        x = self.conv_net(x)  # (N, C_out, T)

        # (N, C_out, T) -> (T, N, C_out) for GRU
        x = x.permute(2, 0, 1)

        # Pass through stacked Conv1DGRUEncoders
        numGRUStack = len([m for m in self.children() if isinstance(m, Conv1DGRUEncoder)])
        for i in range(numGRUStack):
            encoder = getattr(self, f"encoder_{i+1}")
            x = encoder(x)  # (T, N, out_dim)

        return x
        