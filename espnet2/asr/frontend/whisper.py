import copy
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
import whisper


class WhisperFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    """

    def __init__(self, fs):
        super().__init__()
        model = whisper.load_model("large", device="cpu")
        self.whisper_encoder = model.encoder
        self._output_size = model.dims.n_audio_state
    
    def output_size(self) -> int:
        return self._output_size
    
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_feats = self.whisper_encoder(input)
    
        return input_feats, torch.tensor([input_feats.shape[1] for i in range(input_feats.shape[0])])

