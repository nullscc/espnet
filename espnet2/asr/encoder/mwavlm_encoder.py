# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import logging
import os
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.wavlm.WavLM import WavLMConfig, WavLM
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class MWavLMEncoder(AbsEncoder):
    """WavLM encoder module.

    """

    def __init__(
        self,
        input_size: int,
        wavlm_model_file: str = "",
        output_size: int = 256,
        layer_selection: int = None,
        freeze_finetune_updates: int = 0,
    ):
        assert check_argument_types()
        assert wavlm_model_file
        super().__init__()

        self._output_size = output_size

        checkpoint = torch.load(wavlm_model_file)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)

        assert isinstance(model, WavLM)

        self.encoders = model
        self.layer_num = model.cfg.encoder_layers + 1
        self.weights = nn.Parameter(torch.zeros(self.layer_num)) 

        self.pretrained_params = copy.deepcopy(checkpoint['model'])
        self.layer_selection = layer_selection

        if model.cfg.encoder_embed_dim != output_size:
            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

        def generate_hook_handler(hiddens, f):
            def hook_handler(self, input, output):
                hiddens.append(f(input, output))

            return hook_handler

        self._hook_hiddens = []
        layers = self.encoders.encoder.layers
        for module_id in range(len(layers)):
            m = layers[module_id]
            m.register_forward_hook(generate_hook_handler(self._hook_hiddens, lambda input, output: input[0].transpose(0, 1)))
        self.encoders.encoder.register_forward_hook(generate_hook_handler(self._hook_hiddens, lambda input, output: output[0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward WavLM Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)
        new_xs_pad = torch.zeros(xs_pad.shape).to(xs_pad.device)
        for i, wav in enumerate(xs_pad):
            valid_index = ~(masks[i])
            tmp = xs_pad[i][valid_index]
            new_xs_pad[i][valid_index] = F.layer_norm(tmp, tmp.shape)
        xs_pad = new_xs_pad
            
        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wavlm parameters!")

        self._hook_hiddens.clear()
        with torch.no_grad() if not ft else contextlib.nullcontext():
            res = self.encoders(
                xs_pad,
                masks,
            )

        # xs_pad = [x.transpose(0, 1) for x, _ in res[0][1]] # (B T C)
        xs_pad = self._hook_hiddens
        masks = res[1]  # (B, T)
        feature = self._select_feature(xs_pad)
        if isinstance(feature, (list, tuple)):
            feature = self._weighted_sum(feature)
       
        xs_pad = feature.new_zeros(feature.shape)
        xs_pad[~masks] = feature[~masks]
        del feature
        
        olens = (~masks).sum(dim=1)  # (B)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained WavLM model parameters reloaded!")

    def _select_feature(self, features):
        # feature = features.get(self.feature_selection)
        feature = features

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]
        
        if isinstance(feature, (list, tuple)) and isinstance(self.layer_selection, int):
            feature = feature[self.layer_selection]

        return feature

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), ("self.layer_num != len(feature):{} {}".format(self.layer_num, len(feature)))
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
