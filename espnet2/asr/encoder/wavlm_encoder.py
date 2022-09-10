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
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class TransFormersWavLMEncoder(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        wavlm_model_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        feature_selection: str = "hidden_states",
        layer_selection: int = None,
        freeze_finetune_updates: int = 0,
    ):
        assert check_argument_types()
        super().__init__()

        if wavlm_model_path != "":
            try:
                import transformers 
                from transformers import WavLMModel
            except Exception as e:
                print("Error: Transformers is not properly installed.")
                print(
                    "Please install Transformers: cd ${MAIN_ROOT}/tools && make transformers.done"
                )
                raise e

        self.wavlm_model_path = wavlm_model_path

        self._output_size = output_size

        model = WavLMModel.from_pretrained(self.wavlm_model_path)

        if not isinstance(model, WavLMModel):
            print(
                "Error: pretrained models should be within: "
                "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
            )
            raise Exception("Error: pretrained models should be within: \n'Wav2Vec2Model' classes, etc.")

        self.encoders = model

        self.pretrained_params = copy.deepcopy(model.state_dict())
        self.layer_num = model.config.num_hidden_layers + 1
        self.weights = nn.Parameter(torch.zeros(self.layer_num)) 

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if model.config.output_hidden_size != output_size:
            model.config.output_hidden_size = output_size
            model.config.add_adapter = True
            model.adapter = WavLMAdapter(model.config)
            
            # TODO: may not need to call model.post_init() again
            model.post_init()

        self.freeze_finetune_updates = freeze_finetune_updates
        # Extra added
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        config = self.encoders.config

        for layer_id in range(config.num_feat_extract_layers):
            input_lengths = _conv_out_length(
                input_lengths, config.conv_kernel[layer_id], config.conv_stride[layer_id]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wavlm parameters!")

        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                attention_mask=masks,
                output_hidden_states=True
            )

        feature = self._select_feature(enc_outputs)
        if isinstance(feature, (list, tuple)):
            feature = self._weighted_sum(feature)
        xs_pad = feature
        # xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]

        if enc_outputs.get("padding_mask", None) is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            if mask.any():
                input_lengths = (1 - masks.long()).sum(-1)
                # apply conv formula to get real output_lengths
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)
                padding_mask = torch.zeros(
                    feature.shape[:2], dtype=feature.dtype, device=feature.device
                )

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

                masks = padding_mask
                olens = (~masks).sum(dim=1)  # (B)
            else:
                olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        # TODO: clear how this method is called
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained WavLM model parameters reloaded!")

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

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
