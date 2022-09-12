from espnet2.asr.decoder.abs_decoder import AbsDecoder
import torch

class IdentityDecoder(AbsDecoder):
    def __init__(self, *arg, **kw):
        super(IdentityDecoder, self).__init__()

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor, ys_in_pad: torch.Tensor, ys_in_lens: torch.Tensor):
        return hs_pad, hlens
