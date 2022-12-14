from espnet2.layers.ds2 import SequenceWise, BatchRNN
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import math
from collections import OrderedDict
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import numpy as np
supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}




class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths




class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'n_features=' + str(self.n_features) \
            + ', context=' + str(self.context) + ')'


class DS2Encoder(AbsEncoder):
    def __init__(self, input_size, rnn_type="lstm", rnn_hidden_size=768, nb_layers=5, bidirectional=True, context=20,
            pretrained_model=""):
        super(DS2Encoder, self).__init__()
        stat_dict = OrderedDict()
        if pretrained_model:
            old_sd = torch.load(pretrained_model)
            for k in old_sd.keys():
                if k.startswith("encoder."):
                    stat_dict[k[8:]] = old_sd[k]

        # model metadata needed for serialization/deserialization
        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.rnn_type = supported_rnns[rnn_type]
        self.bidirectional = bidirectional

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1

        rnns = []
        rnn_input_size = input_size
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=self.rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=self.rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def forward(self, x, lengths, rnn_split=2):
        x = x.unsqueeze(1).transpose(2, 3)
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size() # B C D T
        # Collapse feature dimension
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) # B D T
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # T B D, TxNxH
        rnn_count = 0
        # T B D
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
            # T B D
            if (rnn_count == rnn_split):
                y = x
                split_verify = rnn_count
            rnn_count = rnn_count + 1
        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x, output_lengths, y

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (
                    (seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    def output_size(self) -> int:
        return 1024

    def reload_pretrained_parameters(self):
        # TODO: clear how this method is called
        self.encoders.load_state_dict(self.stat_dict)
        logging.info("Pretrained DS2 encoder model parameters reloaded!")

