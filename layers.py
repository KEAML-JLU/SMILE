import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def mixup():
    lam_mix = self.dist.sample().to("cuda")
    task_2_shuffle_id = np.arange(self.args.num_classes)
    np.random.shuffle(task_2_shuffle_id)
    task_2_shuffle_id_s = np.array(
        [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
         range(self.args.num_classes)]).flatten()
    task_2_shuffle_id_q = np.array(
        [np.arange(self.args.update_batch_size_eval) + task_2_shuffle_id[idx] * self.args.update_batch_size_eval for
         idx in range(self.args.num_classes)]).flatten()

    x2s = x2s[task_2_shuffle_id_s]
    x2q = x2q[task_2_shuffle_id_q]

    x_mix_s, _ = self.mixup_data(self.learner.net[0](x1s), self.learner.net[0](x2s), lam_mix)

    x_mix_q, _ = self.mixup_data(self.learner.net[0](x1q), self.learner.net[0](x2q), lam_mix)
