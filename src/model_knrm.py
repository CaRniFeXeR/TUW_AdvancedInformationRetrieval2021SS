from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from allennlp.modules.matrix_attention import CosineMatrixAttention
from torch.autograd import Variable
from allennlp.nn import util
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings
        self.cross_match = CosineMatrixAttention()
        self.ranking_layer = nn.Linear(n_kernels, 1, bias=True)

        torch.nn.init.xavier_uniform(self.ranking_layer.weight)

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1,
                                                                                                          n_kernels)


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings({"tokens": query})
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings({"tokens": document})

        # calculate match matrix with allen-nlp layer
        match_matrix = self.cross_match.forward(query_embeddings, document_embeddings)

        # calculate all kernels per batch
        masked_kernels_per_mm = torch.exp(
            -(torch.pow(match_matrix.unsqueeze(-1) - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))).transpose(0,-1) * document_pad_oov_mask.transpose(0, 1)

        # calculate all k_k(M_i) vectors
        kernels_per_term = torch.sum(masked_kernels_per_mm, 2)

        # use small epsilon to avoid log(0) but then mask away padding fields
        masked_log_kernels_per_term = (torch.log(torch.clamp(kernels_per_term, min=1e-10)) * query_pad_oov_mask.transpose(0,1))

        # calculate phi(M)
        pooling_result = torch.sum(masked_log_kernels_per_term, 1).transpose(0, 1)

        # compute rank with tanh and dense linear
        output = torch.tanh(self.ranking_layer(pooling_result))

        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma

    def moveModelToGPU(self) -> nn.Module:

        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()
        return self.cuda()