from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers:int,
                 n_tf_dim:int,
                 n_tf_heads:int):

        super(TK, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        #todo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo
        
        ##### contextualization
        #^t_i =t_i * alpha + context(t1:n)_i * (1- alpha) --> alpha controls the influence of contextualization --> is also learned

        #query & document is processed separately --> learn parameters are shared

        #1. positional embedding added --> p
        #transformer(p) = MutliHead(FF(p)) + FF(p)       FF: two-layer fully connected non-linear activation
        # 2 transformer layers


        ####### intercation scoring

        #1. query sequence and document sequence match in a single match-matrix
        # M_ij = cosine_similarity(q_i,d_j)

        #2. each entry in M is transformed with a set of RBF-kernels
            # K^k_ij = exp(-(M_ij _ mu_k)^2 / (2\sigma^2))              ...mu & sigma are from the kenrel
        #3. each kernel results in a kernel matrix K^k
        #4. document dimension j is summed for each query term and kernel
        #5a. log normalization
            # log_b is applied on each query term before summing them up resulting in s^k_log
        #5b. length normalization
            # /document_length is applied on each query term before summing them up resulting in s^k_len
        #6 kernel scores (one value per kernel) is weighted and summed up with simple linear layer (w_log, W_len)
            # results in one scalar for log-normalized and length normalized kernels --> s_log & s_len

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
