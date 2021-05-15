from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp_models.rc.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.nn.util import add_positional_features
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.modules.layer_norm import LayerNorm
from torch.types import Device


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers: int,
                 n_tf_dim: int,
                 n_tf_heads: int):

        super(TK, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        self.cosinematrix = CosineMatrixAttention()

        # Contextualization
        # n_layers of transformers stored in a List
        self.ffLayers: List[FeedForward] = []
        self.attentionHeadLayers: List[MultiHeadSelfAttention] = []
        self.normLayers : List[LayerNorm] = []
        for i in range(n_layers):
            self.ffLayers.append(
                FeedForward(
                    n_tf_dim,
                    activations=[Activation.by_name("relu")(), Activation.by_name("linear")()],
                    hidden_dims=[n_tf_dim, n_tf_dim],
                    num_layers=2
                )
            )

            self.attentionHeadLayers.append(
                MultiHeadSelfAttention(
                    num_heads=n_tf_heads,
                    input_dim=n_tf_dim,
                    attention_dim=30,
                    values_dim=30
                ))
            self.normLayers.append(LayerNorm(n_tf_dim))

        self.linear_Slog = nn.Linear(n_kernels, 1, bias=False)
        self.linear_Slen = nn.Linear(n_kernels, 1, bias=False)

        # should be value between 0 and 1
        self.alpha = nn.parameter.Parameter(torch.tensor(0.5))  # alpha --> to control amount contextualization
        self.beta = nn.parameter.Parameter(torch.tensor(0.5))  # beta --> to control amount of s_log on the score
        self.gamma = nn.parameter.Parameter(torch.tensor(0.5))  # gamma --> to control amount of s_len on the score

        # todo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask_bool = query["tokens"] > 0
        query_pad_oov_mask = query_pad_oov_mask_bool.float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        # doc_max --> maximal document length of the longest document in the batch --> each other document is padded with zeros
        document_pad_oov_mask_bool = document["tokens"] > 0
        document_pad_oov_mask = document_pad_oov_mask_bool.float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings({"tokens": query})
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings({"tokens": document})

        # todo

        # contextualization
    
        # query & document is processed separately --> learn parameters are shared

        # 1. positional embedding added --> p
        #query_embeddings_pos: (batch_size, query_len, embedding_dim)
        query_embeddings_pos = add_positional_features(query_embeddings)
        #document_embeddings_pos: (batch_size, document_len, embedding_dim)
        document_embeddings_pos = add_positional_features(document_embeddings)

        # 2 transformer layers

        # transformer(p) = MutliHead(FF(p)) + FF(p)       FF: two-layer fully connected non-linear activation
        query_contextualized = query_embeddings_pos
        document_contextualized = document_embeddings_pos

        # n transformer blocks
        for ff, mutlihead, layer_norm in zip(self.ffLayers, self.attentionHeadLayers, self.normLayers):
            # feedforward layer
            ff_query_contextualized = ff(query_contextualized)
            ff_document_contextualized = ff(document_contextualized)
            # mutlihead attention (provide padding mask)
            attenion_query_contextualized = mutlihead(ff_query_contextualized, query_pad_oov_mask_bool)
            attenion_document_contextualized = mutlihead(ff_document_contextualized, document_pad_oov_mask_bool)
            query_contextualized = layer_norm(attenion_query_contextualized + ff_query_contextualized)
            document_contextualized = layer_norm(attenion_document_contextualized + ff_document_contextualized)

        # ^t_i =t_i * alpha + context(t1:n)_i * (1- alpha) --> alpha controls the influence of contextualization --> is also learned
        query_embedded_contextualized =(self.alpha * query_embeddings) + ( 1- self.alpha) * query_contextualized
        document_embedded_contextualized = (self.alpha * document_embeddings_pos) + ( 1- self.alpha) * document_contextualized

        # the contextualizations adds values to the words that are only padded (therefore we need to remove the values again by multipyling with the paddding masks)
        query_embedded_contextualized = query_embedded_contextualized * query_pad_oov_mask.unsqueeze(-1)
        document_embedded_contextualized = document_embedded_contextualized * document_pad_oov_mask.unsqueeze(-1)
        #since we kept the shape intact through out the transformer part have the following shape for the contextulized embeddings
        #query_embedded_contextualized: (batch_size, query_len, embedding_dim)
        #document_embedded_contextualized: (batch_size, document_len, embedding_dim)

        # intercation scoring

        # 1. query sequence and document sequence match in a single match-matrix
        # M_ij = cosine_similarity(q_i,d_j)
        # query: (batch_size, query_len, embedding_dim) = (batch_size, 14, 300) #for an example input
        # document: (batch_size, doc_len, embedding_dim) = (batch_size, 180, 300)
        # cosine matrix m: (batch_size, query_len, doc_len, 1) = (batch_size, 14, 180, 1)
        cosine_matrix_m = self.cosinematrix.forward(query_embedded_contextualized, document_embedded_contextualized)

        #since we now work on values per query term and document term we also need a mask that indicate this padding values in this interaction matrix
        #query_by_doc_mask: (batch_size, query_len, document_len)
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))

        cosine_matrix_m = cosine_matrix_m * query_by_doc_mask

        # todo explain why unsqueez is needed
        cosine_matrix_m = cosine_matrix_m.unsqueeze(-1)
        # 2. each entry in M is transformed with a set of RBF-kernels

        # K^k_ij = exp(-(M_ij _ mu_k)^2 / (2\sigma^2))              ...mu & sigma are from the kenel
        # 3. each kernel results in a kernel matrix K^k
        # kernel_matrises: (batch_size, query_len, doc_len, n_kernels) = (batch_size, 14, 180, 11)
        kernel_matrises = torch.exp(- torch.pow(cosine_matrix_m - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        #masking needed after the kernel 
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        kernel_matrises = kernel_matrises * query_by_doc_mask_view
        
        # 4. document dimension j is summed for each query term and kernel
        # result_summed_document_axis: (batch_size, query_len, n_kernels) = (batch_size, 14, 11)
        # because doc is in the 3rd dimension we use torch.sum(m, 2)
        result_summed_document_axis = torch.sum(kernel_matrises, 2)

        # The kernel models need masking after the kernels -> the padding 0's will become non-zero, because of the kernel (but when summed up again will distort the output)

        # 5a. log normalization
        # log_b is applied on each query term before summing them up resulting in s^k_log
        #log2(0) = -inf we therefore need to clamp zero values to a very low values in order to a result != -inf
        log_result_summed_document_axis = torch.log2(torch.clamp(result_summed_document_axis, min=1e-10)) 
        #log_result_k: (batch_size, n_kernel) = (batch_size, 11)
        # since we sumed over all query terms we retrive now one value per kernel
        log_result_k = torch.sum(log_result_summed_document_axis, 1)
        # 5b. length normalization
        # /document_length is applied on each query term before summing them up resulting in s^k_len
        document_length = document_embeddings.shape[1] #doc_leng 180 in our case
        normed_result_summed_document_axis = result_summed_document_axis / document_length
         #normed_result_k: (batch_size, n_kernel) = (batch_size, 11)
        # since we sumed over all query terms we retrive now one value per kernel
        normed_result_k = torch.sum(normed_result_summed_document_axis, 1)

        # 6 kernel scores (one value per kernel) is weighted and summed up with simple linear layer (w_log, W_len)
        # results in one scalar for log-normalized and length normalized kernels --> s_log & s_len

        # s_log and s_len --> one value per batch --> shape: (batch_size, 1)
        s_log = self.linear_Slog(log_result_k)
        s_len = self.linear_Slen(normed_result_k)

        # 7 final score of the query-document pair as weighted sum of s_log & s_len
        # beta & gamma controll the magnitude of influce of s_log and s_len on the putput
        output = s_log * self.beta + s_len * self.gamma

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

        for ff, mutlihead, layernrom in zip(self.ffLayers, self.attentionHeadLayers, self.normLayers):
            ff.cuda()
            mutlihead.cuda()
            layernrom.cuda()

        return self.cuda()