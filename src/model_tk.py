from typing import Dict, Iterator, List, Optional, Tuple, Union

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
from torch.nn.modules import dropout
from torch.types import Device
from secondary_output_logger import SecondaryBatchOutput, SecondaryBatchOutputLogger


class TransformerBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, projection_dim: int):
        super().__init__()

        self.ff = FeedForward(
            input_dim,
            activations=[Activation.by_name("relu")(), Activation.by_name("linear")()],
            hidden_dims=[hidden_dim, input_dim],
            num_layers=2
        )

        self.ff_layer_norm = LayerNorm(input_dim)

        self.mutlihead_att = MultiHeadSelfAttention(
            num_heads=n_heads,
            input_dim=input_dim,
            attention_dim=projection_dim,
            values_dim=projection_dim,
        )

        self.layer_norm = LayerNorm(self.mutlihead_att.get_output_dim())

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):

        residual = x
        ff_output = self.ff(x)
        ff_output = self.ff_layer_norm(ff_output + residual)

        att_output = self.mutlihead_att(ff_output, mask)
        output = self.layer_norm(att_output + ff_output)

        return output


class ContextualizationLayer(nn.Module):

    def __init__(self,
                 n_tf_dim: int,
                 tf_projection_dim: int,
                 n_layers: int,
                 n_tf_heads: int,
                 hidden_dim: int):
        super(ContextualizationLayer, self).__init__()

        self.transformerBlocks : nn.ModuleList[TransformerBlock] = nn.ModuleList()
        for i in range(n_layers):
            self.transformerBlocks.append(
                TransformerBlock(input_dim= n_tf_dim,
                                 hidden_dim= hidden_dim,
                                 projection_dim= tf_projection_dim,
                                 n_heads = n_tf_heads)
            )

        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.5, dtype=torch.float32, requires_grad=True))

        # should be value between 0 and 1
        # self.alpha = nn.parameter.Parameter(torch.tensor(0.5))  # alpha --> to control amount contextualization

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor, query_mask: torch.BoolTensor, document_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # contextualization

        # query & document is processed separately --> learn parameters are shared

        # 1. positional embedding added --> p
        #query_embeddings_pos: (batch_size, query_len, embedding_dim)
        query_embeddings_pos = add_positional_features(query_embeddings)
        # #document_embeddings_pos: (batch_size, document_len, embedding_dim)
        document_embeddings_pos = add_positional_features(document_embeddings)

        # 2 transformer layers

        # transformer(p) = MutliHead(FF(p)) + FF(p)       FF: two-layer fully connected non-linear activation
        query_contextualized = query_embeddings_pos * query_mask.unsqueeze(-1)
        document_contextualized = document_embeddings_pos * document_mask.unsqueeze(-1)

        # # n transformer blocks
        for transformerBlock in self.transformerBlocks:
            query_contextualized = transformerBlock(query_contextualized, query_mask)
            document_contextualized = transformerBlock(document_contextualized, document_mask)

        # ^t_i =t_i * alpha + context(t1:n)_i * (1- alpha) --> alpha controls the influence of contextualization --> is also learned
        query_embedded_contextualized = (self.mixer * query_embeddings) + (1 - self.mixer) * query_contextualized
        document_embedded_contextualized = (self.mixer * document_embeddings) + (1 - self.mixer) * document_contextualized
        # query_embedded_contextualized = (self.alpha * query_embeddings) + (1 - self.alpha) * query_contextualized
        # document_embedded_contextualized = (self.alpha * document_embeddings_pos) + (1 - self.alpha) * document_contextualized

        # since we kept the shape intact through out the transformer part have the following shape for the contextulized embeddings
        #query_embedded_contextualized: (batch_size, query_len, embedding_dim)
        #document_embedded_contextualized: (batch_size, document_len, embedding_dim)

        return query_embedded_contextualized, document_embedded_contextualized

    def moveModelToGPU(self) -> nn.Module:

        # for transformerBlock in self.transformerBlocks:
        #     transformerBlock = transformerBlock.cuda()
        # self.transformerBlocks = [transformerBlock]
        self.transformerBlocks = self.transformerBlocks.cuda()

        return self.cuda()


class CrossMatchlayer(nn.Module):

    def __init__(self):
        super(CrossMatchlayer, self).__init__()

        self.cosinematrix = CosineMatrixAttention()

    def forward(self, query_embbeddings: torch.Tensor, document_embeddings: torch.Tensor, query_by_doc_mask: torch.Tensor):

        # 1. query sequence and document sequence match in a single match-matrix
        # M_ij = cosine_similarity(q_i,d_j)
        # query: (batch_size, query_len, embedding_dim) = (batch_size, 14, 300) #for an example input
        # document: (batch_size, doc_len, embedding_dim) = (batch_size, 180, 300)
        # cosine matrix m: (batch_size, query_len, doc_len, 1) = (batch_size, 14, 180, 1)
        cosine_matrix_m = self.cosinematrix.forward(query_embbeddings, document_embeddings)

        cosine_matrix_m = torch.tanh(cosine_matrix_m * query_by_doc_mask)
        # cosine_matrix_m = cosine_matrix_m * query_by_doc_mask

        # unsqueeze to ad addtional dim
        cosine_matrix_m = cosine_matrix_m.unsqueeze(-1)

        return cosine_matrix_m


class KernelPoolingLayer(nn.Module):

    def __init__(self, n_kernels: int):
        super(KernelPoolingLayer, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

    def forward(self, cross_match_matrix: torch.tensor, query_mask: torch.tensor, document_mask: torch.tensor, query_by_doc_mask: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:

        # 2. each entry in M is transformed with a set of RBF-kernels

        # K^k_ij = exp(-(M_ij _ mu_k)^2 / (2\sigma^2))              ...mu & sigma are from the kenel
        # 3. each kernel results in a kernel matrix K^k
        # kernel_matrises: (batch_size, query_len, doc_len, n_kernels) = (batch_size, 14, 180, 11)
        kernel_matrises = torch.exp(- torch.pow(cross_match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        # masking needed after the kernel
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        kernel_matrises = kernel_matrises * query_by_doc_mask_view

        # 4. document dimension j is summed for each query term and kernel
        # result_summed_document_axis: (batch_size, query_len, n_kernels) = (batch_size, 14, 11)
        # because doc is in the 3rd dimension we use torch.sum(m, 2)
        result_summed_document_axis = torch.sum(kernel_matrises, 2)

        # The kernel models need masking after the kernels -> the padding 0's will become non-zero, because of the kernel (but when summed up again will distort the output)

        # 5a. log normalization
        # log_b is applied on each query term before summing them up resulting in s^k_log
        # log2(0) = -inf we therefore need to clamp zero values to a very low values in order to a result != -inf
        #log_result_summed_document_axis (batch_size, query_len, n_kernels)
        log_result_summed_document_axis = torch.log2(torch.clamp(result_summed_document_axis, min=1e-10)) * self.nn_scaler
        # since the clamping added non zero values for the padded values we need to remove them again
        log_result_summed_document_axis = log_result_summed_document_axis * query_mask.unsqueeze(-1)

        #log_result_k: (batch_size, n_kernel) = (batch_size, 11)
        # since we sumed over all query terms we retrive now one value per kernel
        log_result_k = torch.sum(log_result_summed_document_axis, 1)
        # 5b. length normalization
        # /document_length is applied on each query term before summing them up resulting in s^k_len
        # sum of the mask gives use the length for each document --> (batch_size)
        document_lengths = torch.sum(document_mask, 1)
        normed_result_summed_document_axis = result_summed_document_axis / (document_lengths.view(-1, 1, 1) + 0.0001) * self.nn_scaler
        normed_result_summed_document_axis = normed_result_summed_document_axis * query_mask.unsqueeze(-1)
        #normed_result_k: (batch_size, n_kernel) = (batch_size, 11)
        # since we sumed over all query terms we retrive now one value per kernel
        normed_result_k = torch.sum(normed_result_summed_document_axis, 1)

        return log_result_k, normed_result_k

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


class LearningToRankLayer(nn.Module):

    def __init__(self, n_kernels: int):
        super(LearningToRankLayer, self).__init__()

        self.linear_Slog = nn.Linear(n_kernels, 1, bias=False)
        self.linear_Slen = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.linear_Slog.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.linear_Slen.weight, -0.014, 0.014)  # inits taken from matchzoo

        self.dense_comb = nn.Linear(2, 1, bias=False)
        # self.beta = nn.parameter.Parameter(torch.tensor(0.5))  # beta --> to control amount of s_log on the score
        # self.gamma = nn.parameter.Parameter(torch.tensor(0.5))  # gamma --> to control amount of s_len on the score

    def forward(self, log_normed: torch.tensor, length_normed: torch.tensor):

        # 6. kernel scores (one value per kernel) is weighted and summed up with simple linear layer (s_log, s_len)
        # results in one scalar for log-normalized and length normalized kernels --> s_log & s_len

        # s_log and s_len --> one value per batch --> shape: (batch_size, 1)
        s_log = self.linear_Slog(log_normed)
        s_len = self.linear_Slen(length_normed)

        # 7. final score of the query-document pair as weighted sum of s_log & s_len
        # beta & gamma controll the magnitude of influce of s_log and s_len on the putput
        # output = s_log * self.beta + s_len * self.gamma
        output = self.dense_comb(torch.cat([s_log, s_len], dim=1))

        return output


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers: int,
                 n_tf_dim: int,
                 n_tf_heads: int,
                 tf_projection_dim: int,
                 secondary_batch_output_logger: SecondaryBatchOutputLogger = None):

        super(TK, self).__init__()

        if not isinstance(word_embeddings, TextFieldEmbedder):
            raise TypeError("word_embeddings must be a TextFieldEmbedder")

        if not isinstance(n_kernels, int):
            raise TypeError("n_kernels must be a int")

        if not isinstance(n_layers, int):
            raise TypeError("n_layers must be a int")

        if not isinstance(n_tf_dim, int):
            raise TypeError("n_tf_dim must be a int")

        if not isinstance(n_tf_heads, int):
            raise TypeError("n_tf_heads must be a int")

        if not isinstance(tf_projection_dim, int):
            raise TypeError("tf_projection_dim must be a int")

        self.n_kernels = n_kernels
        self.n_tf_heads = n_tf_heads
        self.n_tf_dim = n_tf_dim
        self.tf_projection_dim = tf_projection_dim

        self.word_embeddings = word_embeddings
        self.contextualization = ContextualizationLayer(n_tf_dim, tf_projection_dim, n_layers, n_tf_heads, hidden_dim= 100)
        self.crossmatch = CrossMatchlayer()
        self.kernelpooling = KernelPoolingLayer(n_kernels)
        self.learning_to_rank = LearningToRankLayer(n_kernels)
        self.secondary_batch_output_logger: SecondaryBatchOutputLogger = secondary_batch_output_logger

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
        query_embeddings = self.word_embeddings({"tokens": {"tokens": query["tokens"]}})
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings({"tokens": {"tokens": document["tokens"]}})

        # contextualization
        query_contextualized, document_contextualized = self.contextualization(query_embeddings, document_embeddings, query_pad_oov_mask_bool, document_pad_oov_mask_bool)

        # the contextualizations adds values to the words that are only padded (therefore we need to remove the values again by multipyling with the paddding masks)
        query_contextualized = query_contextualized * query_pad_oov_mask.unsqueeze(-1)
        document_contextualized = document_contextualized * document_pad_oov_mask.unsqueeze(-1)
        # since we kept the shape intact through out the transformer part have the following shape for the contextulized embeddings
        #query_contextualized: (batch_size, query_len, embedding_dim)
        #document_contextualized: (batch_size, document_len, embedding_dim)

        # intercation scoring
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        cosine_matrix_m = self.crossmatch(query_contextualized, document_contextualized, query_by_doc_mask)

        # kernel pooling
        s_log, s_len = self.kernelpooling(cosine_matrix_m, query_pad_oov_mask, document_pad_oov_mask, query_by_doc_mask)
        # learning to rank
        output = self.learning_to_rank(s_log, s_len)

        if not self.secondary_batch_output_logger is None:
            self.secondary_batch_output_logger.log(secondary_batch_output=SecondaryBatchOutput(score=output, per_kernel=s_log, query_embeddings=query_embeddings, query_embeddings_oov_mask=query_pad_oov_mask, cosine_matrix=cosine_matrix_m, query_id=query["id"], doc_id=document["id"]))

        return output

    def moveModelToGPU(self) -> nn.Module:

        self.contextualization.moveModelToGPU()
        self.kernelpooling.moveModelToGPU()

        return self.cuda()

    def get_named_parameters(self):
        named_pars = [(pName, par) for pName, par in self.named_parameters()]

        # for transformerBlock in self.transformerBlocks:
        #     transformer_pars = [(pName, par) for pName, par in transformerBlock.named_parameters()]
        #     named_pars = named_pars + transformer_pars

        return named_pars

    def fill_wandb_config(self, config: dict):

        config["n_kernels"] = self.n_kernels
        config["n_tf_heads"] = self.n_tf_heads
        config["n_tf_dim"] = self.n_tf_dim
        config["tf_projection_dim"] = self.tf_projection_dim
