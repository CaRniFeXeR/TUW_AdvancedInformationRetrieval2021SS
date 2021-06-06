from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention


class WordEmbeddingLayer(nn.Module):
    def __init__(self, word_embeddings: TextFieldEmbedder):
        super(WordEmbeddingLayer, self).__init__()

        self.word_embeddings = word_embeddings

    def forward(self, query_input: Dict[str, torch.Tensor], document_intput: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (batch, query_max,emb_dim)
        query_embeddings_tensor = self.word_embeddings({"tokens": query_input})
        # shape: (batch, document_max,emb_dim)
        document_embeddings_tensor = self.word_embeddings({"tokens": document_intput})

        m = nn.ZeroPad2d((0, 0, 0, document_embeddings_tensor.shape[1] - query_embeddings_tensor.shape[1]))

        query_embeddings_tensor = m(query_embeddings_tensor)

        return query_embeddings_tensor.transpose(-1, -2), document_embeddings_tensor.transpose(-1, -2)


class ConvolutionalLayer(nn.Module):
    def __init__(self, n_grams: int, conv_in_dim: int, conv_out_dim: int):
        super(ConvolutionalLayer, self).__init__()

        self.convolutions = nn.ModuleList()

        # adds convolution for each ngram
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0, i - 1), 0),  # adds padding to keep same dimension
                    # kernel size gets changed for each convolution, sets input channels to dimensions of word embeddings, sets outputput to desired output dimensions
                    nn.Conv1d(kernel_size=i, in_channels=conv_in_dim, out_channels=conv_out_dim),
                    nn.ReLU()
                )
            )

    def forward(self, query_tensor: torch.Tensor, document_tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        query_results = []
        document_results = []

        for i, conv in enumerate(self.convolutions):
            query_conv = conv(query_tensor)
            document_conv = conv(document_tensor)

            query_results.append(query_conv)
            document_results.append(document_conv)

        query_n_gram_tensor = torch.stack(query_results)
        document_n_gram_tensor = torch.stack(document_results)

        return query_n_gram_tensor.transpose(-1,-2), document_n_gram_tensor.transpose(-1,-2)

    # def cuda(self: ConvolutionalLayer, device: Optional[Union[int, device]] = None) -> ConvolutionalLayer:
    #     for conv in self.convolutions:
    #         conv.cuda(device)

    #     return super(ConvolutionalLayer, self).cuda(device)

    def moveModelToGPU(self) -> nn.Module:
        for conv in self.convolutions:
            conv.cuda()

        return self.cuda()


class CrossmatchLayer(nn.Module):
    def __init__(self):
        super(CrossmatchLayer, self).__init__()

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights)
        self.cosine_module = CosineMatrixAttention()

    def forward(self, query_tensor: torch.Tensor, document_tensor: torch.Tensor, query_by_doc_mask: torch.Tensor) -> List[torch.Tensor]:
        match_matrices = []

        for i in range(len(query_tensor)):
            for t in range(len(query_tensor)):
                cosine_matrix: torch.Tensor = self.cosine_module.forward(query_tensor[i], document_tensor[t])
                cosine_matrix_masked: torch.Tensor = cosine_matrix * query_by_doc_mask

                match_matrices.append(cosine_matrix_masked.unsqueeze(-1))

        return match_matrices


class KernelPoolingLayer(nn.Module):
    def __init__(self, n_kernels: int):
        super(KernelPoolingLayer, self).__init__()

        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

    def forward(self, match_matrices_all_batches: List[torch.Tensor], query_by_doc_mask: torch.Tensor, query_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        soft_tf_features = []

        for i, match_matrix in enumerate(match_matrices_all_batches):
            raw_kernel_results = torch.exp(- torch.pow(match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
            kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

            per_kernel_query = torch.sum(kernel_results_masked, 2)
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01  # clamp defines an extremely low value as lower bound
            log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)  # make sure we mask out padding values

            per_kernel = torch.sum(log_per_kernel_query_masked, 1)

            soft_tf_features.append(per_kernel)

        return torch.stack(soft_tf_features)

    # def cuda(self: KernelPoolingLayer, device: Optional[Union[int, device]] = None) -> KernelPoolingLayer:
    #     self.mu = self.mu.cuda(device)
    #     self.sigma = self.sigma.cuda(device)

    #     return super(KernelPoolingLayer, self).cuda(device)

    def moveModelToGPU(self) -> nn.Module:
        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()

        return self.cuda()

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


class LearningToRankLayer(nn.Module):
    def __init__(self, n_kernels: int, n_grams: int):
        super(LearningToRankLayer, self).__init__()

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(self, soft_tf_features_all_batches: torch.Tensor) -> torch.Tensor:
        all_grams = torch.cat(soft_tf_features_all_batches.unbind(), 1)
        dense_out = self.dense(all_grams)
        tanh_out = torch.tanh(dense_out)

        output = torch.squeeze(tanh_out, 1)
        return output


class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_grams: int,
                 n_kernels: int,
                 conv_out_dim: int):

        super(Conv_KNRM, self).__init__()

        # todo

        # define layers
        self.word_embedding_layer = WordEmbeddingLayer(word_embeddings)
        self.convolutional_layer = ConvolutionalLayer(n_grams, word_embeddings.get_output_dim(), conv_out_dim)
        self.crossmatch_layer = CrossmatchLayer()
        self.kernel_pooling_layer = KernelPoolingLayer(n_kernels)
        self.learning_to_rank_layer = LearningToRankLayer(n_kernels, n_grams)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # todo
        m = nn.ZeroPad2d((0, document["tokens"].shape[1] - query["tokens"].shape[1], 0, 0))

        query["tokens"] = m(query["tokens"])

        # we assume 0 is padding - both need to be removed
        # shape: (batch, query_max)
        query_pad_mask: torch.Tensor = (query["tokens"] > 0).float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_mask: torch.Tensor = (document["tokens"] > 0).float()

        query_by_doc_mask: torch.Tensor = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        query_pad_oov_mask = (query["tokens"] > 1).float()

        query_embeddings_tensor, document_embeddings_tensor = self.word_embedding_layer.forward(query, document)  # this layer creates the word embeddings
        query_n_gram_tensor, document_n_gram_tensor = self.convolutional_layer.forward(query_embeddings_tensor, document_embeddings_tensor)  # this layer uses convolutions to compose n-gram embeddings
        # this layer matches query n-grams and document n-grams of different resolutions
        match_matrices_all_batches: List[torch.Tensor] = self.crossmatch_layer.forward(query_n_gram_tensor, document_n_gram_tensor, query_by_doc_mask)
        soft_tf_features_all_batches: torch.Tensor = self.kernel_pooling_layer.forward(match_matrices_all_batches, query_by_doc_mask, query_pad_oov_mask)
        scores_all_batches: torch.Tensor = self.learning_to_rank_layer.forward(soft_tf_features_all_batches)  # combines soft-TF ranking into ranking score

        return scores_all_batches

    def moveModelToGPU(self) -> nn.Module:

        self.convolutional_layer.moveModelToGPU()
        self.kernel_pooling_layer.moveModelToGPU()

        return self.cuda()
