from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention              

class WordEmbeddingLayer(nn.Module):
    def __init__(self, word_embeddings: TextFieldEmbedder):
        super(WordEmbeddingLayer, self)

        self.word_embeddings = word_embeddings        

    def forward(self, query_input: Dict[str, torch.Tensor], document_intput: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]
        query_embeddings_tensor = query_embeddings.transpose(1, 2)
        document_embeddings_tensor = document_embeddings.transpose(1, 2)

        return torch.stack([query_embeddings_tensor, document_embeddings_tensor])

class ConvolutionalLayer(nn.Module):
    def __init__(self, n_grams: int, conv_out_dim: int):
        super(ConvolutionalLayer, self)

        self.convolutions = nn.ModuleList()

        #adds convolution for each ngram
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0, i - 1), 0), #adds padding to keep same dimension
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim), #kernel size gets changed for each convolution, sets input channels to dimensions of word embeddings, sets outputput to desired output dimensions
                    nn.ReLU()
                )
            )

    def forward(self, query_document_tensor: torch.Tensor) -> torch.Tensor:
        query_results = []
        document_results = []

        query_tensor, document_tensor = torch.unbind(query_document_tensor)

        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_tensor).transpose(1, 2) 
            document_conv = conv(document_tensor).transpose(1, 2)

            query_results.append(query_conv)
            document_results.append(document_conv)

        query_n_gram_tensor = torch.stack(query_results)
        document_n_gram_tensor = torch.stack(document_results)

        return torch.stack([query_n_gram_tensor, document_n_gram_tensor])

class CrossmatchLayer(nn.Module):
    def __init__(self):
        super(CrossmatchLayer, self)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

    def forward(self, query_document_n_gram_tensor: torch.Tensor, query_by_doc_mask: torch.Tensor) -> torch.Tensor:
        matched_results_all_batches = []

        query_tensor, document_tensor = torch.unbind(query_document_n_gram_tensor)

        for i in range(len(query_tensor)):
            matched_results_per_batch = []

            for t in range(len(query_tensor)):
                cosine_matrix: torch.Tensor = self.cosine_module.forward(query_tensor, document_tensor)
                cosine_matrix_masked: torch.Tensor = cosine_matrix * query_by_doc_mask
                cosine_matrix_extradim: torch.Tensor = cosine_matrix_masked.unsqueeze(-1)

                matched_results_per_batch.append(cosine_matrix_extradim)

            matched_results_all_batches.append(torch.stack(matched_results_per_batch))

        return torch.stack(matched_results_all_batches)

class KernelPoolingLayer(nn.Module):
    def __init__(self, n_kernels: int):
        super(KernelPoolingLayer, self)

        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

    def forward(self, match_matrices_all_batches: torch.Tensor, query_by_doc_mask: torch.Tensor, query_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        soft_tf_features_all_batches = []

        for i, match_matrices_per_batch in enumerate(match_matrices_all_batches):
            soft_tf_features = []

            for j, match_matrix in enumerate(match_matrices_per_batch):
                raw_kernel_results = torch.exp(- torch.pow(match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
                kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

                per_kernel_query = torch.sum(kernel_results_masked, 2)
                log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01 #clamp defines an extremely low value as lower bound
                log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

                per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

                soft_tf_features.append(per_kernel)

            soft_tf_features_all_batches.append(tensor.stack(soft_tf_features))

        return torch.stack(soft_tf_features_all_batches)

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
    def __init__(self):
        super(LearningToRankLayer, self)

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=False) 

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(self, soft_tf_features_all_batches: torch.Tensor) -> torch.Tensor:
        all_grams = torch.cat(soft_tf_features_all_batches, 1)

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
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Conv_KNRM, self).__init__()

        # todo

        #define layers
        self.word_embedding_layer = WordEmbeddingLayer(word_embeddings) 
        self.convolutional_layer = ConvolutionalLayer(n_grams, conv_out_dim) 
        self.crossmatch_layer = CrossmatchLayer() 
        self.kernel_pooling_layer = KernelPoolingLayer()
        self.learning_to_rank_layer = LearningToRankLayer() 

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        # we assume 0 is padding - both need to be removed
        # shape: (batch, query_max)
        query_pad_mask: torch.Tensor = (query["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_mask: torch.Tensor = (document["tokens"] > 0).float()

        #todo
        query_by_doc_mask: torch.Tensor = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        query_pad_oov_mask = (query["tokens"] > 1).float()

        query_document_embedding_tensor: torch.Tensor = self.word_embedding_layer.forward(query, document) #this layer creates the word embeddings
        query_document_n_gram_tensor: torch.Tensor = self.convolutional_layer.forward(query_document_embedding_tensor) #this layer uses convolutions to compose n-gram embeddings
        match_matrices_all_batches: torch.Tensor = self.crossmatch_layer.forward(query_document_n_gram_tensor, query_by_doc_mask) #this layer matches query n-grams and document n-grams of different resolutions
        soft_tf_features_all_batches: torch.Tensor = self.kernel_pooling_layer.forward(match_matrices_all_batches, query_by_doc_mask, query_pad_oov_mask)
        scores_all_batches: torch.Tensor = self.learning_to_rank_layer.forward(soft_tf_features_all_batches) #combines soft-TF ranking into ranking score

        return scores_all_batches
