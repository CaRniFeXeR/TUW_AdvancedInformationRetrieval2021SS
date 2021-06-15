from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention


class WordEmbeddingLayer(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    
    The word-embedding layer maps earch word of to an n-dimensional vector.
    The result is matrix with dimension w x n, where n is the dimension of the embedding vector and w is the number of words.
    '''

    def __init__(self, word_embeddings: TextFieldEmbedder):
        super(WordEmbeddingLayer, self).__init__()

        self.word_embeddings = word_embeddings

    def forward(self, query_input: Dict[str, torch.Tensor], document_intput: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : Dict[str, torch.Tensor]
            A dictionary containing the query tokens.
            shape: (batch_size, query_max)
        document : Dict[str, torch.Tensor]
            A dictionary containing the document tokens.
            shape:(batch_size, document_max)

        Returns
        -------
        scores: Tuple[torch.Tesnor, torch.Tensor]
            A tuple containing both the query and document word-embeddings
            query-tensor shape: (batch_size, embedding_dimension, query_max)
            document-tensor shape: (batch_size, embedding_dimension, document_max)
        """

        query_embeddings_tensor = self.word_embeddings({"tokens": query_input})
        document_embeddings_tensor = self.word_embeddings({"tokens": document_intput})
        return query_embeddings_tensor.transpose(1, 2), document_embeddings_tensor.transpose(1, 2)


class ConvolutionalLayer(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    The convolutional layer applies filter to produce the desired n-grams using the query- and document-embeddings.
    The larger the dimension of the filter, the more words are used as context. The convolutional layer therefore
    combines the embedding and context information to a new unit.    
    '''

    def __init__(self, n_grams: int, conv_in_dim: int, conv_out_dim: int):
        super(ConvolutionalLayer, self).__init__()

        self.convolutions = nn.ModuleList()

        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0, i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=conv_in_dim, out_channels=conv_out_dim),
                    nn.ReLU()
                )
            )

    def forward(self, query_tensor: torch.Tensor, document_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Parameters
        ----------
        query_tensor: torch.Tensor
            A tensor containing the query word-embeddings for the whole batch.
            shape: (batch_size, embedding_dimension, query_max)
        document_tensor: torch.Tensor
            A tensor containing the document word-embeddings for the whole batch.
            shape: (batch_size, embedding_dimension, document_max)
        Returns
        -------
        result: Tuple[List[torch.Tensor], List[torch.Tensor]]
            A tuple containing containing the n-grams for both document and query
            query n-gram tensor shape: (batch_size, query_max, output_dimension)
            document n-gram tensor shape: (batch_size, document_max, output_dimension)
        """

        query_results = []
        document_results = []

        for i, conv in enumerate(self.convolutions):
            query_conv = conv(query_tensor)
            document_conv = conv(document_tensor)

            query_results.append(query_conv.transpose(1, 2))
            document_results.append(document_conv.transpose(1, 2))

        return query_results, document_results

    def moveModelToGPU(self) -> nn.Module:
        for conv in self.convolutions:
            conv.cuda()

        return self.cuda()


class CrossmatchLayer(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    
    Matches query and document n-grams in different length variations. The number resulting matrices is max(query_max, document_x)^2 and
    each matrix has a dimension of query_max x document_max.
    '''

    def __init__(self):
        super(CrossmatchLayer, self).__init__()

        self.cosine_module = CosineMatrixAttention()

    def forward(self, query_tensors: List[torch.Tensor], document_tensors: List[torch.Tensor], query_by_doc_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        query_tensors: List[torch.Tensor]
            A list containing the query n-grams.
            shape: (batch_size, query_max, output_dimension)
        document_tensors: List[torch.Tensor]
            A list containing the document n-grams
            shape: (batch_size, document_max, output_dimension)
        Returns
        -------
        match_matrices: List[torch.Tensor]
            A list containing all matchmatrices for the whole batch and n-grams.
            tensor shape: (batch_size, query_max, document_max, 1)
        """

        match_matrices = []

        for i in range(len(query_tensors)):
            for t in range(len(query_tensors)):
                cosine_matrix: torch.Tensor = self.cosine_module.forward(query_tensors[i], document_tensors[t])
                cosine_matrix_masked: torch.Tensor = cosine_matrix * query_by_doc_mask

                match_matrices.append(cosine_matrix_masked.unsqueeze(-1))

        return match_matrices


class KernelPoolingLayer(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    
    Kernel-pooling uses a number of Gaussian kernels to count the matches of n-gram pairs. Depending on the
    number of kernels, it produces k soft-tf features for earch of the match-matrices.
    '''

    def __init__(self, n_kernels: int):
        super(KernelPoolingLayer, self).__init__()

        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

    def forward(self, match_matrices_all_batches: List[torch.Tensor], query_by_doc_mask: torch.Tensor, query_pad_oov_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        match_matrices: List[torch.Tensor]
            A list containing all matchmatrices for the whole batch and n-grams.
            tensor shape: (batch_size, query_max, document_max, 1)
        Returns
        -------
        soft_tf_features: List[torch.Tensor]
            A list containing all soft-tf features for each match-matrix and the whole batch
            tensor shape: (batch_size, n_kernels)
        """

        soft_tf_features = []

        for i, match_matrix in enumerate(match_matrices_all_batches):
            raw_kernel_results = torch.exp(- torch.pow(match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
            kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

            per_kernel_query = torch.sum(kernel_results_masked, 2)
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
            log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)

            per_kernel = torch.sum(log_per_kernel_query_masked, 1)

            soft_tf_features.append(per_kernel)

        return soft_tf_features

    def moveModelToGPU(self) -> nn.Module:
        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()

        return self.cuda()

    def kernel_mus(self, n_kernels: int):
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1) 
        l_mu.append(1 - bin_size / 2)
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.001]
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


class LearningToRankLayer(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    The learning-to-rank layer combines all soft-tf features into one score.
    '''

    def __init__(self, n_kernels: int, n_grams: int):
        super(LearningToRankLayer, self).__init__()

        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.01, 0.01)

    def forward(self, soft_tf_features_all_batches: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        soft_tf_features: List[torch.Tensor]
            A list containing all soft-tf features for each match-matrix and the whole batch
            tensor shape: (batch_size, n_kernels)
        Returns
        -------
        output: torch.Tensor
            A tensor containing the scores for each query document pair in the batch
            shape: (batch_size)
        """

        all_grams = torch.cat(soft_tf_features_all_batches, 1)
        dense_out = self.dense(all_grams)

        output = torch.squeeze(dense_out, 1)
        return output


class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    Conv-KNRM stands for Convolutional Kernel base Neural Ranking Model.
    Instead of matching query document n-grams directly it first generates n-grams of different length and
    soft-matches them in a single embedding space. The soft-matches are further processed by the kernel-pooling
    and learning to rank layer to produce the final score.
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_grams: int,
                 n_kernels: int,
                 conv_out_dim: int):

        super(Conv_KNRM, self).__init__()

        self.word_embedding_layer = WordEmbeddingLayer(word_embeddings)
        self.convolutional_layer = ConvolutionalLayer(n_grams, word_embeddings.get_output_dim(), conv_out_dim)
        self.crossmatch_layer = CrossmatchLayer()
        self.kernel_pooling_layer = KernelPoolingLayer(n_kernels)
        self.learning_to_rank_layer = LearningToRankLayer(n_kernels, n_grams)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        query : Dict[str, torch.Tensor]
            A dictionary containing the query tokens.
            shape: (batch_size, query_max)
        document : Dict[str, torch.Tensor]
            A dictionary containing the document tokens.
            shape:(batch_size, document_max)

        Returns
        -------
        scores: torch.Tensor
            A tensor containing the scores for each query document pair in the batch
            shape: (batch_size)
        """

        query_pad_mask: torch.Tensor = (query["tokens"] > 0).float()
        document_pad_mask: torch.Tensor = (document["tokens"] > 0).float()

        query_by_doc_mask: torch.Tensor = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        query_pad_oov_mask = (query["tokens"] > 1).float()

        query_embeddings_tensor, document_embeddings_tensor = self.word_embedding_layer.forward(query, document)
        query_n_gram_tensor, document_n_gram_tensor = self.convolutional_layer.forward(query_embeddings_tensor, document_embeddings_tensor)
        match_matrices_all_batches: List[torch.Tensor] = self.crossmatch_layer.forward(query_n_gram_tensor, document_n_gram_tensor, query_by_doc_mask)
        soft_tf_features_all_batches: torch.Tensor = self.kernel_pooling_layer.forward(match_matrices_all_batches, query_by_doc_mask, query_pad_oov_mask)
        scores: torch.Tensor = self.learning_to_rank_layer.forward(soft_tf_features_all_batches)

        return scores

    def moveModelToGPU(self) -> nn.Module:

        self.convolutional_layer.moveModelToGPU()
        self.kernel_pooling_layer.moveModelToGPU()

        return self.cuda()
