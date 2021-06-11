from abc import ABC
import torch
import numpy
from model_fk import FK
from model_tk import TK
from data_loading import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader
from re_ranking import evaluateModel
from core_metrics import calculate_metrics_plain, load_qrels, unrolled_to_ranked_result
from secondary_output_logger import SecondaryBatchOutput, SecondaryBatchOutputFileLogger, SecondaryBatchOutputFileLoggerConfig, SecondaryBatchOutputLogger, TKModelData, FKModelData
from pathlib import Path


def main():
    config = {
        "vocab_directory": "data/allen_vocab_lower_10",
        "pre_trained_embedding": "data/glove.42B.300d.txt",
        "model": "fk",
        "train_data": "data/triples.train.tsv",
        "validation_data": "data/msmarco_tuples.validation.tsv",
        "test_data": "data/msmarco_tuples.test.tsv",
        "qrels_data": "data/msmarco_qrels.txt",
        "onGPU": torch.cuda.is_available(),
        "eval_batch_size": 128,
        "validation_interval": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.000000000000001,
    }

    onGPU = config["onGPU"]
    vocab = Vocabulary.from_files(config["vocab_directory"])
    tokens_embedder = Embedding(vocab=vocab,
                                pretrained_file=config["pre_trained_embedding"],
                                embedding_dim=300,
                                trainable=True,
                                padding_index=0)
    word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

    logger_config: SecondaryBatchOutputFileLoggerConfig = SecondaryBatchOutputFileLoggerConfig(logger_name=f"{config['model']}_logger", root_path=Path('logs/'))

    with SecondaryBatchOutputFileLogger(config=logger_config) as secondary_output_logger:

        if config["model"] == "tk":
            # "learning_rate": 0.0001 needed for tk to perform
            model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10, tf_projection_dim=30, secondary_batch_output_logger=secondary_output_logger)
            secondary_output_logger.model_data = TKModelData(dense_weight=None, dense_mean_weight=None, dense_comb_weight=None)
        elif config["model"] == "fk":
            # learning_rate" : 0.001 needed for fk to perform
            model = FK(word_embedder, n_kernels=11, n_layers=2, n_fnet_dim=300, secondary_batch_output_logger=secondary_output_logger)
            secondary_output_logger.model_data = FKModelData()
        else:
            raise ValueError("no known model configured!")

        if not onGPU:
            model.load_state_dict(torch.load(f"outdir/model_{config['model']}_best.pt", map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f"outdir/model_{config['model']}_best.pt"))
            model.moveModelToGPU()

        qrels = load_qrels(config["qrels_data"])

        _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
        _tuple_reader = _tuple_reader.read(config["test_data"])
        _tuple_reader.index_with(vocab)
        testdata_loader = PyTorchDataLoader(_tuple_reader, batch_size=config["eval_batch_size"])

        model.train(mode=False)

        result = evaluateModel(model, testdata_loader, relevanceLabels=qrels, onGPU=config["onGPU"])


if __name__ == "__main__":
    main()
