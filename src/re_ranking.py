from re import purge
import re
from typing import Tuple
from numpy import ndarray
from torch.functional import Tensor
from model_tk import *
from model_conv_knrm import *
from model_knrm import *
from data_loading import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.vocabulary import Vocabulary
import torch
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
from core_metrics import calculate_metrics_plain, load_qrels, unrolled_to_ranked_result
prepare_environment(Params({}))  # sets the seeds to be fixed


def modelForwardPassOnTripleBatchData(model: nn.Module, batch: dict, targetValues: Tensor, onGPU: bool):
    # batch["query_tokens"]["tokens"]["tokens"] --> query_tokens
    # batch["doc_pos_tokens"]["tokens"]["tokens"] --> tokens of relevent documents
    # batch["doc_neg_tokens"]["tokens"]["tokens"] --> tokens of non-relevant documents

    # include GPU support
    if onGPU:
        batch["query_tokens"]["tokens"]["tokens"] = batch["query_tokens"]["tokens"]["tokens"].to(device="cuda")
        batch["doc_pos_tokens"]["tokens"]["tokens"] = batch["doc_pos_tokens"]["tokens"]["tokens"].to(device="cuda")
        batch["doc_neg_tokens"]["tokens"]["tokens"] = batch["doc_neg_tokens"]["tokens"]["tokens"].to(device="cuda")

    # first forward pass query + relevant doc
    output_score_relevant = model(batch["query_tokens"]["tokens"], batch["doc_pos_tokens"]["tokens"])
    # second forward pass query + non-relevant doc
    output_score_unrelevant = model(batch["query_tokens"]["tokens"], batch["doc_neg_tokens"]["tokens"])

    current_batch_size = batch["query_tokens"]["tokens"]["tokens"].shape[0]
    if current_batch_size != targetValues.shape[0]:
        targetValues = torch.ones(current_batch_size)
        if onGPU:
            targetValues = targetValues.cuda()

    return output_score_relevant, output_score_unrelevant, targetValues


def modelForwardPassOnTupleBatchData(model: nn.Module, batch: dict, onGPU: bool):
    # batch["query_tokens"] --> query tokens
    # batch["doc_tokens"] --> document tokens

    if onGPU:
        batch["query_tokens"]["tokens"]["tokens"] = batch["query_tokens"]["tokens"]["tokens"].to(device="cuda")
        batch["doc_tokens"]["tokens"]["tokens"] = batch["doc_tokens"]["tokens"]["tokens"].to(device="cuda")

    return model(batch["query_tokens"]["tokens"], batch["doc_tokens"]["tokens"])


def evaluateModelOnBatch(model: nn.Module, batch: dict, resultDict: Dict[int, List[Tuple[int, float]]], onGPU: bool) -> dict:
    # batch["query_tokens"] --> query tokens
    # batch["doc_tokens"] --> document tokens
    # batch["query_ids"] --> query ids
    # batch["doc_ids"] --> document ids

    output = modelForwardPassOnTupleBatchData(model, batch, onGPU)

    for idx, query_id in enumerate(batch["query_id"]):
        if not query_id in resultDict:
            resultDict[query_id] = []
        resultDict[query_id].append((batch["doc_id"][idx], float(output[idx].cpu())))

    return resultDict


def evaluateModel(model: nn.Module, tupleLoader: PyTorchDataLoader, relevanceLabels: dict, onGPU: bool):

    resultDict: Dict[int, List[Tuple[int, float]]] = {}

    for batch in Tqdm.tqdm(tupleLoader):
        resultDict = evaluateModelOnBatch(model, batch, resultDict, onGPU)

    # reorder documents by query according to result score
    ranked_result = unrolled_to_ranked_result(resultDict)

    # calculate ir metrics
    return calculate_metrics_plain(ranked_result, relevanceLabels)


# change paths to your data directory
config = {
    "vocab_directory": "data/allen_vocab_lower_10",
    "pre_trained_embedding": "data/glove.42B.300d.txt",
    "model": "tk",
    "train_data": "data/triples.train.tsv",
    "validation_data": "data/msmarco_tuples.validation.tsv",
    "test_data": "data/msmarco_queries.test.tsv",
    "qrels_data": "data/msmarco_qrels.txt",
    "onGPU": False,
    "traning_batch_size": 64,
    "eval_batch_size" : 256,
}

config["onGPU"] = torch.cuda.is_available()

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding(vocab=vocab,
                            pretrained_file=config["pre_trained_embedding"],
                            embedding_dim=300,
                            trainable=True,
                            padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10)


# todo optimizer, loss

print('Model', config["model"], 'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#

_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
loader = PyTorchDataLoader(_triple_reader, batch_size=config["traning_batch_size"])


onGPU = config["onGPU"]

if onGPU:
    if hasattr(model, "moveModelToGPU"):
        model = model.moveModelToGPU()
    else:
        model = model.cuda()

# loss = max(0, s_nonrel - s_rel + 1) .... called marginrankingloss
marginRankingLoss = torch.nn.MarginRankingLoss(margin=1, reduction='mean')  # .cuda(cuda_device)
# since we always want a "big" distance between unrelevant and releveant documents we can use Ones for each pair as targetValue
targetValues = torch.ones(config["traning_batch_size"])
if onGPU:
    targetValues = targetValues.cuda()

# load labels
qrels = load_qrels(config["qrels_data"])

# todo set learningrate and weight decay
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2):

    # activate training mode on model
    model.train(mode=True)
    trainLossList = []

    for batch in Tqdm.tqdm(loader):
        output_score_relevant, output_score_unrelevant, targetValues = modelForwardPassOnTripleBatchData(model, batch, targetValues, onGPU)

        batch_loss = marginRankingLoss(output_score_relevant, output_score_unrelevant, targetValues)
        batch_loss.backward()
        optimizer.step()
        if onGPU:
            current_loss = batch_loss.cpu().detach().numpy()
        else:
            current_loss = batch_loss.detach().numpy()
        trainLossList.append(current_loss)
        print(f"                                                                    current loss: {current_loss:.3f}")

    meanLoss = np.mean(trainLossList)
    stdLoss = np.std(trainLossList)
    print(f'epoch {epoch + 1}\n train loss: {meanLoss:.3f} ± {stdLoss:.3f}')

    # ValidationSet
    model.train(mode=False)
    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
    _tuple_reader = _tuple_reader.read(config["validation_data"])
    _tuple_reader.index_with(vocab)
    validation_loader = PyTorchDataLoader(_tuple_reader, batch_size=config["eval_batch_size"])

    result = evaluateModel(model, validation_loader, relevanceLabels=qrels, onGPU=onGPU)
    print(f"MRR@10 : {result['MRR@10']:.3f}")

    # set model in eval mode


#
# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
testdata_loader = PyTorchDataLoader(_tuple_reader, batch_size=config["eval_batch_size"])

model.train(mode=False)

result = evaluateModel(model, testdata_loader, relevanceLabels=qrels, onGPU=onGPU)
print(f"MRR@10 : {result['MRR@10']:.3f}")
