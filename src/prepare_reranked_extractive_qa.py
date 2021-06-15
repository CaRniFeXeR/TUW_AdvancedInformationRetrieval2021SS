from datetime import datetime
from typing import List, Dict, Tuple

import torch
from allennlp.common import Tqdm
from allennlp.data import PyTorchDataLoader, Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch import nn

from src.core_metrics import unrolled_to_ranked_result
from src.data_loading import IrLabeledTupleDatasetReader
from src.model_fk import FK
from src.model_tk import TK


def prepare_top1_documents(config: dict, top: int = 1):
    onGPU = True
    vocab = Vocabulary.from_files("data/allen_vocab_lower_10")
    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
    _tuple_reader = _tuple_reader.read(config["testset"])
    _tuple_reader.index_with(vocab)
    testdata_loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

    tokens_embedder = Embedding(vocab=vocab,
                                pretrained_file="data/glove.42B.300d.txt",
                                embedding_dim=300,
                                trainable=True,
                                padding_index=0)
    word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
    model = FK(word_embedder, n_kernels=11, n_layers=2, n_fnet_dim=300)
    model.load_state_dict(torch.load(config["model_path"]))

    if onGPU:
        if hasattr(model, "moveModelToGPU"):
            model = model.moveModelToGPU()
        else:
            model = model.cuda()

    def modelForwardPassOnTupleBatchData(model: nn.Module, batch: dict, onGPU: bool):
        if onGPU:
            batch["query_tokens"]["tokens"]["tokens"] = batch["query_tokens"]["tokens"]["tokens"].to(device="cuda")
            batch["doc_tokens"]["tokens"]["tokens"] = batch["doc_tokens"]["tokens"]["tokens"].to(device="cuda")

        return model(batch["query_tokens"]["tokens"], batch["doc_tokens"]["tokens"])

    def evaluateModelOnBatch(model: nn.Module, batch: dict, resultDict: Dict[int, List[Tuple[int, float]]],
                             onGPU: bool) -> dict:
        output = modelForwardPassOnTupleBatchData(model, batch, onGPU)
        for idx, query_id in enumerate(batch["query_id"]):
            if not query_id in resultDict:
                resultDict[query_id] = []
            resultDict[query_id].append((batch["doc_id"][idx], float(output[idx].cpu())))
        return resultDict

    resultDict: Dict[int, List[Tuple[int, float]]] = {}

    for batch in Tqdm.tqdm(testdata_loader):
        resultDict = evaluateModelOnBatch(model, batch, resultDict, onGPU)

    ranked_result = unrolled_to_ranked_result(resultDict)

    with open(config["intermediate_output"], 'w') as fp:
        for qid, docs in ranked_result.items():
            for t in range(top):
                fp.write(f'{qid}\t{docs[t]}\n')


def read_file(path: str, sep='\t'):
    with open(path, 'r', encoding="utf8") as fp:
        return [x.strip().split(sep) for x in fp.readlines()]


def prepare_output_file(config: dict):
    answers = {(x[0], x[1]): (x[4:]) for x in read_file(config["answers"])}
    answers_queries = {x[0] for x in read_file(config["answers"])}

    ms_marco_tuples_with_text = {(x[0], x[1]): (x[2], x[3]) for x in read_file(config["testset"])}
    ms_marco_top1_docs_per_query = {(x[0], x[1]) for x in read_file(config["intermediate_output"])}
    with open(config["output_file"], 'w', encoding='utf8') as fp:
        empty = 0
        skipped = 0
        for q, d in ms_marco_top1_docs_per_query:
            if q not in answers_queries:
                skipped += 1
                continue
            query_txt, document_txt = ms_marco_tuples_with_text[(q, d)]

            answer = []
            if (q, d) in answers:
                answer = answers[(q, d)]

            if len(answer) == 0:
                empty += 1
            answer_str = "\t".join(answer)
            fp.write(f'{q}\t{d}\t0\t{query_txt}\t{document_txt}\t{answer_str}\n')
        print(f'No appropriate reference answer found for {empty} queries, skipped = {skipped}')


def prepare(output_file: str = None):
    if output_file is None:
        output_file = f'./outdir/ms_marco_top1_query_and_doc_{datetime.now().strftime("%d_%m_%Y %H_%M")}.tsv'
    config = {
        "model_path": "./data/models/model_fk_best.pt",
        "intermediate_output": f'./outdir/ms_marco_top1_{datetime.now().strftime("%d_%m_%Y %H_%M")}.tsv',
        "output_file": output_file,
        "testset": "data/msmarco_tuples.test.tsv",
        "answers": './data/fira.qrels.qa-answers.tsv'
    }

    prepare_top1_documents(config, top=1)
    prepare_output_file(config)


if __name__ == '__main__':
    prepare()
