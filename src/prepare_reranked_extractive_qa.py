from collections import namedtuple, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

import torch
from allennlp.common import Tqdm
from allennlp.data import PyTorchDataLoader, Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch import nn
import pandas as pd
from src.core_metrics import unrolled_to_ranked_result
from src.data_loading import IrLabeledTupleDatasetReader
from src.model_conv_knrm import Conv_KNRM
from src.model_knrm import KNRM
from src.model_tk import TK


def prepare_top1_documents(output_file:str, top:int = 1):
    onGPU = True
    vocab = Vocabulary.from_files("data/allen_vocab_lower_10")
    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
    _tuple_reader = _tuple_reader.read("data/msmarco_tuples.test.tsv")
    # _tuple_reader = _tuple_reader.read("data/only_dinosaur.txt")
    _tuple_reader.index_with(vocab)
    testdata_loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

    tokens_embedder = Embedding(vocab=vocab,
                                pretrained_file="data/glove.42B.300d.txt",
                                embedding_dim=300,
                                trainable=True,
                                padding_index=0)
    word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
    model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10, tf_projection_dim=30)
    model_path = "./data/models/model_tk_08_06_2021 20_04.pt"
    model.load_state_dict(torch.load(model_path))

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

    with open(output_file, 'w') as fp:
        for qid, docs in ranked_result.items():
            for t in range(top):
                fp.write(f'{qid}\t{docs[t]}\n')

def read_file(path: str, sep='\t'):
    with open(path, 'r', encoding="utf8") as fp:
        return [x.strip().split(sep) for x in fp.readlines()]


def prepare_output_file(input_file:str, output_file:str):
    answers = {(x[0], x[1]):(x[4:]) for x in read_file('./data/fira.qrels.qa-answers.tsv')}
    answers_queries = {x[0] for x in read_file('./data/fira.qrels.qa-answers.tsv')}
    qrels = {(x[0],x[2]) for x in read_file('./data/fira.qrels.retrieval.tsv', sep=' ')}

    ms_marco_tuples_with_text= {(x[0], x[1]):(x[2], x[3]) for x in read_file('./data/msmarco_tuples.test.tsv')}
    ms_marco_top1_docs_per_query = {(x[0],x[1]) for x in read_file(input_file)}
    with open(output_file, 'w', encoding='utf8') as fp:
        empty = 0
        skipped = 0
        for q,d in ms_marco_top1_docs_per_query:

            if q not in answers_queries:#or q not in qrel_queries:
                skipped += 1
                continue
            query_txt, document_txt =  ms_marco_tuples_with_text[(q,d)]

            answer = []
            if (q,d) in answers:
                answer = answers[(q,d)]

            if len(answer) == 0:
                empty += 1
            answer_str = "\t".join(answer)
            fp.write(f'{q}\t{d}\t0\t{query_txt}\t{document_txt}\t{answer_str}\n')
        print(f'No appropriate reference answer found for {empty} queries, skipped = {skipped}')


if __name__ == '__main__':
    prepared_file = f'./outdir/ms_marco_top1_{datetime.now().strftime("%d_%m_%Y %H_%M")}.tsv'
    output_file = f'./outdir/ms_marco_top1_query_and_doc_{datetime.now().strftime("%d_%m_%Y %H_%M")}.tsv'

    prepare_top1_documents(prepared_file, top=1)
    prepare_output_file(prepared_file, output_file)

# ms_marco_tuples= {(x[0], x[1]) for x in read_file('./data/msmarco_tuples.test.tsv')}
#
# test_pairs = {(q,d) for q,d in ms_marco_tuples if q in retrieval_queries and q in answers_queries}
# test_queries = {q for q,d in ms_marco_tuples if q in retrieval_queries and q in answers_queries}
# retrieval = {(x[0], x[2]) for x in read_file('./data/fira.qrels.retrieval.tsv', sep=' ')}
# retrieval_queries = {x[0] for x in read_file('./data/fira.qrels.retrieval.tsv', sep=' ')}

# print(len(answers))
# print(len(answers_queries))
# print(len(ms_marco_tuples_with_text))
# print(len(ms_marco_tuples))
# print(len(test_queries))
# print(len(test_pairs))
#
# exit(0)
#
# fira_qrels = read_file('./data/fira.qrels.retrieval.tsv', sep=' ')
# fira_tuples = {(x[0], x[1]): (x[3], x[4]) for x in read_file('./data/fira.qrels.qa-tuples.tsv', sep='\t')}
#
# fira_tuple_docs = {x[1] for x in fira_tuples}
#
# # queries that are in retrieval file
# retrieval_queries = {i[0] for i in fira_qrels}
#
# asd = len({i[0] for i in fira_qrels if (i[0], i[2]) in fira_tuples})
# print(asd)
#
# # query document pairs with an answer
# answers = {(a[0], a[1]): a[3] for a in read_file("./data/fira.qrels.qa-answers.tsv")}
#
# answer_queries = {a[0] for a, b in answers if a in retrieval_queries}
#
# query_documents_relevance = defaultdict(list)
# for queryid, documentid in {(i[0], i[2]) for i in fira_qrels}:
#     query_documents_relevance[queryid].append(documentid)
#
# ms_marco_documents = {x[1]: x[3] for x in read_file('./data/msmarco_tuples.test.tsv')}
# ms_marco_queries = {x[0]: x[2] for x in read_file('./data/msmarco_tuples.test.tsv') if
#                     x[0] in retrieval_queries and x[0] in query_documents_relevance}
#
# for queryid, query in ms_marco_queries.items():
#     docs_to_consider = query_documents_relevance[queryid]
#     documents = {d: ms_marco_documents[d] for d in docs_to_consider if d in ms_marco_documents}
#
#     print(f'{queryid} - {len(documents)}')
#
# # fira_tuples = read_file('./data/fira.qrels.qa-tuples.tsv')
# # fira_tuples = {(i[0], i[1], i[3], i[4]): i[5:] for i in fira_tuples if i[0] in dict}
# #
# # aa = {a for a, b, c, d in fira_tuples.keys()}
#
# print(len(ms_marco_queries))
