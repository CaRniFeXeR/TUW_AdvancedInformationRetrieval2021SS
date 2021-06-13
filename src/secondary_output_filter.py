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
from secondary_output_logger import SecondaryBatchOutput, SecondaryBatchOutputFileLogger, SecondaryBatchOutputFileLoggerConfig, SecondaryBatchOutputLogger, TKModelData, FKModelData, SecondaryOutput, ScoreDelta
from pathlib import Path


def main():
    config = {
        "secondary_output_file_1": "logs/fk_logger/log_13_06_2021 17_09.npz",
        "secondary_output_file_2": "logs/tk_logger/log_13_06_2021 17_38.npz",
        "out_dir": "filtered_logs",
        "top_n": 20
    }

    secondary_output_file_1 = numpy.load(file=config["secondary_output_file_1"], allow_pickle=True)
    secondary_output_file_2 = numpy.load(file=config["secondary_output_file_2"], allow_pickle=True)

    secondary_outputs_1: List[SecondaryOutput] = SecondaryOutput.from_dict(secondary_output_file_1.get("qd_data")[()])
    secondary_outputs_2: List[SecondaryOutput] = SecondaryOutput.from_dict(secondary_output_file_2.get("qd_data")[()])

    score_deltas: List[ScoreDelta] = []

    for secondary_output_1 in secondary_outputs_1:
        corresponding_secondary_outputs_2: List[SecondaryOutput] = [secondary_output_2 for secondary_output_2 in secondary_outputs_2 if secondary_output_2.query_id == secondary_output_1.query_id and secondary_output_2.doc_id == secondary_output_1.doc_id]

        if not len(corresponding_secondary_outputs_2) > 0:
            continue

        corresponding_secondary_output_2: SecondaryOutput = corresponding_secondary_outputs_2[0]

        score_deltas.append(ScoreDelta(query_id=secondary_output_1.query_id, doc_id=secondary_output_1.doc_id, score_1=secondary_output_1.score[0], score_2=corresponding_secondary_output_2.score[0]))

    score_deltas.sort(key=lambda x: x.value, reverse=True)

    new_secondary_outputs_1: Dict[str, SecondaryOutput] = {}
    new_secondary_outputs_2: Dict[str, SecondaryOutput] = {}

    top_n = config["top_n"] if not config["top_n"] > len(score_deltas) else len(score_deltas)

    for score_delta in score_deltas:
        if score_delta.query_id in new_secondary_outputs_1 or score_delta.query_id in new_secondary_outputs_2:
            continue

        new_secondary_outputs_1[score_delta.query_id] = [secondary_output_1 for secondary_output_1 in secondary_outputs_1 if secondary_output_1.query_id == score_delta.query_id and secondary_output_1.doc_id == score_delta.doc_id][0]
        new_secondary_outputs_2[score_delta.query_id] = [secondary_output_2 for secondary_output_2 in secondary_outputs_2 if secondary_output_2.query_id == score_delta.query_id and secondary_output_2.doc_id == score_delta.doc_id][0]

        if len(new_secondary_outputs_1) == top_n:
            break

    out_file_path_1: Path = Path(config["out_dir"]) / Path(config["secondary_output_file_1"]).name
    out_file_path_2: Path = Path(config["out_dir"]) / Path(config["secondary_output_file_2"]).name

    numpy.savez_compressed(out_file_path_1.absolute(), model_data=secondary_output_file_1.get("model_data")[()], qd_data=SecondaryOutput.to_dict(list(new_secondary_outputs_1.values())))
    numpy.savez_compressed(out_file_path_2.absolute(), model_data=secondary_output_file_2.get("model_data")[()], qd_data=SecondaryOutput.to_dict(list(new_secondary_outputs_2.values())))

if __name__ == "__main__":
    main()
