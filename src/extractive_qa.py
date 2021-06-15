import os
import pickle
from datetime import datetime
from types import FunctionType
from typing import Callable, List

import numpy as np
from numpy import average
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from core_metrics import compute_exact, compute_f1
from src.prepare_reranked_extractive_qa import prepare


class QAInput:

    def __init__(self, question: str, context: str) -> None:
        self.question = question
        self.context = context


class QAResult:

    def __init__(self, answer: str) -> None:
        self.answer = answer


class QAPredicton(QAResult):
    def __init__(self, score: float, start: int, end: int, answer: str) -> None:
        self.score = score
        self.start = start
        self.end = end
        super(QAPredicton, self).__init__(answer)


class QAScore:

    def __init__(self, predicted_result: QAResult, target_results: List[QAResult],
                 scoreFn: Callable[[str, str], float]) -> None:
        self.predicted_result = predicted_result
        self.target_results = target_results
        self.scoreFn = scoreFn

    @property
    def scores(self) -> List[float]:
        '''
        returns all answer combiniations
        '''
        result = []
        for target_answer in self.target_results:
            result.append(self.scoreFn(target_answer.answer, self.predicted_result.answer))

        return result

    @property
    def max_score(self) -> float:
        '''
        returns the max score of all answers
        '''

        scores = self.scores
        if len(scores) == 0:
            return -1
        return max(scores)

    def __str__(self) -> str:

        parsed_scores = [f"'{t.answer}' : '{s}'" for t, s in zip(self.target_results, self.scores)]
        result = f" max: '{self.max_score:.3f}' all: '{parsed_scores}'"

        return result


class QAEvaluation:

    def __init__(self, qa_input: QAInput, predicted_result: QAResult, target_results: List[QAResult]) -> None:
        self.qa_input = qa_input
        self.predicted_result = predicted_result
        self.target_results = target_results

    @property
    def f1(self) -> QAScore:
        '''
        define score of this evaulation
        '''
        f1_scorer = QAScore(self.predicted_result, self.target_results, compute_f1)

        return f1_scorer

    @property
    def exact(self) -> QAScore:
        '''
        define score of this evaulation
        '''
        exact_scorer = QAScore(self.predicted_result, self.target_results, compute_exact)

        return exact_scorer

    def __str__(self) -> str:
        result = f"question: '{self.qa_input.question}' predicted: '{self.predicted_result.answer}' \n f1: {self.f1} \n exact: {self.exact}"

        return result


class QAData:
    def __init__(self, qa_input: QAInput, answers: List[QAResult]) -> None:
        self.qa_input = qa_input
        self.answers = answers


class QADataLoader:

    def __init__(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise ValueError(f"path not found '{filepath}'")

        self.filepath = filepath

    def load(self) -> List[QAData]:
        result = []
        print(f"loading qa data from: '{self.filepath}'... ")
        with open(self.filepath, "r", encoding="utf8") as file:
            for line in file:
                try:
                    tab_sep_line = line.strip().split("\t")  # load data by tab seperated
                    tab_sep_line = list(filter(None, tab_sep_line))  # remove "empty" columns

                    if not len(tab_sep_line) >= 5:
                        raise IOError(f"'{line}' is not valid format")

                    qa_input = QAInput(tab_sep_line[3], tab_sep_line[4])

                    qa_answers = []

                    for qa_answer in tab_sep_line[5:]:
                        qa_answers.append(QAResult(qa_answer))

                    qa_data = QAData(qa_input, qa_answers)
                    result.append(qa_data)
                except:
                    raise IOError(f"'{line}' is not valid format")

        print(f"loaded qa data from: '{self.filepath}' successfully data length: '{len(result)}'")
        return result


def evaluate_results(evaluations: List[QAEvaluation]):
    f1_scores = []
    exact_scores = []
    for evaluation in evaluations:
        f1_scores.append(evaluation.f1.max_score)
        exact_scores.append(evaluation.exact.max_score)

    f1_scores = np.array(f1_scores)
    exact_scores = np.array(exact_scores)

    print(
        f'F1:     Average = {f1_scores.mean()}, StdDev = {f1_scores.std()}, Zero = {len(f1_scores) - np.count_nonzero(f1_scores)}, Total = {len(f1_scores)}')
    print(
        f'Exact:  Average = {exact_scores.mean()}, StdDev = {exact_scores.std()}, Zero = {len(exact_scores) - np.count_nonzero(exact_scores)}, Total = {len(f1_scores)}')


def runModel(config: dict):
    qa_loader = QADataLoader(config["input_data_path"])
    model_name = config["model_name"]
    qa_dataset = qa_loader.load()

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)

    qa_evaluations = []

    for i, qa_data in enumerate(qa_dataset):
        print(f'iteration {i}')
        res = nlp(qa_data.qa_input.__dict__, max_seq_len=180, max_question_len=30)
        qa_prediction = QAPredicton(**res)

        qa_evaluation = QAEvaluation(qa_data.qa_input, qa_prediction, qa_data.answers)
        qa_evaluations.append(qa_evaluation)

        if i == 42 or i % 100 == 0:
            print(qa_evaluation)

        if i % 1000 == 0:
            with open(f'./outdir/pickles/intermediate_res_{datetime.now().strftime("%d_%m_%Y %H_%M")}.pickle',
                      "wb") as fp:
                pickle.dump(qa_evaluations, fp)

    with open(f'./outdir/pickles/final_res_{datetime.now().strftime("%d_%m_%Y %H_%M")}.pickle', "wb") as fp:
        pickle.dump(qa_evaluations, fp)

    evaluate_results(qa_evaluations)


def evaluate(config: dict):
    with open(config["model_result_pickle"], "rb") as fp:
        data = pickle.load(fp)

        f1_scores = np.array([d.f1.max_score for d in data])
        exact_scores = np.array([d.exact.max_score for d in data])

        print(
            f'F1:     Average = {f1_scores.mean()}, StdDev = {f1_scores.std()}, Zero = {len(f1_scores) - np.count_nonzero(f1_scores)}, Total = {len(f1_scores)}')
        print(
            f'Exact:  Average = {exact_scores.mean()}, StdDev = {exact_scores.std()}, Zero = {len(exact_scores) - np.count_nonzero(exact_scores)}, Total = {len(f1_scores)}')


if __name__ == '__main__':
    config = {
        "mode": "rerank",   # else arbitrary... :)
        "input_data_path": "data/fira.qrels.qa-tuples.tsv",
        # "input_data_path": "outdir/ms_marco_top1_query_and_doc_15_06_2021 21_27.tsv",
        "model_name": "distilbert-base-uncased-distilled-squad",  # deepset/roberta-base-squad2
        "model_result_pickle": "./outdir/pickles/final_res_14_06_2021 03_02.pickle",
        # "model_result_pickle":'./outdir/pickles/final_res_15_06_2021 22_47.pickle'
    }

    if config["mode"] == "rerank":
        prepare(config["input_data_path"])

    runModel(config)
    # evaluate(config)
