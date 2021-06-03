
import os
from types import FunctionType
from typing import Callable, List
from numpy import average
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from core_metrics import compute_exact, compute_f1


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

    def __init__(self, predicted_result: QAResult, target_results: List[QAResult], scoreFn: Callable[[str, str], float]) -> None:
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
                    tab_sep_line = line.split("\t")  # load data by tab seperated
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

    scores = []
    for evalulation in evaluations:
        scores.append(evalulation.score)

    print(f"evaluted : '{len(scores)}' avg: result: '{average(scores):.3f}'")


def runModel(config: dict):

    qa_loader = QADataLoader(config["fira_data_path"])
    model_name = config["model_name"]
    qa_dataset = qa_loader.load()

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    qa_evaluations = []

    for i, qa_data in enumerate(qa_dataset):
        res = nlp(qa_data.qa_input.__dict__)
        qa_prediction = QAPredicton(**res)

        qa_evaluation = QAEvaluation(qa_data.qa_input, qa_prediction, qa_data.answers)
        qa_evaluations.append(qa_evaluation)

        if i == 42 or i % 100 == 0:
            print(qa_evaluation)

    evaluate_results(qa_evaluations)


if __name__ == '__main__':

    config = {
        "fira_data_path": "data/fira.qrels.qa-tuples.tsv",
        "model_name": "distilbert-base-uncased-distilled-squad"  # deepset/roberta-base-squad2
    }

    runModel(config)
