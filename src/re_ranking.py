from abc import ABC, abstractmethod
from datetime import datetime
from re import purge
import wandb
from typing import Tuple, Type
from numpy import isin, ndarray
from torch._C import Value
import torch
from torch.functional import Tensor
from model_fk import FK
from model_tk import TK
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

# region [EarlyStopping]


class EarlyStoppingCriteria(ABC):

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self._reason = ""

    @abstractmethod
    def check(self, loss: float) -> bool:
        pass

    @property
    def reason(self) -> str:
        if self._reason == "":
            return ""
        else:
            return f"{self.name} violated: {self._reason}"


class MinStdCritera(EarlyStoppingCriteria):

    def __init__(self, min_std: float, window_size: int):
        super().__init__(name="MinStdCritera")

        if not isinstance(min_std, float):
            raise TypeError("min_std must be a float")

        if not min_std > 0:
            raise ValueError("min_std must be a positive value")

        if not isinstance(window_size, int):
            raise TypeError("window_size must be a int")

        if not window_size > 0:
            raise ValueError("window_size must be a positive value")

        self.min_std = min_std
        self.window_size = window_size

    def reset(self):
        self.__lossList: List[float] = []
        super().reset()

    def check(self, loss: float) -> bool:

        if len(self.__lossList) > self.window_size:
            self.__lossList.pop(0)

        result = False

        if len(self.__lossList) >= self.window_size and np.std(self.__lossList) <= self.min_std:
            result = True
            self._reason = f"window_size '{self.window_size}' loss std: '{np.std(self.__lossList):.3f}' min_std: '{self.min_std}'"

        return result


class MaxIterationCriteria(EarlyStoppingCriteria):

    def __init__(self, max_iteration: int):
        super().__init__("MaxIterationCriteria")

        if not isinstance(max_iteration, int):
            raise TypeError("max_iteration must be an integer")

        if not max_iteration > 0:
            raise ValueError("max_interation must be positive")

        self.max_iteration = max_iteration

    def reset(self):
        self.n_iteration = 0
        super().reset()

    def check(self, loss: float) -> bool:
        result = False

        self.n_iteration += 1

        if self.n_iteration >= self.max_iteration:
            result = True
            self._reason = f"MaxIteration"

        return result


class MinDeltaCriteria(EarlyStoppingCriteria):

    def __init__(self, min_delta: float):
        super().__init__("MinDeltaCriteria")

        if not isinstance(min_delta, float):
            raise TypeError("min_delta must be a float")

        if not min_delta > 0:
            raise ValueError("min_delta must be positive")

        self.min_delta = min_delta

    def reset(self):
        self.best_score = None
        super().reset()

    def check(self, loss: float) -> bool:
        result = False

        if not self.best_score == None and not (self.best_score - loss) >= self.min_delta:
            result = True
            self._reason = f"best_score '{self.best_score}' loss '{loss}' delta: '{self.best_score - loss}' min_delta: '{self.min_delta}'"

        if self.best_score == None or self.best_score > loss:
            self.best_score = loss

        return result


class EarlyStoppingWatcher:

    def __init__(self, patience: int = 5):

        if not isinstance(patience, int):
            raise TypeError("patience must be an integer")

        if patience < 0:
            raise ValueError(f"patience must be greater zero value given :'{patience}'")

        self.patience = patience

        self.n_iterations = 0
        self.n_strike = 0
        self.criteriaList: List[EarlyStoppingCriteria] = []

    def addCriteria(self, criteria: EarlyStoppingCriteria):

        if not isinstance(criteria, EarlyStoppingCriteria):
            raise TypeError("criteria must be a EarlyStoppingCriteria")

        self.criteriaList.append(criteria)
        return self

    def checkCriteriasViolated(self, loss: float) -> bool:

        result = False

        for criteria in self.criteriaList:
            if criteria.check(loss):
                result = True

        return result

    def watchLoss(self, loss: float) -> bool:

        result = False

        if (self.checkCriteriasViolated(loss)):
            self.n_strike += 1
        else:
            self.n_strike = 0

        if self.n_strike >= self.patience:
            result = True

        return result

    def reset(self):

        for criteria in self.criteriaList:
            criteria.reset()

    @property
    def has_no_strikes(self) -> bool:
        return self.n_strike == 0

    @property
    def reason(self) -> str:

        result = f"strikes: {self.n_strike} "

        for criteria in self.criteriaList:
            result += criteria.reason

        return result

# endregion

# region [Model ForwardPass]


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
    batch["query_tokens"]["tokens"]["id"] = batch["query_id"]
    batch["doc_tokens"]["tokens"]["id"] = batch["doc_id"]

    if onGPU:
        batch["query_tokens"]["tokens"]["tokens"] = batch["query_tokens"]["tokens"]["tokens"].to(device="cuda")
        batch["doc_tokens"]["tokens"]["tokens"] = batch["doc_tokens"]["tokens"]["tokens"].to(device="cuda")

    return model(batch["query_tokens"]["tokens"], batch["doc_tokens"]["tokens"])

# endregion

# region [Model Evaluation]


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

# endregion

# region [Config & wandb]

def main():
    torch.manual_seed(32)

    # change paths to your data directory
    config = {
        "vocab_directory": "data/allen_vocab_lower_10",
        "pre_trained_embedding": "data/glove.42B.300d.txt",
        "model": "conv_knrm",
        "train_data": "data/triples.train.tsv",
        "validation_data": "data/msmarco_tuples.validation.tsv",
        "test_data": "data/msmarco_tuples.test.tsv",
        "qrels_data": "data/msmarco_qrels.txt",
        "onGPU": torch.cuda.is_available(),
        "train_word_embedding": True,
        "n_training_epochs": 3,
        "traning_batch_size": 128,
        "eval_batch_size": 128,  
        "validation_interval": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.000000000000001,
        "use_wandb": True,
        "wandb_entity": "floko",
        "wandb_log_interval": 10

    }

    onGPU = config["onGPU"]

    use_wandb = config["use_wandb"]
    wandb_config = {}

    if use_wandb:
        # todo refactor wandb config use
        wandb.init(project='air-2021SS', entity=config["wandb_entity"])
        wandb_config = wandb.config
        wandb_config["model"] = config["model"]
        wandb_config["validation_data"] = config["validation_data"]
        wandb_config["test_data"] = config["test_data"]
        wandb_config["train_data"] = config["train_data"]
        wandb_config["learning_rate"] = config["learning_rate"]
        wandb_config["weight_decay"] = config["weight_decay"]

    # endregion

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
        #"learning_rate": 0.001 useful for conv_knrm to perform
        model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
    elif config["model"] == "tk":
        # "learning_rate": 0.0001 needed for tk to perform
        model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10, tf_projection_dim=30)
    elif config["model"] == "fk":
        # learning_rate" : 0.001 needed for fk to perform
        model = FK(word_embedder, n_kernels=11, n_layers=2, n_fnet_dim=300)
    else:
        raise ValueError("no known model configured!")

    if use_wandb and hasattr(model, "fill_wandb_config"):
        model.fill_wandb_config(wandb_config)


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

    # don't train word embedder
    paramsToTrain = []
    if hasattr(model, "get_named_parameters"):
        namedParamsIt = model.get_named_parameters()
    else:
        namedParamsIt = model.named_parameters()
    for p_name, par in namedParamsIt:
        if config["train_word_embedding"] == True or not "word_embeddings" in p_name:
            paramsToTrain.append(par)

    optimizer = torch.optim.AdamW(paramsToTrain, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # early stopping
    earlyStoppingWatchter = EarlyStoppingWatcher(patience=150) \
        .addCriteria(MaxIterationCriteria(100000)) \
        .addCriteria(MinDeltaCriteria(0.001)) \
        .addCriteria(MinStdCritera(min_std=0.001, window_size=40))

    earlyStoppingReached = False
    total_batch_count = 0
    model_path = ""

    if use_wandb:
        wandb.watch(model, log='all') 

    for epoch in range(config["n_training_epochs"]):

        if earlyStoppingReached:
            break

        # activate training mode on model
        model.train(mode=True)
        trainLossList = []

        for i, batch in enumerate(Tqdm.tqdm(loader)):

            total_batch_count += 1
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

            if use_wandb and (i + 1) % config["wandb_log_interval"] == 0:
                wandb.log({"batch_loss": current_loss, "global_step": total_batch_count})

            if (i + 1) % config["validation_interval"] == 0:
                # validate only after n_iterations
                # ValidationSet
                model.train(mode=False)
                _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
                _tuple_reader = _tuple_reader.read(config["validation_data"])
                _tuple_reader.index_with(vocab)
                validation_loader = PyTorchDataLoader(_tuple_reader, batch_size=config["eval_batch_size"])

                result = evaluateModel(model, validation_loader, relevanceLabels=qrels, onGPU=onGPU)
                print(f"validationset MRR@10 : {result['MRR@10']:.3f}")
                target_metric = 1 - result['MRR@10']
                if earlyStoppingWatchter.watchLoss(target_metric):
                    print("early stopping criteria reached")
                    print(f"early stopping reason: {earlyStoppingWatchter.reason}")
                    earlyStoppingReached = True
                    break
                elif i > 15 and result['MRR@10'] > 0.14 and earlyStoppingWatchter.has_no_strikes:
                    # best model --> save
                    if model_path == "":
                        model_path = f"outdir/model_{config['model']}_{datetime.now().strftime('%d_%m_%Y %H_%M')}.pt"
                    torch.save(model.state_dict(), model_path)

                model.train(mode=True)

                if use_wandb:
                    wandb.log({"validiation_MRR@10": result['MRR@10'], "global_step": total_batch_count})

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
        print(f"validationset MRR@10 : {result['MRR@10']:.3f}")
        if use_wandb:
            wandb.log({"validiation_MRR@10": result['MRR@10'], "global_step": total_batch_count})


    # region [Evaluation]

    #
    # eval (duplicate for validation inside train loop - but rename "loader", since
    # otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
    #

    if model_path == "":
        torch.save(model.state_dict(), f"outdir/model_{config['model']}_{datetime.now().strftime('%d_%m_%Y %H_%M')}.pt")
    else:
        model.load_state_dict(torch.load(model_path))

    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
    _tuple_reader = _tuple_reader.read(config["test_data"])
    _tuple_reader.index_with(vocab)
    testdata_loader = PyTorchDataLoader(_tuple_reader, batch_size=config["eval_batch_size"])

    model.train(mode=False)

    result = evaluateModel(model, testdata_loader, relevanceLabels=qrels, onGPU=onGPU)
    print(f"testset Score MRR@10 : {result['MRR@10']:.3f}")
    if use_wandb:
        wandb.log({"test_MRR@10": result['MRR@10'], "global_step": total_batch_count})


# endregion

if __name__ == "__main__":
    main()