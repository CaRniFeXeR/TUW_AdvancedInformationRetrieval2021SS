from abc import ABC, abstractmethod
import torch
import numpy
from pathlib import Path
import copy
from typing import IO, TypeVar, Generic, List, Dict, Any
from datetime import datetime

ROOT_PATH: Path = Path('logs/')

class ModelData(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, numpy.ndarray]:
        pass

class TKModelData(ModelData):
    def __init__(self, dense_weight: torch.Tensor, dense_mean_weight: torch.Tensor, dense_comb_weight: torch.Tensor):
        self.dense_weight: torch.Tensor = dense_weight
        self.dense_mean_weight: torch.Tensor = dense_mean_weight
        self.dense_comb_weight: torch.Tensor = dense_comb_weight

    def to_dict(self) -> Dict[str, numpy.ndarray]:
        return {"dense_weight": self.dense_weight.data.cpu().numpy(), "dense_mean_weight": self.dense_mean_weight.data.cpu().numpy(), "dense_comb_weight": self.dense_comb_weight.data.cpu().numpy()}

class FKModelData(ModelData):
    def __init__(self, dense_weight: torch.Tensor, dense_mean_weight: torch.Tensor, dense_comb_weight: torch.Tensor):
        self.dense_weight: torch.Tensor = dense_weight
        self.dense_mean_weight: torch.Tensor = dense_mean_weight
        self.dense_comb_weight: torch.Tensor = dense_comb_weight

    def to_dict(self) -> Dict[str, numpy.ndarray]:
        return {"dense_weight": self.dense_weight.data.cpu().numpy(), "dense_mean_weight": self.dense_mean_weight.data.cpu().numpy(), "dense_comb_weight": self.dense_comb_weight.data.cpu().numpy()}

class SecondaryBatchOutput():
    def __init__(self, score: torch.Tensor, per_kernel: torch.Tensor, per_kernel_mean: torch.Tensor, cosine_matrix: torch.Tensor, query_id: List[str], doc_id: List[str]):
        self.__score: torch.Tensor = score
        self.__per_kernel: torch.Tensor = per_kernel
        self.__per_kernel_mean: torch.Tensor = per_kernel_mean
        self.__cosine_matrix: torch.Tensor = cosine_matrix
        self.__query_id: List[str] = query_id
        self.__doc_id: List[str] = doc_id

    @property
    def score(self) -> torch.Tensor:
        return self.__score

    @property
    def per_kernel(self) -> torch.Tensor:
        return self.__per_kernel

    @property
    def per_kernel_mean(self) -> torch.Tensor:
        return self.__per_kernel_mean

    @property
    def cosine_matrix(self) -> torch.Tensor:
        return self.__cosine_matrix

    @property
    def query_id(self) -> List[str]:
        return self.__query_id

    @property
    def doc_id(self) -> List[str]:
        return self.__doc_id

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, numpy.ndarray]]]:
        __secondary_output: Dict[str, Dict[str, Dict[str, numpy.ndarray]]] = {}

        for sample_index, query_id in enumerate(self.query_id):
            doc_id: str = self.doc_id[sample_index]

            if not query_id in __secondary_output.keys():
                __secondary_output[query_id] = {}

            if not doc_id in __secondary_output[query_id].keys():
                __secondary_output[query_id][doc_id] = {}
                __secondary_output[query_id][doc_id]["score"] = self.score.cpu()[sample_index].data.numpy()
                __secondary_output[query_id][doc_id]["per_kernel"] = self.per_kernel.cpu()[sample_index].data.numpy()
                __secondary_output[query_id][doc_id]["per_kernel_mean"] = self.per_kernel_mean.cpu()[sample_index].data.numpy()
                __secondary_output[query_id][doc_id]["cosine_matrix_masked"] = self.cosine_matrix.cpu()[sample_index].data.numpy()

        return __secondary_output

class SecondaryOutput():
    def __init__(self, score: numpy.ndarray, per_kernel: numpy.ndarray, per_kernel_mean: numpy.ndarray, cosine_matrix: numpy.ndarray, query_id: str, doc_id: str):
        self.__score: numpy.ndarray = score
        self.__per_kernel: numpy.ndarray = per_kernel
        self.__per_kernel_mean: numpy.ndarray = per_kernel_mean
        self.__cosine_matrix: numpy.ndarray = cosine_matrix
        self.__query_id: str = query_id
        self.__doc_id: str = doc_id

    @property
    def score(self) -> numpy.ndarray:
        return self.__score

    @property
    def per_kernel(self) -> numpy.ndarray:
        return self.__per_kernel

    @property
    def per_kernel_mean(self) -> numpy.ndarray:
        return self.__per_kernel_mean

    @property
    def cosine_matrix(self) -> numpy.ndarray:
        return self.__cosine_matrix

    @property
    def query_id(self) -> str:
        return self.__query_id

    @property
    def doc_id(self) -> str:
        return self.__doc_id

    @staticmethod
    def to_dict(secondary_outputs: List[Any]) -> Dict[str, Dict[str, Dict[str, numpy.ndarray]]]:
        __secondary_output: Dict[str, Dict[str, Dict[str, numpy.ndarray]]] = {}

        for entry in secondary_outputs:
            if not entry.query_id in __secondary_output.keys():
                __secondary_output[entry.query_id] = {}

            if not entry.doc_id in __secondary_output[entry.query_id].keys():
                __secondary_output[entry.query_id][entry.doc_id] = {}
                __secondary_output[entry.query_id][entry.doc_id]["score"] = entry.score
                __secondary_output[entry.query_id][entry.doc_id]["per_kernel"] = entry.per_kernel
                __secondary_output[entry.query_id][entry.doc_id]["per_kernel_mean"] = entry.per_kernel_mean
                __secondary_output[entry.query_id][entry.doc_id]["cosine_matrix_masked"] = entry.cosine_matrix

        return __secondary_output

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Dict[str, numpy.ndarray]]]):
        secondary_outputs: List[SecondaryOutput] = []

        for query_id, doc_dict in data.items():
            for doc_id, secondary_output_dict in doc_dict.items():
                secondary_outputs.append(cls(score=secondary_output_dict["score"], per_kernel=secondary_output_dict["per_kernel"], per_kernel_mean=secondary_output_dict["per_kernel_mean"], cosine_matrix=secondary_output_dict["cosine_matrix_masked"], query_id=query_id, doc_id=doc_id))

        return secondary_outputs

class ScoreDelta():
    def __init__(self, query_id: str, doc_id: str, score_1: float, score_2: float):
        self.__query_id: str = query_id
        self.__doc_id: str = doc_id
        self.__score_1: float = score_1
        self.__score_2: float = score_2

    @property
    def query_id(self) -> str:
        return self.__query_id

    @property
    def doc_id(self) -> str:
        return self.__doc_id

    @property
    def value(self) -> float:
        return abs(self.__score_1 - self.__score_2)

class SecondaryBatchOutputLogger(ABC):
    @abstractmethod
    def log(self, secondary_batch_output: SecondaryBatchOutput):
        pass

    @property
    @abstractmethod
    def model_data(self) -> ModelData:
        pass

    @model_data.setter 
    @abstractmethod
    def model_data(self, value: ModelData):
        pass

class SecondaryBatchOutputFileLoggerConfig():
    def __init__(self, logger_name: str, root_path: Path = None):
        self.logger_name: str = logger_name
        self.root_path: Path = root_path

class SecondaryBatchOutputFileLogger(SecondaryBatchOutputLogger):
    def __init__(self, config: SecondaryBatchOutputFileLoggerConfig):
        if config is None:
            raise ValueError("config can't be empty")

        if config.logger_name is None:
            raise ValueError("logger name can't be empty")

        self.__config: SecondaryBatchOutputFileLoggerConfig = config

        self.__create_folder_structure()
        self.__secondary_output: Dict[str, numpy.ndarray] = {}
        self.__model_data: ModelData = None
        self.__log_file: IO[Any] = open(self.__file_path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__log_file.close()

    @property
    def __root_path(self) -> Path:
        return self.__config.root_path if not self.__config.root_path is None else ROOT_PATH

    @property
    def __store_path(self) -> Path:
        return self.__root_path / self.__config.logger_name

    @property
    def __file_name(self) -> str:
        return f"log_{datetime.now().strftime('%d_%m_%Y %H_%M')}.npz"

    @property
    def __file_path(self) -> Path:
        return self.__store_path / Path(self.__file_name)
            
    def __create_folder_structure(self):
        self.__store_path.absolute().mkdir(parents=True, exist_ok=True)

    def __write_data_to_file(self):
        numpy.savez_compressed(self.__log_file, model_data=(self.__model_data.to_dict() if not self.model_data is None else None), qd_data=self.__secondary_output)

    @property
    def model_data(self) -> ModelData:
        return self.__model_data

    @model_data.setter 
    def model_data(self, value: ModelData):
        self.__model_data = value

        self.__write_data_to_file()
        
    def log(self, secondary_batch_output: SecondaryBatchOutput):
        self.__secondary_output = {**self.__secondary_output, **secondary_batch_output.to_dict()}
        
        self.__write_data_to_file()