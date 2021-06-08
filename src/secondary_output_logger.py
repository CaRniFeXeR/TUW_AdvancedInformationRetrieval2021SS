from abc import ABC, abstractmethod
import torch
import numpy
from pathlib import Path
import copy
from typing import TypeVar, Generic, List, Dict, Any
from datetime import datetime

ROOT_PATH: Path = Path('logs/')

class SecondaryBatchOutput():
    def __init__(self, score: torch.Tensor, per_kernel: torch.Tensor, query_embeddings: torch.Tensor, query_embeddings_oov_mask: torch.Tensor, cosine_matrix: torch.Tensor):
        self.__score: torch.Tensor = score
        self.__per_kernel: torch.Tensor = per_kernel
        self.__query_mean_vector: torch.Tensor = (query_embeddings.sum(dim=1) / query_embeddings_oov_mask.sum(dim=1).unsqueeze(-1))
        self.__cosine_matrix: torch.Tensor = cosine_matrix

    @property
    def score(self) -> torch.Tensor:
        return self.__score

    @property
    def per_kernel(self) -> torch.Tensor:
        return self.__per_kernel

    @property
    def query_mean_vector(self) -> torch.Tensor:
        return self.__query_mean_vector

    @property
    def cosine_matrix(self) -> torch.Tensor:
        return self.__cosine_matrix

class SecondaryBatchOutputLogger(ABC):
    @abstractmethod
    def log(self, secondary_batch_output: SecondaryBatchOutput):
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
        
    def log(self, secondary_batch_output: SecondaryBatchOutput):
        pass