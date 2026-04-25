"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable,
# too-many-arguments, super-init-not-called
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

current_path = Path(__file__).parent
settings = LabSettings(current_path / "settings.json")


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the importer without requiring hf_name.
        """
        pass

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        import pandas as pd
        from datasets import load_dataset

        dataset_name = settings.parameters.dataset
        raw_data = load_dataset(dataset_name, split="train")

        if hasattr(raw_data, "to_pandas"):
            self._raw_data = raw_data.to_pandas()
        elif isinstance(raw_data, pd.DataFrame):
            self._raw_data = raw_data
        else:
            raise TypeError(
                f"not pd.DataFrame, but {type(raw_data)}"
            )


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """

        df = self._raw_data

        analysis = {
            "num_samples": len(df),  # количество примеров
            "num_columns": len(df.columns),  # количество колонок
            "column_names": list(df.columns),  # названия колонок
            "missing_values": df.isnull().sum().to_dict(),  # пропуски
            # типы данных
            "dtypes": {str(col): str(dtype) for col, dtype
                       in df.dtypes.items()},
        }

        # проверка на ошибки + добавляем статистику по длине текста
        if "Reviews" in df.columns:
            analysis["avg_review_length"] = df["Reviews"].str.len().mean()
            analysis["min_review_length"] = df["Reviews"].str.len().min()
            analysis["max_review_length"] = df["Reviews"].str.len().max()

        if "Summary" in df.columns:
            analysis["avg_summary_length"] = df["Summary"].str.len().mean()
            analysis["min_summary_length"] = df["Summary"].str.len().min()
            analysis["max_summary_length"] = df["Summary"].str.len().max()

        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        df = self._raw_data.copy()

        # пропуски
        df = df.dropna(subset=["Reviews", "Summary"])

        # дубликаты и лишние пробелы
        df["Reviews"] = df["Reviews"].str.strip()
        df["Summary"] = df["Summary"].str.strip()
        df = df.drop_duplicates()

        df = df.reset_index(drop=True)
        self._preprocessed_data = df


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data.reset_index(drop=True)

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        row = self._data.iloc[index]
        return (row["Reviews"], row["Summary"])

    @property
    def data(self) -> pd.DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int,
        batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._model_name = model_name
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model = self._model.to(device)

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
                dict: Properties of a model
        """
        # параметры модели
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters()
                               if p.requires_grad)

        analysis = {
            "model_name": self._model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self._device,
            "max_length": self._max_length,
            "batch_size": self._batch_size,
            "dataset_size": len(self._dataset),
        }

        return analysis

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        import torch

        review = sample[0]

        # токенизация текста
        inputs = self._tokenizer(
            review,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length
        )

        # генерация текста
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self._max_length,
                num_beams=4,
                early_stopping=True
            )

        # декодирование
        prediction = self._tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        return prediction

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        import torch
        from torch.utils.data import DataLoader

        all_predictions = []
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=False
        )
        # проход по батчам
        for batch in dataloader:
            # batch - это список кортежей [(review1, summary1), (review2,
            # summary2), ...]
            reviews = [item[0] for item in batch]

            # токенизация каждого батча
            inputs = self._tokenizer(
                reviews,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt"
            )
            # генерация для каждого батча
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=self._max_length,
                    num_beams=4,
                    early_stopping=True
                )
            # декодирование
            predictions = self._tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            all_predictions.extend(predictions)

        # датафрейм с результатами
        result_df = self._dataset.data.copy()
        result_df["predictions"] = all_predictions[:len(result_df)]

        return result_df

    @torch.no_grad()
    def _infer_batch(
            self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        reviews = [item[0] for item in sample_batch]

        # токенизация
        inputs = self._tokenizer(
            reviews,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt"
        )
        # генерация
        outputs = self._model.generate(
            **inputs,
            max_length=self._max_length,
            num_beams=4,
            early_stopping=True
        )

        # декодирование
        predictions = self._tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        return predictions


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """

    def run(self) -> dict:
        """
        Evaluate the predictions against the references
        using the specified metric.

        Returns:
            dict: A dictionary containing information
            about the calculated metric
        """
