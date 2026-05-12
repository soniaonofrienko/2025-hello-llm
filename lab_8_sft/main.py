"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Callable, cast, Iterable, Sequence

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import SFTParams
from core_utils.project.lab_settings import LabSettings

current_path = Path(__file__).parent
settings = LabSettings(current_path / "settings.json")


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        dataset_name = settings.parameters.dataset

        raw_data = load_dataset(dataset_name, name="simplified", split="validation").to_pandas()

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
    Custom implementation of data preprocessor.
    """

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        df = self._data if self._data is not None else self._raw_data
        
        df_for_duplicates = df.copy()
        for col in df_for_duplicates.columns:
            if df_for_duplicates[col].dtype == 'object':
                df_for_duplicates[col] = df_for_duplicates[col].apply(
                    lambda x: str(x) if isinstance(x, (list, tuple, np.ndarray)) else x
                )

        text_column = "source" if "source" in df.columns else "ru_text"
        df_clean = df.dropna(subset=[text_column]) if text_column in df.columns else df


        analysis = {
            "dataset_number_of_samples": int(len(df)),
            "dataset_columns": int(len(df.columns)),
            "dataset_duplicates": int(df_for_duplicates.duplicated().sum()),
            "dataset_empty_rows": int(df.isna().all(axis=1).sum()),
            "dataset_sample_min_len": int(df_clean[text_column].str.len().min()) if text_column in df.columns else 0,
            "dataset_sample_max_len": int(df_clean[text_column].str.len().max()) if text_column in df.columns else 0,
        }

        return analysis
    
    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        df = self._raw_data.copy()

        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(lambda x: tuple(x))
        
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        if "text" in df.columns:
            df = df.drop(columns=["text"])

        remove_labels = [0, 4, 5, 6, 7, 8, 10, 12, 15, 18, 21, 22, 23]
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(lambda x: tuple(label for label in x if label not in remove_labels))

        if "labels" in df.columns:
            df = df.rename(columns={"labels": ColumnNames.TARGET})
        if "ru_text" in df.columns:
            df = df.rename(columns={"ru_text": ColumnNames.SOURCE})

        emotion_mapping = {
            1: 1, 13: 1, 17: 1, 20: 1,
            9: 2, 16: 2, 24: 2, 25: 2,
            14: 3, 19: 3,
            2: 4, 3: 4,
            27: 7,
            26: 6,
        }

        if "target" in df.columns:
            df["target"] = df["target"].apply(lambda x: emotion_mapping.get(x[0], 8) if isinstance(x, tuple) and len(x) > 0 else 8)
        if "target" in df.columns:
            df = df[df["target"] != 8]

        sequential_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
        
        if "target" in df.columns:
            df["target"] = df["target"].map(sequential_mapping)

        if "source" in df.columns:
            df["source"] = df["source"].str.strip()

        df = df.reset_index(drop=True)
        self._data = df


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
        self._data = data

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
        return str(row.iloc[0]), str(row.iloc[1])
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(
    sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Tokenize sample.

    Args:
        sample (pandas.Series): sample from a dataset
        tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
            original data
        max_length (int): max length of sequence

    Returns:
        dict[str, torch.Tensor]: Tokenized sample
    """
    tokens = tokenizer(sample[ColumnNames.SOURCE.value], return_tensors='pt', padding='max_length',
                       truncation=True, max_length=max_length)

    return {
        'input_ids': tokens['input_ids'][0],
        'attention_mask': tokens['attention_mask'][0],
        'labels': sample[ColumnNames.TARGET.value]
    }


class TokenizedTaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
            tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
                tokenize the dataset
            max_length (int): max length of a sequence
        """
        self._data = [tokenize_sample(row, tokenizer, max_length) for _, row in data.iterrows()]

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """
        return self._data[index]


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)

        model_config = AutoConfig.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model = self._model.to(device)
        self._model.eval()

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
        if self._model is None:
            return {}
        
        config = self._model.config
        max_len = config.max_position_embeddings
        embedding_size = self._model.config.hidden_size
        vocab_size = self._tokenizer.vocab_size

        input_ids = torch.ones((1, max_len), dtype=torch.long).to(self._device)
        attention_mask = torch.ones((1, max_len), dtype=torch.long).to(self._device)
        
        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not isinstance(self._model, Module):
            raise ValueError("The model has incompatible type")

        stats_summary = summary(
            self._model,
            input_data=tokens,
            device=self._device,
            verbose=0
        )

        print({"input_shape": [1, self._max_length], "embedding_size": self._model.config.hidden_size, "vocab_size": self._tokenizer.vocab_size, "num_parameters": int(stats_summary.total_params)})

        return {
            "embedding_size": max_len,
            "input_shape": {
                "input_ids": [1, max_len],
                "attention_mask": [1, max_len]
            },
            "max_context_length": 20,
            "num_trainable_params": int(stats_summary.trainable_params),
            "output_shape": [1, config.num_labels],
            "size": int(stats_summary.total_param_bytes),
            "vocab_size": config.vocab_size
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if not self._model or not self._tokenizer:
            return None

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions, references = [], []

        for sample in self._dataset:
            source_text = sample[0]
            target_text = sample[1]

            pred = self._infer_batch([(source_text,)])[0]

            predictions.append(pred)
            references.append(target_text)

        return pd.DataFrame({
            ColumnNames.TARGET.value: references,
            ColumnNames.PREDICTION.value: predictions
        })

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        reviews = [item[0] for item in sample_batch]

        if not sample_batch or self._model is None:
            return []

        inputs = self._tokenizer(
            reviews,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt"
        )

        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        outputs = self._model(**inputs)
        logits = outputs.logits

        predicted_indices = torch.argmax(logits, dim=-1)

        result = [str(idx.item()) for idx in predicted_indices]
    
        return cast(list[str], result)


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
        super().__init__(
            data_path=data_path,
            metrics=metrics
        )
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        result = {}
        for metric in self._metrics:
            metric_name = metric.value
            metric_fn = evaluate.load(metric_name)
            
            preds = data[ColumnNames.PREDICTION.value].astype(int).tolist()
            
            refs = []
            for val in data[ColumnNames.TARGET.value]:
                if isinstance(val, str) and val.startswith('('):
                    clean = val.strip('(), ')
                    refs.append(int(clean))
                elif isinstance(val, tuple):
                    refs.append(int(val[0]))
                else:
                    refs.append(int(val))
            
            metric_result = metric_fn.compute(
                predictions=preds,
                references=refs
            )

            for key, value in metric_result.items():
                if key not in ['mid', 'overall', 'confidence_interval']:
                    result[metric_name] = value
                    break
        
        return result


class SFTPipeline(AbstractSFTPipeline):
    """
    A class that initializes a model, fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        sft_params: SFTParams,
        data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize an instance of ClassificationSFTPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            sft_params (SFTParams): Fine-Tuning parameters.
            data_collator (Callable[[AutoTokenizer], torch.Tensor] | None, optional): processing
                                                                    batch. Defaults to None.
        """

    def run(self) -> None:
        """
        Fine-tune model.
        """
