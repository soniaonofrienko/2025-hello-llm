"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable,
# too-many-arguments, super-init-not-called, useless-parent-delegation
# protected-access, no-any-return

from pathlib import Path
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoConfig, AutoTokenizer, EncoderDecoderModel

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings

current_path = Path(__file__).parent
settings = LabSettings(current_path / "settings.json")


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    def __init__(self, dataset_path: str | None = None) -> None:
        """
        Initialize the importer without requiring hf_name.
        """
        hf_name = dataset_path if dataset_path else settings.parameters.dataset
        super().__init__(hf_name=hf_name)

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
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

    def __init__(self, raw_data: pd.DataFrame) -> None:
        super().__init__(raw_data)
        self._data: pd.DataFrame | None = None

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """

        df = self._data if self._data is not None else self._raw_data

        text_column = "Reviews" if "Reviews" in df.columns else (
            "source" if "source" in df.columns else df.columns[0]
            )

        analysis = {
            "dataset_number_of_samples": int(len(df)),
            "dataset_columns": int(len(df.columns)),
            "dataset_duplicates": int(df.duplicated().sum()),
            "dataset_empty_rows": int(df.isna().all(axis=1).sum()),
            "dataset_sample_min_len": int(df[text_column].str.len().min()) if text_column in df.columns else 0,
            "dataset_sample_max_len": int(df[text_column].str.len().max()) if text_column in df.columns else 0,
        }

        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        df = self._raw_data.copy()

        df = df.dropna(subset=["Reviews", "Summary"])

        df["Reviews"] = df["Reviews"].str.strip()
        df["Summary"] = df["Summary"].str.strip()
        df = df.drop_duplicates()

        df = df.rename(columns={
            "Reviews": ColumnNames.SOURCE.value,
            "Summary": ColumnNames.TARGET.value
        })
        
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
        return str(row[ColumnNames.SOURCE.value]), str(row[ColumnNames.TARGET.value])

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
        super().__init__(model_name, dataset, max_length, batch_size, device)

        model_config = AutoConfig.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = EncoderDecoderModel.from_pretrained(model_name, config=model_config)
        self._model = self._model.to(device)
        self._model.eval()
        self._input_max_length = 512

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

        config = self._model.config
        
        vocab_size = len(self._tokenizer) if hasattr(self._tokenizer, '__len__') else config.vocab_size
        embedding_size = config.hidden_size if hasattr(config, 'hidden_size') else 0
        max_context_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 512

        try:
            from torchinfo import summary
            dummy_input = torch.ones(1, self._input_max_length, dtype=torch.long).to(self._device)
            
            model_stats = summary(
                self._model,
                input_data=dummy_input,
                device=self._device,
                verbose=0,
            )
            total_params = model_stats.total_params
            trainable_params = model_stats.trainable_params
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            print(f"Note: torchinfo fallback due to model architecture,{e}")
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        return {
            "input_shape": [1, self._input_max_length],
            "embedding_size": int(embedding_size),
            "output_shape": [1, self._max_length],
            "num_trainable_params": int(trainable_params),
            "vocab_size": int(vocab_size),
            "size": int(total_params),
            "max_context_length": int(max_context_length),
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
        
        for batch in DataLoader(self._dataset, batch_size=self._batch_size):
            predictions.extend(self._infer_batch(batch))
            references.extend(list(batch[1]))
        

        return pd.DataFrame({
            ColumnNames.TARGET.value: references,
            ColumnNames.PREDICTION.value: predictions
        })

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
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        reviews = [item[0] for item in sample_batch]

        inputs = self._tokenizer(
            reviews,
            padding=True,
            truncation=True,
            max_length=self._input_max_length,
            return_tensors="pt"
        ).to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._max_length,
            num_beams=4,
            early_stopping=True
        )

        predictions: list[str] = self._tokenizer.batch_decode(
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
        super().__init__(
            data_path=data_path,
            metrics=metrics
        )
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references
        using the specified metric.

        Returns:
            dict: A dictionary containing information
            about the calculated metric
        """
        df = pd.read_csv(self._data_path)

        predictions = df["predictions"].tolist()
        references = df["target"].tolist()

        final_metrics = {}

        for metric_name in self._metrics:
            metric_str = (
                metric_name.value
                if hasattr(metric_name, 'value')
                else str(metric_name)
            )

            if "rouge" in metric_str.lower():
                rouge_metric = evaluate.load("rouge", seed=77)

                results = rouge_metric.compute(
                    predictions=predictions,
                    references=references,
                )

                final_metrics[metric_str] = float(results["rougeL"])

        print(f"Final metrics: {final_metrics}")
        return final_metrics
