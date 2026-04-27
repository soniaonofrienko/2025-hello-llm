"""
Starter for demonstration of laboratory work.
"""
# pylint: disable=protected-access
import sys
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings \
    import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter()
    importer.obtain()

    assert importer._raw_data is not None, "Raw data is None"

    preprocessor = RawDataPreprocessor(raw_data=importer._raw_data)
    preprocessor.transform()

    assert preprocessor._preprocessed_data is not None, \
        "Preprocessed data is None"

    dataset = TaskDataset(preprocessor._preprocessed_data)

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=50,
        batch_size=2,
        device="cpu"
    )

    model_props = pipeline.analyze_model()
    print("Model properties:", model_props)

    sample_prediction = pipeline.infer_sample(dataset[0])
    print("Sample prediction:", sample_prediction)

    result = pipeline.infer_dataset()
    print(result.head())
    assert result is not None, "Demo does not work correctly"

    output_path = current_path / "predictions.csv"
    result.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    metrics_to_check = [Metrics(m) for m in settings.parameters.metrics]

    evaluator = TaskEvaluator(data_path=output_path, metrics=metrics_to_check)
    scores = evaluator.run()

    print("Final metrics")
    if scores:
        for name, value in scores.items():
            print(f"{name}: {value:.4f}")
    else:
        print("No metrics calculated")


if __name__ == "__main__":
    main()
