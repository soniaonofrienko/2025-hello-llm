"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")
    
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    # preprocessor = RawDataPreprocessor(importer.raw_data)
    # analysis = preprocessor.analyze()
    
    assert importer._raw_data is not None, "Raw data is None"

    preprocessor = RawDataPreprocessor(raw_data=importer._raw_data)
    preprocessor.transform()
    
    assert preprocessor._data is not None, "Preprocessed data is None"

    dataset = TaskDataset(preprocessor._data)
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    model_props = pipeline.analyze_model()
    print("Model properties:", model_props)

    sample_prediction = pipeline.infer_sample(dataset[0])
    print("Sample prediction:", sample_prediction)

    result = pipeline.infer_dataset()
    print(result.head())
    assert result is not None, "Demo does not work correctly"

    output_path = current_path / "dist" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding='utf-8-sig')

    metrics_to_check = [Metrics(m) for m in settings.parameters.metrics]

    evaluator = TaskEvaluator(data_path=output_path, metrics=metrics_to_check)
    scores = evaluator.run()

    print("Final metrics")
    if scores:
        for name, value in scores.items():
            print(f"{name}: {value:.4f}")
    else:
        print("No metrics calculated")
    # result = None
    # assert result is not None, "Fine-tuning does not work correctly"

if __name__ == "__main__":
    main()
