"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset
from core_utils.project.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
import sys
from pathlib import Path

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

    preprocessor = RawDataPreprocessor(raw_data=importer._raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor._preprocessed_data)

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=50,
        batch_size=2,
        device="cpu"
    )

    result = pipeline.infer_dataset()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
