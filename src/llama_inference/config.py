from pathlib import Path
import yaml

from pydantic import BaseModel, Field, ValidationError, field_validator

class InferenceConfig(BaseModel):
    """Configuration for inference"""
    model_name: str = Field(..., description="Name of the model to use")
    gpus: int = Field(1, description="Number of GPUs to use")
    test_data: str = Field(..., description="Path to the input data file")
    output_file: str = Field(..., description="Path to save the output results")

    @field_validator("output_path")
    def create_output_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)