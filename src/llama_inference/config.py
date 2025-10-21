from pathlib import Path
import yaml

from pydantic import BaseModel, Field, field_validator

class InferenceConfig(BaseModel):
    """Configuration for inference"""
    model: str = Field(..., description="Model name or path")
    gpus: int = Field(1, description="Number of GPUs to use")
    test_data: str = Field(..., description="Path to the input data file")
    output_file: str = Field(..., description="Path to save the output results")
    max_tokens: int = Field(256, description="Maximum tokens to generate per prompt")

    @field_validator("output_file")
    def ensure_output_dir_exists(cls, v: str):
        """Ensure the parent directory for output_file exists."""
        output_path = Path(v)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return v

def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)