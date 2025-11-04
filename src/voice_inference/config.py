"""Configuration management for VOICE inference package."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class InferenceConfig(BaseModel):
    """Configuration for inference."""

    model: str = Field(..., description="Model name or path")
    test_data: str = Field(..., description="Path to the input data file")
    split: str = Field("test", description="Dataset split to use for inference")

    gpus: int = Field(1, description="Number of GPUs to use")
    quantization: Optional[str] = Field(
        None, description="Quantization method to use (must be '4bit' if provided)"
    )
    max_tokens: int = Field(256, description="Maximum tokens to generate per prompt")

    output_file: str = Field(..., description="Path to save the output results")

    @field_validator("output_file")
    def ensure_output_dir_exists(cls, v: str) -> str:
        """
        Ensure the parent directory for output_file exists.

        :param v: The value of output_file.
        :return: The value of output_file.
        """
        output_path = Path(v)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("quantization")
    def validate_quantization(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate the quantization field.

        :param v: The value of quantization.
        :raises ValueError: If the quantization field is not 4bit when provided.
        :return: The validated value.
        """
        if v is not None and v != "4bit":
            raise ValueError("Quantization must be '4bit' if provided")
        return v


def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)
