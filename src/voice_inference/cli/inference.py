"""CLI for running inference with vLLM and specified configuration."""

import json
from pathlib import Path

import click
from datasets import load_dataset
from loguru import logger
from vllm import LLM, SamplingParams

from voice_inference.config import (
    is_wandb_artifact,
    load_config,
    load_config_from_wandb_artifact,
)
from voice_inference.hf import configure_hf, get_token
from voice_inference.logging import setup_logging

ROOT_DIR = Path.cwd()
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--max-tokens", type=int, help="Override max tokens for generation")
@click.option("--n-gpus", type=int, help="Number of GPUs to use for inference")
def main(
    config_path: str,
    log_level: str,
    log_file: str,
    max_tokens: int,
    n_gpus: int
) -> None:
    """Run inference with vLLM and specified configuration."""
    setup_logging(log_level, log_file)

    # Detect local vs W&B artifact
    if is_wandb_artifact(config_path):
        logger.info("Detected W&B artifact config: {}", config_path)
        config_file = load_config_from_wandb_artifact(config_path)
        logger.info("Downloaded config to {}", str(config_path))
    else:
        config_file = Path(config_path).expanduser()

    try:
        logger.info("Loading config from {}", config_file)
        config = load_config(str(config_file))

        # Override sampling parameters if provided via CLI
        if max_tokens is not None:
            config.max_tokens = max_tokens
            logger.info("Overriding max_tokens to: {}", max_tokens)
        if n_gpus is not None:
            config.gpus = n_gpus
            logger.info("Overriding number of GPUs to: {}", n_gpus)

        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")

    except Exception as e:
        logger.error("Failed to load config: {}", e)
        raise

    configure_hf(config.model)
    get_token()

    dataset = load_dataset(config.test_data, split=config.split)

    prompts = [
        next(msg["content"] for msg in example["messages"] if msg["role"] == "user")
        for example in dataset
    ]

    quantization = None
    if config.quantization:
        quantization = "bitsandbytes"

    logger.info("Instantiating model {}", config.model)
    llm = LLM(
        model=config.model,
        quantization=quantization,
        tensor_parallel_size=config.gpus,
        dtype="bfloat16",
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
    )

    logger.info("Generating results for {} prompts", len(prompts))
    outputs = []
    for example in dataset:
        # Use only system and user messages
        messages = [m for m in example["messages"] if m["role"] != "assistant"]

        # Run chat inference
        response = llm.chat(messages, sampling_params)
        text = response[0].outputs[0].text.strip()

        ref = next(
            (m["content"] for m in example["messages"] if m["role"] == "assistant"),
            None,
        )

        outputs.append(
            {
                "messages": messages,
                "generated_response": text,
                "reference_response": ref,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / config.output_file

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    logger.success("Results successfully written to {}", output_path)
