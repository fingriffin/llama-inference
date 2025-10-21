from loguru import logger
from pathlib import Path
import click
import json

from huggingface_hub import snapshot_download
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llama_inference.logging import setup_logging
from llama_inference.config import load_config
from llama_inference.hf import configure_hf, get_token

ROOT_DIR = Path.cwd()
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--max-tokens", type=int, help="Override max tokens for generation")
@click.option("--n-gpus", type=int, help="Number of GPUs to use for inference")
def main(config_path, log_level, log_file, max_tokens, n_gpus):
    """Run inference with vLLM and specified configuration."""
    setup_logging(log_level, log_file)

    try:
        logger.info("Loading config from {}", config_path)
        config = load_config(config_path)

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

    configure_hf(config.base_model)
    get_token()

    dataset = load_dataset(config.test_data, split="test")

    prompts = [
        next(msg["content"] for msg in example["messages"] if msg["role"] == "user")
        for example in dataset
    ]


    logger.info("Downloading adaptor from HF: {}", config.adaptor)
    adaptor_path = snapshot_download(repo_id=config.adaptor)

    logger.info("Instantiate base model {}", config.base_model)
    llm = LLM(model=config.base_model, enable_lora=True)

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
    )

    logger.info("Generating results for {} prompts", len(config.prompts))
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("bush_adaptor", 1, adaptor_path)
    )

    data = [
        {"prompt": o.prompt, "text": o.outputs[0].text}
        for o in outputs
    ]

    output_path = OUTPUT_DIR / "results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Results successfully written to {}", output_path)