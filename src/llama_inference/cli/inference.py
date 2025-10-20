from loguru import logger

import click
from transformers import AutoModelForCausalLM
from peft import PeftModel

from llama_inference.logging import setup_logging
from llama_inference.config import load_config
from llama_inference.hf import configure_hf, get_token

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
            config.sampling_params.max_tokens = max_tokens
            logger.info("Overriding max_tokens to: {}", max_tokens)
        if n_gpus is not None:
            config.gpus = n_gpus
            logger.info("Overriding number of GPUs to: {}", n_gpus)

        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")

    except Exception as e:
        logger.error("Failed to load config: ", e)
        raise

    configure_hf(config.base_model)
    get_token()

    logger.info("Loading base model from HF: {}", config.base_model)
    base = AutoModelForCausalLM.from_pretrained(config.base_model)

    logger.info("Merging adapter with PEFT: {}", config.adapter)
    model_with_adapter = PeftModel.from_pretrained(
        base,
        config.adapter,
        device_map="auto" if config.gpus > 0 else None,
    )

    merged = model_with_adapter.merge_and_unload()

