"""Merge a QLoRA adapter into its base model in place and push to the Hugging Face Hub."""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download, HfApi, HfFolder
from loguru import logger
import os

from llama_inference.logging import setup_logging
from llama_inference.hf import configure_hf, get_token

BASE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
ADAPTER_REPO = "AccelerateScience/LLama-3.1-70B-Instruct-QLoRA-Bush"
MERGED_REPO = "AccelerateScience/LLama-3.1-70B-Instruct-QLoRA-Bush-Merged"

if __name__ == "__main__":
    setup_logging()
    configure_hf(BASE_MODEL)
    get_token()

    logger.info("Downloading adapter from HF: {}", ADAPTER_REPO)
    adapter_path = snapshot_download(repo_id=ADAPTER_REPO)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # Ensure pad token matches fine-tune config
    if tokenizer.pad_token is None or tokenizer.pad_token != "<PAD>":
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading base model from cache: {}", BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="bfloat16",
        quantization_config=bnb_config,
        device_map={"": 0},     # no auto as want to avoid meta tensors
    )

    # Resize embeddings to match tokenizer
    new_vocab_size = len(tokenizer)
    current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    if current_vocab_size != new_vocab_size:
        logger.info(
            "Resizing token embeddings from {} to {} (with mean_resizing=False)",
            current_vocab_size,
            new_vocab_size,
        )
        # Skip the covariance computation which causes memory issues
        base_model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
        base_model.config.vocab_size = new_vocab_size

    logger.info("Loading LoRA adapter and merging into base weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Get HF local cache path for the base model
    hf_home = os.getenv("HF_HOME", "models")
    model_dir = os.path.join(hf_home, BASE_MODEL)

    logger.info("Overwriting model weights directly at {}", model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(model_dir)

    logger.info("Pushing merged model to Hugging Face Hub...")
    api = HfApi()
    api.create_repo(MERGED_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=MERGED_REPO,
        repo_type="model",
    )

    logger.success("Successfully pushed merged model to {}", MERGED_REPO)