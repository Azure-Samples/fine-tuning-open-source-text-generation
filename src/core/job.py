import argparse
import os

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature

import logging
from train import TextGenerationTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_tokenizer(
    train_dataset: Dataset,
    base_model_id: str,
    model_name: str,
    mlflow_model_folder: str,
    signature: ModelSignature,
    max_length: int,
    apply_lora: bool,
    apply_qlora: bool,
    token: str,
    prompt_template: str,
    use_conda: bool,
) -> None:
    """
    Sets up the tokenizer and model for training, tokenizes the training dataset, and initiates the training process.

    Args:
        train_dataset (Dataset): The dataset to be used for training.
        base_model_id (str): The identifier of the base model to be used.
        model_name (str): The name of the model to be saved.
        mlflow_model_folder (str): The folder where the model will be saved in MLflow.
        signature (ModelSignature): The signature of the model.
        max_length (int): The maximum length for tokenization.
        apply_lora (bool): A flag indicating whether to apply LoRA (Low-Rank Adaptation).
        apply_qlora (bool): A flag indicating whether to apply QLoRA (Quantized Low-Rank Adaptation).
        token (str): The authentication token for model loading.
        prompt_template (str): The template for the prompt to be used during training.

    Returns:
        None
    """

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        model_max_length=max_length,
        padding_side="left",
        add_eos_token=True,
        use_fast=False,
        use_auth_token=token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_pad_to_fixed_length(
        sample: dict, label_name: str = "prompt"
    ) -> dict:
        result = tokenizer(
            sample[label_name],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(
        tokenize_and_pad_to_fixed_length,
        cache_file_name="/tmp/tokenized_train_dataset.arrow",
    )

    # Drop the cache file after use
    os.remove("/tmp/tokenized_train_dataset.arrow")

    setup_model(
        tokenized_train_dataset,
        tokenizer,
        base_model_id,
        model_name,
        mlflow_model_folder,
        signature,
        max_length,
        apply_lora,
        apply_qlora,
        token,
        prompt_template,
        use_conda,
    )


def setup_model(
    tokenized_train_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    base_model_id: str,
    model_name: str,
    mlflow_model_folder: str,
    signature: ModelSignature,
    max_length: int,
    apply_lora: bool,
    apply_qlora: bool,
    token: str,
    prompt_template: str,
    use_conda: bool,
) -> None:
    """
    Sets up the model for training and initiates the training process.

    Args:
        tokenized_train_dataset (Dataset): The tokenized dataset to be used for training.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing the dataset.
        base_model_id (str): The identifier of the base model to be used.
        model_name (str): The name of the model to be saved.
        mlflow_model_folder (str): The folder where the model will be saved in MLflow.
        signature (ModelSignature): The signature of the model.
        max_length (int): The maximum length for tokenization.
        apply_lora (bool): A flag indicating whether to apply LoRA (Low-Rank Adaptation).
        apply_qlora (bool): A flag indicating whether to apply QLoRA (Quantized Low-Rank Adaptation).
        token (str): The authentication token for model loading.
        prompt_template (str): The template for generating prompts during training.

    Returns:
        None
    """

    # Loading the model with the required configuration
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map=None,
        use_auth_token=token,
    )

    try:
        assert all(len(x["input_ids"]) == max_length for x in tokenized_train_dataset)
    except AssertionError:
        logger.error("Not all input_ids are of the expected max_length.")

    if apply_qlora:
        # This parameter embodies the key technique of [QLoRA](https://github.com/artidoro/qlora) that significantly reduces memory usage during fine-tuning.
        # The following paragraph details the method and the implications of this configuration.
        # We rarely need to modify the `quantization_config` values ourselves.
        logger.info("Applying QLoRA.")
        quantization_config = BitsAndBytesConfig(
            # Load the model with 4-bit quantization
            load_in_4bit=True,
            # Use double quantization
            bnb_4bit_use_double_quant=True,
            # Use 4-bit Normal Float for storing the base model weights in GPU memory
            bnb_4bit_quant_type="nf4",
            # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=quantization_config, **model_kwargs
        )
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_model_id, **model_kwargs
        )

    trainer = TextGenerationTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=mlflow_model_folder,
        model_name=model_name,
        signature=signature,
        base_model_id=base_model_id,
        apply_lora=apply_lora,
        prompt_template=prompt_template,
        use_conda_yaml=use_conda,
    )

    trainer.train_model(tokenized_train_dataset)


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "garbage_collection_threshold:0.6,max_split_size_mb:128"
    )

    parser = argparse.ArgumentParser(
        description="Setup model and tokenizer for training"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the training dataset"
    )
    parser.add_argument("--base_model_id", type=str, required=True, help="Model id")

    parser.add_argument(
        "--prompt_template", type=str, required=True, help="Prompt template"
    )

    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    parser.add_argument(
        "--max_length", type=int, required=True, help="Max length for tokenization"
    )

    parser.add_argument(
        "--use_conda",
        type=bool,
        required=False,
        default=False,
        help="Use conda.yaml or not",
    )

    parser.add_argument(
        "--apply_lora",
        type=bool,
        required=False,
        default=True,
        help="Flag to apply lora or not",
    )

    parser.add_argument(
        "--apply_qlora",
        type=bool,
        required=False,
        default=True,
        help="Flag to apply qlora or not",
    )

    parser.add_argument(
        "--mlflow_model_folder",
        default="mlflow_model_folder_test",
        required=False,
        type=str,
        help="Output dir to save the finetune model as mlflow model",
    )

    parser.add_argument(
        "--token",
        type=str,
        required=False,
        default=None,
        help="Authentication token for model loading",
    )

    args = parser.parse_args()
    train_dataset: Dataset = load_from_disk(args.data)

    # MLflow infers schema from the provided sample input/output/params
    # Inference parameters can be saved with MLflow model as a part of [Model Signature](https://mlflow.org/docs/latest/model/signatures.html).
    # The signature defines model input and output format with additional parameters passed to the model prediction,
    # and you can let MLflow to infer it from some sample input using [mlflow.models.infer_signature](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.infer_signature) API.
    # If you pass the concrete value for parameters, MLflow treats them as default values and apply them at the inference if they are not provided by users.
    # For more details about the Model Signature, please refer to the [MLflow documentation](https://mlflow.org/docs/latest/model/signatures.html).
    sample = train_dataset[1]
    signature = infer_signature(
        model_input=sample["prompt"],
        model_output=sample["answer"],
        # Parameters are saved with default values if specified
        params={
            "max_new_tokens": args.max_length,
            "repetition_penalty": 1.15,
            "return_full_text": False,
        },
    )

    print(args.prompt_template)
    setup_tokenizer(
        train_dataset,
        args.base_model_id,
        args.model_name,
        args.mlflow_model_folder,
        signature,
        args.max_length,
        args.apply_lora,
        args.apply_qlora,
        args.token,
        args.prompt_template,
        args.use_conda,
    )
