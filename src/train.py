from datetime import datetime
from typing import Dict
from datasets import Dataset
import os
from dataclasses import asdict

import transformers
from transformers import TrainingArguments, AutoTokenizer, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import mlflow


import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import TrainingArgsConfig, ModelLoraConfig

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerationTrainer:

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        output_dir: str,
        model_name: str,
        signature: str,
        base_model_id: str,
        prompt_template: str,
        apply_lora: bool = True,
        config: TrainingArgsConfig = TrainingArgsConfig(),
        lora_config: ModelLoraConfig = ModelLoraConfig(),
        use_conda_yaml: bool = False,
    ):
        """
        Initializes the TextToSQLTrainer class.

        Args:
            model (transformers.PreTrainedModel): The pre-trained model to be fine-tuned.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
            output_dir (str): Directory to save the trained model.
            model_name (str): Name of the model.
            signature (str): Signature for the model.
            base_model_id (str): ID of the base model.
            prompt_template (str): Template for generating prompts.
            apply_lora (bool, optional): Whether to apply LoRA for training. Defaults to True.
            config (TrainingArgsConfig, optional): Configuration for training arguments. Defaults to TrainingArgsConfig().
            lora_config (ModelLoraConfig, optional): Configuration for LoRA. Defaults to ModelLoraConfig().
            use_conda_yaml (bool, optional): Whether to use conda yaml. Defaults to False.
        """
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.model_original = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.model_name = model_name
        self.signature = signature
        self.base_model_id = base_model_id
        self.apply_lora = apply_lora
        self.config = asdict(config)
        self.lora_config = asdict(lora_config)
        self.prompt_template = prompt_template
        self._use_conda_yaml = use_conda_yaml

    def train_model(self, tokenized_train_dataset: Dataset) -> None:
        """
        Trains the model using the provided tokenized training dataset.

        Args:
            tokenized_train_dataset (Dataset): The tokenized training dataset.
        """
        self.model_original.gradient_checkpointing_enable()

        if self.apply_lora:
            logger.info("Using LoRA for training.\n")
            self.model = prepare_model_for_kbit_training(self.model_original)
            peft_config = LoraConfig(**self.lora_config)
        else:
            self.model = self.model_original

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Now create a batch of examples using DataCollatorForSeq2Seq.
        # It's more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.base_model_id
        )

        training_args = TrainingArguments(
            report_to="none",
            run_name=f"{self.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
            output_dir=self.output_dir,
            **self.config,
        )

        train_size = int(0.98 * len(tokenized_train_dataset))
        eval_size = len(tokenized_train_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            tokenized_train_dataset, [train_size, eval_size]
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=transformers.DataCollatorForLanguageModeling(
            #     self.tokenizer, mlm=False
            # ),
            args=training_args,
            compute_metrics=self.compute_metrics_text_generation,
            data_collator=data_collator,
        )

        tokenizer_no_pad = AutoTokenizer.from_pretrained(
            self.base_model_id, add_bos_token=True
        )

        mlflow.start_run()
        start_time = datetime.now()
        logger.info(f"Training started at {start_time}")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")

        trainer_stats = trainer.train()

        end_time = datetime.now()
        logger.info(f"Training finished at {end_time}")
        logger.info(f"Total training time: {end_time - start_time}")

        self.log_params_from_dict(self.config)
        self.log_params_from_dict(asdict(peft_config))

        metrics = trainer_stats.metrics
        mlflow.log_metrics(metrics)

        if self.apply_lora:
            # Define the directory to save artifacts
            adaptor_dir = f"/tmp/model-output-peft"
            os.makedirs(adaptor_dir, exist_ok=True)

            trainer.save_model(adaptor_dir)

            # Adaptor directory on your local filesystem
            merged_model = PeftModel.from_pretrained(self.model_original, adaptor_dir)
            merged_model = merged_model.merge_and_unload()

            logger.info(self.model_original)
            logger.info(merged_model)

            # The user prompt itself is free text, but you can harness the input by applying a ‘template’.
            # MLflow Transformer flavor supports saving a prompt template with the model, and apply it automatically before the prediction. This also allows you to hide the system prompt from model clients. To save the prompt template, we have to define a single string that contains {prompt} variable, and pass it to the prompt_template argument of mlflow.transformers.log_model API.
            # Refer to Saving Prompt Templates with Transformer Pipelines for more detailed usage of this feature.

            if self._use_conda_yaml:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": merged_model,
                        "tokenizer": tokenizer_no_pad,
                    },
                    prompt_template=self.prompt_template,
                    signature=self.signature,
                    artifact_path="model",
                    conda_env="conda.yaml",
                )
            else:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": merged_model,
                        "tokenizer": tokenizer_no_pad,
                    },
                    prompt_template=self.prompt_template,
                    signature=self.signature,
                    artifact_path="model",
                )
            
            # Saving the adapter
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                },
                artifact_path="adapter_peft",
            )

        else:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                    "tokenizer": tokenizer_no_pad,
                },
                prompt_template=self.prompt_template,
                signature=self.signature,
                artifact_path="model",
            )

        mlflow.end_run()

    def log_params_from_dict(
        self, config: Dict[str, any], parent_key: str = ""
    ) -> None:
        """
        Given a dictionary of parameters, logs non-dictionary values to MLflow.
        Ignores nested dictionaries.

        Args:
            config (Dict[str, any]): The dictionary of parameters to log.
            parent_key (str): Used to prefix keys (for nested logging).
        """
        max_length = 199  # max length of a parameter value to log
        for key, value in config.items():
            if isinstance(value, dict):
                continue
            elif isinstance(value, list):
                full_key = f"{parent_key}.{key}" if parent_key else key
                value = (
                    value if len(str(value)) < max_length else str(value)[:max_length]
                )
                mlflow.log_param(full_key, ",".join(map(str, value)))
            else:
                full_key = f"{parent_key}.{key}" if parent_key else key
                value = (
                    value if len(str(value)) < max_length else str(value)[:max_length]
                )
                mlflow.log_param(full_key, value)

    @staticmethod
    def compute_metrics_classification(pred):
        """
        Computes classification metrics including accuracy, precision, recall, and F1-score.

        Args:
            pred: The predictions from the model.

        Returns:
            Dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        f1 = f1_score(labels, preds, average="weighted")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def compute_metrics_text_generation(self, eval_pred):
        """
        Computes text generation metrics using the specified evaluation metric (e.g., ROUGE).

        Args:
            eval_pred: The evaluation predictions from the model.

        Returns:
            Dict: A dictionary containing the computed metrics.
        """
        return_loss = True
        import numpy as np

        # Load the metric appropriate for text generation
        metric = evaluate.load("rouge")  # You can change to "rouge" or others as needed
        logits, labels = eval_pred

        # Decode logits to generated text
        predictions = np.argmax(logits, axis=-1)  # Get the predicted token IDs

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Convert token IDs to text (assuming a tokenizer is available)
        reference_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        generated_texts = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Calculate and return the evaluation metric
        result = metric.compute(predictions=generated_texts, references=reference_texts)

        # Log metrics to MLflow
        for key, value in result.items():
            mlflow.log_metric(key, value)

        return result
