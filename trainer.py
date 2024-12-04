# trainer.py
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import IntervalStrategy
import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np
import os
from pathlib import Path
from datetime import datetime

from config import ChatbotConfig
from data_processor import DataProcessor
from model_manager import ModelManager
from utils import setup_logging

class PerformanceCallback(TrainerCallback):
    """Callback for monitoring and optimizing performance"""
    def __init__(self, logger):
        self.logger = logger
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.logger.info("Starting training optimization...")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                current_loss = logs['loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                self.logger.info(f"Current Loss: {current_loss:.4f}, Best Loss: {self.best_loss:.4f}")
            
class OptimizedChatbotTrainer:
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        
        # Disable tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize metrics tracking
        self.metrics_history = []
        
    def _prepare_model(self):
        """Prepare and optimize model for training"""
        try:
            model = self.model_manager.model
            
            # Basic optimizations
            if torch.cuda.is_available():
                model = model.cuda()
                if ChatbotConfig.FP16:
                    model = model.half()
            
            # Enable gradient checkpointing if needed
            if ChatbotConfig.GRADIENT_CHECKPOINTING:
                model.gradient_checkpointing_enable()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error preparing model: {str(e)}")
            raise
            
    def train(
        self,
        num_epochs: int = ChatbotConfig.EPOCHS,
        batch_size: int = ChatbotConfig.TRAIN_BATCH_SIZE,
        learning_rate: float = ChatbotConfig.LEARNING_RATE,
    ):
        """Train the model"""
        try:
            # Prepare model
            model = self._prepare_model()
            
            # Prepare datasets
            self.logger.info("Loading and preparing datasets...")
            datasets = self.data_processor.load_datasets()
            train_dataset, eval_dataset = self.data_processor.prepare_training_data(
                datasets,
                validation_split=0.1
            )
            
            # Prepare output directory
            output_dir = ChatbotConfig.MODEL_DIR / "checkpoints"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_dir=str(ChatbotConfig.LOG_DIR),
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                # Disable features that might cause issues
                fp16=ChatbotConfig.FP16,
                gradient_checkpointing=ChatbotConfig.GRADIENT_CHECKPOINTING,
                dataloader_num_workers=0,  # Disable multi-processing
                remove_unused_columns=False,
                local_rank=-1,  # Disable distributed training
                report_to="none"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[PerformanceCallback(self.logger)],
                # Use simple data collator
                data_collator=lambda examples: {
                    'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
                    'attention_mask': torch.stack([ex['attention_mask'] for ex in examples]),
                    'labels': torch.stack([ex['labels'] for ex in examples])
                }
            )
            
            # Train
            self.logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            self.logger.info("Saving model...")
            final_model_path = ChatbotConfig.MODEL_DIR / f"model_v{ChatbotConfig.MODEL_VERSION}"
            trainer.save_model(str(final_model_path))
            self.model_manager.tokenizer.save_pretrained(str(final_model_path))
            
            # Save training metrics
            self._save_training_metrics(train_result, final_model_path)
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
            
    def _save_training_metrics(self, train_result, save_path: Path):
        """Save training metrics and configuration"""
        metrics = {
            "train_loss": float(train_result.training_loss),
            "train_time_seconds": float(train_result.metrics.get("train_runtime", 0)),
            "samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0)),
            "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0,
            "training_finished": datetime.now().isoformat(),
            "config": {
                "epochs": ChatbotConfig.EPOCHS,
                "batch_size": ChatbotConfig.TRAIN_BATCH_SIZE,
                "learning_rate": ChatbotConfig.LEARNING_RATE,
                "fp16": ChatbotConfig.FP16,
                "gradient_checkpointing": ChatbotConfig.GRADIENT_CHECKPOINTING
            }
        }
        
        # Save metrics
        metrics_path = save_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
            
        # Log metrics
        self.logger.info("\nTraining Complete! Final Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, dict):
                self.logger.info(f"{key}: {value}")
                
    def get_training_status(self):
        """Get current training status"""
        if not self.metrics_history:
            return {
                "status": "Not started",
                "metrics": None
            }
        
        latest_metrics = self.metrics_history[-1]
        return {
            "status": "Completed",
            "metrics": latest_metrics
        }