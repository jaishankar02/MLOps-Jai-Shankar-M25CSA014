"""
Training script for fine-tuning DistilBERT on Goodreads Book Reviews Sentiment Analysis.
Dataset: https://mengtingwan.github.io/data/goodreads.html
"""

import os
import sys
import json
import torch
from datetime import datetime
from transformers import (
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from evaluate import load as load_metric
import numpy as np

from data import (
    create_balanced_dataset, 
    get_tokenizer, 
    prepare_datasets, 
    get_dataset_info
)
from model import create_model, get_model_info
from utils import setup_logging, save_training_config

# Configuration
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Dataset Configuration
GENRES = ['romance', 'fantasy_paranormal', 'mystery_thriller_crime', 'young_adult']
SAMPLES_PER_GENRE = 5000

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: Accuracy and F1 Score.
    """
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

def main():
    print("=" * 70)
    print("ML/DL Ops Assignment 3: Model Training")
    print("Dataset: Goodreads Book Reviews (UCSD)")
    print("=" * 70)
    
    # Setup logging
    setup_logging(LOGGING_DIR)
    
    # Display dataset information
    dataset_info = get_dataset_info()
    print("\n📚 Dataset Information:")
    print(f"Source: {dataset_info['source']}")
    print(f"Task: {dataset_info['task']}")
    print(f"Genres used: {', '.join(GENRES)}")
    print("Rationale for dataset selection:")
    for reason in dataset_info['rationale']:
        print(f"  - {reason}")
    
    # Display model selection rationale
    model_info = get_model_info()
    print(f"\n🤖 Model Information:")
    print(f"Model: {model_info['model_name']}")
    print("Rationale:")
    for reason in model_info['rationale']:
        print(f"  - {reason}")
    
    # Load data
    print(f"\n[1/5] Loading Goodreads dataset...")
    print(f"Selected genres: {GENRES}")
    print(f"Target samples per genre: {SAMPLES_PER_GENRE}")
    
    # Use multi-genre balanced dataset
    dataset = create_balanced_dataset(
        genres=GENRES,
        samples_per_genre=SAMPLES_PER_GENRE,
        test_size=0.2,
        val_size=0.1,
        cache_dir="./data_cache"
    )
    
    print(f"\nDataset splits loaded:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    # Show label distribution
    for split in ['train', 'validation', 'test']:
        labels = dataset[split]['label']
        pos = sum(labels)
        neg = len(labels) - pos
        print(f"  {split} - Positive: {pos}, Negative: {neg}")
    
    # Load tokenizer and model
    print(f"\n[2/5] Loading tokenizer and model ({MODEL_NAME})...")
    tokenizer = get_tokenizer(MODEL_NAME)
    model, id2label, label2id = create_model(MODEL_NAME, num_labels=2)
    
    # Prepare datasets
    print("\n[3/5] Preprocessing datasets...")
    tokenized_datasets = prepare_datasets(tokenizer, dataset, MAX_LENGTH)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    print("\n[4/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=["tensorboard"],
        push_to_hub=False,
        hub_model_id=None,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n[5/5] Starting training...")
    print(f"Training for up to {NUM_EPOCHS} epochs with early stopping...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    
    train_result = trainer.train()
    
    # Log training metrics
    print("\n✅ Training completed!")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    print(f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # Save model locally
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training config
    config = {
        "model_name": MODEL_NAME,
        "task": "book-review-sentiment-analysis",
        "dataset": {
            "source": "goodreads",
            "genres": GENRES,
            "samples_per_genre": SAMPLES_PER_GENRE,
            "total_samples": sum(len(dataset[split]) for split in dataset.keys())
        },
        "training_args": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS
        },
        "metrics": {
            "final_loss": float(train_result.training_loss),
            "runtime_seconds": float(train_result.metrics['train_runtime'])
        }
    }
    save_training_config(config, os.path.join(OUTPUT_DIR, "training_config.json"))
    
    # Final evaluation on test set
    print("\n📊 Running final evaluation on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  Loss:      {test_results['eval_loss']:.4f}")
    
    # Save test results
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training pipeline completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("Next steps:")
    print("  1. Run evaluate_hub.py for detailed evaluation")
    print("  2. Run push_to_hub.py to upload to HuggingFace")
    print("=" * 70)

if __name__ == "__main__":
    main()