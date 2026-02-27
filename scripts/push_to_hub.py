"""
Script to push trained model to Hugging Face Hub.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, create_repo
import os
import sys

def push_model_to_hub(
    local_model_path: str = "./results",
    repo_id: str = None,  # Format: "username/model-name"
    private: bool = False
):
    """
    Push trained model to Hugging Face Hub.
    
    Args:
        local_model_path: Path to local model directory
        repo_id: Target repository ID (e.g., "johndoe/sentiment-analysis-bert")
        private: Whether to create private repository
    """
    if repo_id is None:
        raise ValueError("Please provide repo_id (e.g., 'yourusername/sentiment-analysis-bert')")
    
    print(f"Pushing model to Hugging Face Hub: {repo_id}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    # Create model card
    model_card = f"""
---
language: en
tags:
- text-classification
- sentiment-analysis
- distilbert
- imdb
license: apache-2.0
datasets:
- imdb
metrics:
- accuracy
- f1
---

# Sentiment Analysis Model

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) 
for binary sentiment classification on the IMDB dataset.

## Model Details

- **Base Model:** DistilBERT base uncased (66M parameters)
- **Task:** Binary Sentiment Classification (Positive/Negative)
- **Dataset:** IMDB Movie Reviews (50K samples)
- **Training Epochs:** 3 with early stopping
- **Final Metrics:** See evaluation results

## Usage

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="{repo_id}")
result = classifier("This movie was absolutely fantastic!")