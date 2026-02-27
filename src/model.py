"""
Model module for loading and configuring pre-trained models.
"""
from transformers import AutoModelForSequenceClassification, AutoConfig
from typing import Dict

def create_model(model_name: str = "distilbert-base-uncased", num_labels: int = 2):
    """
    Load pre-trained model for sequence classification.
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels (2 for binary sentiment)
    
    Returns:
        model: Configured model ready for fine-tuning
        id2label: Mapping from IDs to labels
        label2id: Mapping from labels to IDs
    """
    print(f"Loading model: {model_name}")
    
    # Create label mappings
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification"
    )
    
    print(f"Model loaded with {num_labels} labels")
    print(f"ID to Label mapping: {id2label}")
    
    return model, id2label, label2id

def get_model_info() -> Dict:
    """
    Return documentation about model selection.
    """
    return {
        "model_name": "distilbert-base-uncased",
        "architecture": "DistilBERT (distilled BERT)",
        "parameters": "66M (vs 110M for BERT-base)",
        "rationale": [
            "Efficiency: 60% faster inference, 40% smaller size",
            "Performance: Retains 97% of BERT's accuracy on GLUE",
            "Suitability: Excellent for binary text classification",
            "Deployment: Easier to deploy in resource-constrained environments",
            "Pre-training: Same corpus as BERT (BookCorpus + Wikipedia)"
        ],
        "task": "Binary Sentiment Classification (IMDB Reviews)",
        "dataset": "IMDB Movie Reviews (50K samples)"
    }