"""
Task 8: Evaluate Model from HuggingFace Hub
Loads model from HuggingFace and evaluates on Goodreads dataset.
Works in Docker container.
"""

import os
import json
import argparse
import torch
import requests
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoConfig,
    pipeline
)
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Dataset URLs
GENRE_URL_DICT = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

# Configuration
YOUR_HF_USERNAME = "Jaishankar02"
MODEL_REPO_NAME = "distilbert-reviews-genres"
DEFAULT_GENRES = ['romance']
SAMPLES_PER_GENRE = 1000

def download_genre_data(genre, cache_dir="./data_cache"):
    """Download and extract specific genre data."""
    os.makedirs(cache_dir, exist_ok=True)
    
    url = GENRE_URL_DICT[genre]
    gz_filename = os.path.join(cache_dir, f"goodreads_reviews_{genre}.json.gz")
    json_filename = os.path.join(cache_dir, f"goodreads_reviews_{genre}.json")
    
    if not os.path.exists(gz_filename):
        print(f"Downloading {genre}...")
        response = requests.get(url, stream=True)
        with open(gz_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if not os.path.exists(json_filename):
        print(f"Extracting {genre}...")
        with gzip.open(gz_filename, 'rt', encoding='utf-8') as f_in:
            with open(json_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
    
    return json_filename

def load_genre_reviews(genre, max_samples=None, min_review_length=50, cache_dir="./data_cache"):
    """Load reviews for a specific genre."""
    json_file = download_genre_data(genre, cache_dir)
    
    print(f"Loading {genre} reviews...")
    data = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                review = json.loads(line.strip())
                review_text = review.get('review_text', '').strip()
                rating = review.get('rating', 0)
                
                if len(review_text) < min_review_length:
                    continue
                
                if rating in [1, 2]:
                    label = 0
                elif rating in [4, 5]:
                    label = 1
                else:
                    continue
                
                data.append({
                    'text': review_text,
                    'label': label,
                    'rating': rating,
                    'genre': genre
                })
            except:
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} reviews for {genre}")
    return df

def create_balanced_dataset(genres=None, samples_per_genre=1000, cache_dir="./data_cache"):
    """Create balanced dataset."""
    if genres is None:
        genres = DEFAULT_GENRES
    
    print(f"\nCreating dataset from {len(genres)} genres...")
    
    all_data = []
    for genre in genres:
        try:
            df = load_genre_reviews(genre, max_samples=samples_per_genre*3, cache_dir=cache_dir)
            
            pos = df[df['label'] == 1].sample(n=min(samples_per_genre//2, len(df[df['label']==1])), random_state=42)
            neg = df[df['label'] == 0].sample(n=min(samples_per_genre//2, len(df[df['label']==0])), random_state=42)
            
            balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
            all_data.append(balanced)
        except Exception as e:
            print(f"Error loading {genre}: {e}")
    
    full_df = pd.concat(all_data, ignore_index=True).sample(frac=1, random_state=42)
    
    # Split
    train_val, test = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])
    
    return DatasetDict({
        'train': Dataset.from_pandas(train.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val.reset_index(drop=True)),
        'test': Dataset.from_pandas(test.reset_index(drop=True))
    })

def prepare_datasets(tokenizer, dataset, max_length=512):
    """Tokenize datasets."""
    print("Tokenizing...")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized = {}
    for split in ["train", "validation", "test"]:
        # Keep only text and label
        ds = dataset[split].remove_columns([c for c in dataset[split].column_names if c not in ["text", "label"]])
        
        # Tokenize
        tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        # Rename label to labels
        if "label" in tok.column_names:
            tok = tok.rename_column("label", "labels")
        
        # Remove token_type_ids if present
        if "token_type_ids" in tok.column_names:
            tok = tok.remove_columns(["token_type_ids"])
        
        # Set format
        tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        tokenized[split] = tok
        print(f"  {split}: {len(tok)} samples")
    
    return DatasetDict(tokenized)

def evaluate_model(model, dataset_split):
    """Evaluate model."""
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
    
    # Collate function
    def collate_fn(examples):
        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples])
        }
    
    dataloader = DataLoader(dataset_split, batch_size=16, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.eval().to(device)
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    print(f"Evaluating {len(dataset_split)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs, labels=labels)
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    results = {
        "accuracy": float(metric_accuracy.compute(predictions=all_preds, references=all_labels)["accuracy"]),
        "f1_score": float(metric_f1.compute(predictions=all_preds, references=all_labels, average="weighted")["f1"]),
        "precision": float(metric_precision.compute(predictions=all_preds, references=all_labels, average="weighted")["precision"]),
        "recall": float(metric_recall.compute(predictions=all_preds, references=all_labels, average="weighted")["recall"]),
        "loss": float(total_loss / len(dataloader)),
        "num_samples": len(all_labels),
        "timestamp": datetime.now().isoformat()
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace Hub model on Goodreads dataset")
    parser.add_argument("--model-id", type=str, help="HuggingFace model ID (username/repo-name)")
    parser.add_argument("--genres", type=str, default=",".join(DEFAULT_GENRES), 
                       help="Comma-separated list of genres")
    parser.add_argument("--samples-per-genre", type=int, default=SAMPLES_PER_GENRE,
                       help="Number of samples per genre")
    
    args = parser.parse_args()
    
    model_id = args.model_id or f"{YOUR_HF_USERNAME}/{MODEL_REPO_NAME}"
    
    print(f"\n{'='*70}")
    print("Task 8: Evaluating Model from HuggingFace Hub")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"URL: https://huggingface.co/{model_id}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer (same as working Colab code)
    print("Loading model from HuggingFace Hub...")
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_id)
        print(f"Config vocab size: {config.vocab_size}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Fix vocab size if needed
        if config.vocab_size != tokenizer.vocab_size:
            print(f"Fixing vocab size: {config.vocab_size} -> {tokenizer.vocab_size}")
            config.vocab_size = tokenizer.vocab_size
        
        # Load model with fixed config
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            config=config, 
            ignore_mismatched_sizes=True
        )
        print(f"Model loaded with vocab size: {model.config.vocab_size}")
        print("✅ Model ready!\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset
    genres = args.genres.split(",")
    print(f"Loading dataset...")
    print(f"Genres: {genres}")
    print(f"Samples per genre: {args.samples_per_genre}\n")
    
    dataset = create_balanced_dataset(
        genres=genres,
        samples_per_genre=args.samples_per_genre,
        cache_dir="./data_cache"
    )
    
    # Tokenize
    print("Tokenizing...")
    tokenized_datasets = prepare_datasets(tokenizer, dataset)
    
    # Check compatibility
    sample = tokenized_datasets['test'][0]
    max_id = sample['input_ids'].max().item()
    print(f"\nMax input_id in data: {max_id}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    if max_id >= model.config.vocab_size:
        print("❌ Vocab mismatch! Using base tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenized_datasets = prepare_datasets(tokenizer, dataset)
    
    # Run evaluation
    print(f"\n{'='*70}")
    print("Running Evaluation")
    print(f"{'='*70}")
    
    results = evaluate_model(model, tokenized_datasets["test"])
    
    # Add metadata
    results["model_id"] = model_id
    results["hub_url"] = f"https://huggingface.co/{model_id}"
    results["dataset"] = {
        "source": "goodreads",
        "genres": genres,
        "samples_per_genre": args.samples_per_genre
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print(f"{'='*70}")
    
    # Save results
    output_file = "hub_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("Task 8 completed successfully!")

if __name__ == "__main__":
    main()