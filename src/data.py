"""
Data module for loading and preprocessing the Goodreads book reviews dataset.
Source: https://mengtingwan.github.io/data/goodreads.html
"""

import os
import json
import gzip
import pandas as pd
import requests
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# Dataset URLs from the assignment notebook
GENRE_URL_DICT = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def download_genre_data(genre: str, cache_dir: str = "./data_cache") -> str:
    """Download and extract specific genre data from Goodreads dataset."""
    if genre not in GENRE_URL_DICT:
        raise ValueError(f"Genre must be one of {list(GENRE_URL_DICT.keys())}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    url = GENRE_URL_DICT[genre]
    gz_filename = os.path.join(cache_dir, f"goodreads_reviews_{genre}.json.gz")
    json_filename = os.path.join(cache_dir, f"goodreads_reviews_{genre}.json")
    
    if not os.path.exists(gz_filename):
        print(f"Downloading {genre} dataset from UCSD...")
        print(f"URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(gz_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {gz_filename}")
    
    if not os.path.exists(json_filename):
        print(f"Extracting {gz_filename}...")
        with gzip.open(gz_filename, 'rt', encoding='utf-8') as f_in:
            with open(json_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        print(f"Extracted to {json_filename}")
    
    return json_filename

def load_genre_reviews(genre: str, max_samples: Optional[int] = None, 
                      min_review_length: int = 50, cache_dir: str = "./data_cache") -> pd.DataFrame:
    """Load reviews for a specific genre and convert to classification format."""
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
                    label = 0  # Negative
                elif rating in [4, 5]:
                    label = 1  # Positive
                else:
                    continue
                
                data.append({
                    'text': review_text,
                    'label': label,
                    'rating': rating,
                    'genre': genre,
                    'book_id': review.get('book_id'),
                    'review_id': review.get('review_id')
                })
                
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} reviews for {genre}")
    print(f"  Positive (4-5★): {len(df[df['label']==1])}")
    print(f"  Negative (1-2★): {len(df[df['label']==0])}")
    
    return df

def create_balanced_dataset(genres: List[str] = None, samples_per_genre: int = 5000,
                           test_size: float = 0.2, val_size: float = 0.1,
                           random_state: int = 42, cache_dir: str = "./data_cache") -> DatasetDict:
    """Create a balanced multi-genre dataset for classification."""
    if genres is None:
        genres = list(GENRE_URL_DICT.keys())
    
    print(f"\nCreating balanced dataset from {len(genres)} genres...")
    print(f"Target: {samples_per_genre} samples per genre")
    
    all_data = []
    
    for genre in genres:
        try:
            df = load_genre_reviews(genre, max_samples=samples_per_genre*3, cache_dir=cache_dir)
            
            pos_samples = df[df['label'] == 1].sample(n=min(samples_per_genre//2, len(df[df['label']==1])), 
                                                      random_state=random_state)
            neg_samples = df[df['label'] == 0].sample(n=min(samples_per_genre//2, len(df[df['label']==0])), 
                                                      random_state=random_state)
            
            balanced = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=random_state)
            all_data.append(balanced)
            
        except Exception as e:
            print(f"Warning: Could not load {genre}: {e}")
            continue
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nTotal combined dataset: {len(full_df)} samples")
    print(f"  Positive: {len(full_df[full_df['label']==1])}")
    print(f"  Negative: {len(full_df[full_df['label']==0])}")
    
    train_val, test = train_test_split(
        full_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=full_df['label']
    )
    
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val['label']
    )
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val.reset_index(drop=True)),
        'test': Dataset.from_pandas(test.reset_index(drop=True))
    })
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Validation: {len(dataset['validation'])}")
    print(f"  Test: {len(dataset['test'])}")
    
    return dataset

def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Load tokenizer for the specified model."""
    print(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)

def prepare_datasets(tokenizer, dataset, max_length: int = 512):
    """
    Apply tokenization to all dataset splits.
    """
    print("Tokenizing datasets...")
    
    tokenized_datasets = {}
    for split in ["train", "validation", "test"]:
        print(f"Processing {split} split...")
        
        # Keep only necessary columns before tokenization
        columns_to_remove = [col for col in dataset[split].column_names if col not in ["text", "label"]]
        if columns_to_remove:
            dataset_split = dataset[split].remove_columns(columns_to_remove)
        else:
            dataset_split = dataset[split]
        
        def tokenize_function(examples):
            # Tokenize - DistilBERT doesn't use token_type_ids
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            return tokenized
        
        # Apply tokenization
        tokenized = dataset_split.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"]
        )
        
        # Rename label to labels
        if "label" in tokenized.column_names:
            tokenized = tokenized.rename_column("label", "labels")
        
        # Remove token_type_ids if present (DistilBERT doesn't use them)
        if "token_type_ids" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["token_type_ids"])
        
        # Ensure only required columns exist
        required_columns = ["input_ids", "attention_mask", "labels"]
        existing_columns = tokenized.column_names
        
        # Remove any extra columns
        columns_to_drop = [col for col in existing_columns if col not in required_columns]
        if columns_to_drop:
            tokenized = tokenized.remove_columns(columns_to_drop)
        
        # Set format to torch tensors
        tokenized.set_format(type="torch", columns=required_columns)
        
        tokenized_datasets[split] = tokenized
        print(f"  Final columns: {tokenized.column_names}")
    
    return DatasetDict(tokenized_datasets)

def get_dataset_info():
    """Return documentation about dataset selection."""
    return {
        "source": "Goodreads Book Reviews (UCSD)",
        "url": "https://mengtingwan.github.io/data/goodreads.html",
        "genres_available": list(GENRE_URL_DICT.keys()),
        "task": "Binary Sentiment Classification (Book Reviews)",
        "label_mapping": {
            "0": "Negative (1-2 stars)",
            "1": "Positive (4-5 stars)"
        }
    }