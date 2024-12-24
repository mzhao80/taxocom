#!/usr/bin/env python3

import os
import re
import argparse
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.util import ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Congressional stopwords from the paper
CONGRESS_STOPWORDS = {
    'absent', 'adjourn', 'ask', 'can', 'chairman',
    'committee', 'con', 'democrat', 'etc', 'gentleladies',
    'gentlelady', 'gentleman', 'gentlemen', 'gentlewoman', 'gentlewomen',
    'hereabout', 'hereafter', 'hereat', 'hereby', 'herein',
    'hereinafter', 'hereinbefore', 'hereinto', 'hereof', 'hereon',
    'hereto', 'heretofore', 'hereunder', 'hereunto', 'hereupon',
    'herewith', 'month', 'mr', 'mrs', 'nay',
    'none', 'now', 'part', 'per', 'pro',
    'republican', 'say', 'senator', 'shall', 'sir',
    'speak', 'speaker', 'tell', 'thank', 'thereabout',
    'thereafter', 'thereagainst', 'thereat', 'therebefore', 'therebeforn',
    'thereby', 'therefor', 'therefore', 'therefrom', 'therein',
    'thereinafter', 'thereof', 'thereon', 'thereto', 'theretofore',
    'thereunder', 'thereunto', 'thereupon', 'therewith', 'therewithal',
    'today', 'whereabouts', 'whereafter', 'whereas', 'whereat',
    'whereby', 'wherefore', 'wherefrom', 'wherein', 'whereinto',
    'whereof', 'whereon', 'whereto', 'whereunder', 'whereupon',
    'wherever', 'wherewith', 'wherewithal', 'will', 'yea',
    'yes', 'yield'
}

def has_bad_syntax(word):
    """Check if a word has bad syntax according to paper rules."""
    # Contains numbers, symbols, or punctuation (except underscores)
    if not all(c.isalpha() or c == '_' for c in word):
        return True
    
    # For multi-word terms, check each part
    parts = word.split('_')
    
    for part in parts:
        # Fewer than three characters for each part
        if len(part) < 3:
            return True
        
        # One-letter word
        if len(part) == 1:
            return True
        
        # Word beginning with first three letters of a month
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december']
        if any(part.lower().startswith(month[:3]) for month in months):
            return True
    
    return False

def clean_congress_text(text):
    """Clean congressional speech text."""
    # Remove special characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'(?<=\w)\.(?=\w)', ' ', text)  # Split words joined by periods
    text = re.sub(r'(?<=\w)(?=[A-Z])', ' ', text)  # Split camelCase
    text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical expressions
    return text.strip()

def preprocess_text(text):
    """Clean and tokenize text."""
    # First apply congress-specific cleaning
    text = clean_congress_text(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(CONGRESS_STOPWORDS)
    
    # Filter tokens based on stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # Apply syntax rules
    tokens = [t for t in tokens if not has_bad_syntax(t)]
    
    return tokens

def process_speeches(input_file, output_dir, raw_dir):
    """Process congressional speeches and create required files."""
    ensure_dir(output_dir)
    ensure_dir(raw_dir)
    
    # First pass: collect all sentences and create raw/docs.txt
    print("First pass: processing speeches...")
    with open(input_file, 'r', encoding='iso-8859-1') as f, open(os.path.join(raw_dir, 'docs.txt'), 'w') as fout:
        # Skip header if present
        header = f.readline()
        if not header.startswith('speech_id|speech'):
            f.seek(0)
            
        for line in f:
            try:
                parts = line.strip().split('|')
                if len(parts) != 2:
                    continue
                
                _, text = parts
                # Process text
                tokens = preprocess_text(text)
                if tokens:  # Only keep non-empty documents
                    fout.write(' '.join(tokens) + '\n')
                    
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue

    # Create raw/terms.txt with taxonomy terms
    print("Creating raw/terms.txt with taxonomy terms...")
    categories = [
        "agriculture_and_food",
        "animals",
        "armed_forces_and_national_security",
        "arts_culture_religion",
        "civil_rights_and_liberties_minority_issues",
        "commerce",
        "crime_and_law_enforcement",
        "economics_and_public_finance",
        "education",
        "emergency_management",
        "energy",
        "environmental_protection",
        "foreign_trade_and_international_finance",
        "geographic_areas_entities_committees",
        "government_operations_and_politics",
        "health",
        "housing_and_community_development",
        "immigration",
        "international_affairs",
        "labor_and_employment",
        "law",
        "native_americans",
        "private_legislation",
        "public_lands_and_natural_resources",
        "science_technology_communications",
        "social_sciences_and_history",
        "social_welfare",
        "sports_and_recreation",
        "taxation",
        "transportation_and_public_works",
        "water_resources_development"
    ]
    
    # Write categories to raw/terms.txt
    with open(os.path.join(raw_dir, 'terms.txt'), 'w') as f:
        for term in categories:
            f.write(f"{term}\n")

    # Create seed_taxo.txt
    print("Creating seed_taxo.txt")
    with open(os.path.join(output_dir, 'seed_taxo.txt'), 'w') as f:
        f.write("*\t" + "\t".join(categories) + "\n")

    print("Done preprocessing speeches. Now run:")
    print("1. jose -train raw/docs.txt -word-emb input/embeddings.txt -size 100")
    print("2. python preprocess.py to generate the final files")

def main():
    parser = argparse.ArgumentParser(description='Preprocess congressional speeches')
    parser.add_argument('--input', type=str, required=True, help='Input speeches file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--raw_dir', type=str, required=True, help='Raw directory')
    args = parser.parse_args()
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    process_speeches(args.input, args.output_dir, args.raw_dir)

if __name__ == "__main__":
    main()
