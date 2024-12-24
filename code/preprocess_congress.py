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

def process_autophrase_output(text):
    """Process AutoPhrase segmented text to extract phrases and clean text."""
    # AutoPhrase marks phrases with <phrase>...</phrase>
    phrases = re.findall(r'<phrase>([^<]+)</phrase>', text)
    
    # Clean text by removing XML tags and normalizing
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_text = clean_congress_text(clean_text)
    
    return clean_text, phrases

def process_speeches(input_file, output_dir, raw_dir):
    """Process congressional speeches and create required files."""
    ensure_dir(output_dir)
    ensure_dir(raw_dir)
    
    # Process segmented speeches and collect phrases
    print("Processing AutoPhrase segmented speeches...")
    all_phrases = set()
    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(os.path.join(raw_dir, 'docs.txt'), 'w') as fout:
        for line in f:
            try:
                # Process AutoPhrase output
                clean_text, phrases = process_autophrase_output(line)
                tokens = preprocess_text(clean_text)
                
                # Add any multi-word phrases that passed preprocessing
                all_phrases.update(phrases)
                
                if tokens:  # Only keep non-empty documents
                    fout.write(' '.join(tokens) + '\n')
                    
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue

    # Add taxonomy terms
    print("Creating raw/terms.txt with taxonomy and extracted terms...")
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
    
    # Write terms.txt with both taxonomy terms and extracted phrases
    with open(os.path.join(raw_dir, 'terms.txt'), 'w') as f:
        # Write taxonomy terms first
        for term in categories:
            f.write(f"{term}\n")
        # Then write extracted phrases
        for phrase in sorted(all_phrases):
            if not has_bad_syntax(phrase):  # Only include valid phrases
                f.write(f"{phrase}\n")

    # Create seed_taxo.txt if it doesn't exist
    seed_taxo_path = os.path.join(output_dir, 'seed_taxo.txt')
    if not os.path.exists(seed_taxo_path):
        print("Creating seed_taxo.txt")
        with open(seed_taxo_path, 'w') as f:
            f.write("*\t" + "\t".join(categories) + "\n")

    print("Done preprocessing speeches. Now run:")
    print("1. in home folder: ./code/jose -train data/congress/raw/docs.txt -word-emb data/congress/input/embeddings.txt -size 100")
    print("2. in home folder: python code/preprocess.py --data_dir data --dataset congress")

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
