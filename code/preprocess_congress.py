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

# US state names for additional stopwords
US_STATES = {
    'alabama', 'alaska', 'arizona', 'arkansas', 'california',
    'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
    'hawaii', 'idaho', 'illinois', 'indiana', 'iowa',
    'kansas', 'kentucky', 'louisiana', 'maine', 'maryland',
    'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri',
    'montana', 'nebraska', 'nevada', 'hampshire', 'jersey',
    'mexico', 'york', 'carolina', 'dakota', 'ohio',
    'oklahoma', 'oregon', 'pennsylvania', 'rhode', 'island',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia',
    'washington', 'wisconsin', 'wyoming'
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

def get_significant_bigrams(sentences, min_freq=5):
    """Find statistically significant bigrams using PMI."""
    # Flatten sentences into words
    words = [word for sent in sentences for word in sent]
    
    # Find bigram collocations
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    
    # Apply frequency filter
    finder.apply_freq_filter(min_freq)
    
    # Score bigrams by PMI
    scored = finder.score_ngrams(bigram_measures.pmi)
    
    # Convert to dictionary for faster lookup
    return {f"{w1}_{w2}": score for ((w1, w2), score) in scored}

def preprocess_text(text, bigrams=None):
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
    stop_words.update(US_STATES)
    
    # Filter tokens based on stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # If bigrams dictionary is provided, look for multi-word terms
    if bigrams is not None:
        # Get bigrams from tokens
        token_bigrams = list(ngrams(tokens, 2))
        
        # Replace qualifying bigrams with combined form
        final_tokens = []
        skip_next = False
        for i in range(len(tokens)):
            if skip_next:
                skip_next = False
                continue
                
            if i < len(tokens) - 1:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                if bigram in bigrams:
                    final_tokens.append(bigram)
                    skip_next = True
                else:
                    final_tokens.append(tokens[i])
            else:
                final_tokens.append(tokens[i])
        
        tokens = final_tokens
    
    # Apply syntax rules
    tokens = [t for t in tokens if not has_bad_syntax(t)]
    
    return tokens

def get_bert_embeddings(terms, model_name='bert-base-uncased'):
    """Get BERT embeddings for terms."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = {}
    batch_size = 32
    
    print(f"Generating BERT embeddings using {model_name}...")
    for i in tqdm(range(0, len(terms), batch_size)):
        batch_terms = terms[i:i + batch_size]
        
        # For multi-word terms, replace underscore with space for better tokenization
        batch_terms = [t.replace('_', ' ') for t in batch_terms]
        
        # Tokenize terms
        encoded = tokenizer(batch_terms, padding=True, truncation=True, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # Use [CLS] token embedding as term representation
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Store embeddings with original terms (with underscores)
        for term, embedding in zip(terms[i:i + batch_size], batch_embeddings):
            embeddings[term] = embedding
    
    return embeddings

def process_speeches(input_file, output_dir, model_name):
    """Process congressional speeches and create required files."""
    ensure_dir(output_dir)
    
    # First pass: collect all sentences for bigram analysis
    print("First pass: collecting sentences for bigram analysis...")
    all_sentences = []
    with open(input_file, 'r', encoding='iso-8859-1') as f:
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
                # Basic preprocessing to get sentences
                text = clean_congress_text(text)
                tokens = word_tokenize(text.lower())
                all_sentences.append(tokens)
                
            except Exception as e:
                continue
    
    # Find significant bigrams
    print("Finding significant bigrams...")
    bigrams = get_significant_bigrams(all_sentences)
    
    # Second pass: process speeches with bigram information
    print("Second pass: processing speeches with bigrams...")
    documents = []  # List of processed documents
    doc_ids = []   # List of document IDs
    term_freq = defaultdict(int)  # Term frequency counter
    
    with open(input_file, 'r', encoding='iso-8859-1') as f:
        # Skip header if present
        header = f.readline()
        if not header.startswith('speech_id|speech'):
            f.seek(0)
            
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print(f"Processed {i} speeches...")
            
            try:
                parts = line.strip().split('|')
                if len(parts) != 2:
                    continue
                
                speech_id, text = parts
                
                # Process text with bigram information
                tokens = preprocess_text(text, bigrams)
                if tokens:  # Only keep non-empty documents
                    documents.append(' '.join(tokens))
                    doc_ids.append(speech_id)
                    
                    # Update term frequencies
                    for token in tokens:
                        term_freq[token] += 1
            except Exception as e:
                print(f"Error processing line {i}: {str(e)}")
                continue
    
    # Get unique terms that appear at least 5 times
    vocab = [term for term, freq in term_freq.items() if freq >= 5]
    print(f"Vocabulary size: {len(vocab)}")
    
    # Get BERT embeddings for terms
    embeddings_dict = get_bert_embeddings(vocab, model_name)
    
    # Write files
    print("Writing output files...")
    
    # 1. terms.txt - vocabulary
    with open(os.path.join(output_dir, 'terms.txt'), 'w') as f:
        for term in vocab:
            f.write(f"{term}\n")
    
    # 2. docs.txt - processed documents
    with open(os.path.join(output_dir, 'docs.txt'), 'w') as f:
        for doc in documents:
            f.write(f"{doc}\n")
    
    # 3. doc_ids.txt - document IDs
    with open(os.path.join(output_dir, 'doc_ids.txt'), 'w') as f:
        for doc_id in doc_ids:
            f.write(f"{doc_id}\n")
    
    # 4. embeddings.txt - BERT embeddings
    with open(os.path.join(output_dir, 'embeddings.txt'), 'w') as f:
        for term in vocab:
            embedding_str = ' '.join([f"{x:.6f}" for x in embeddings_dict[term]])
            f.write(f"{term} {embedding_str}\n")
    
    # 5. term_freq.txt - term frequencies
    with open(os.path.join(output_dir, 'term_freq.txt'), 'w') as f:
        for term in vocab:
            f.write(f"{term} {term_freq[term]}\n")
    
    # 6. index.txt - term to ID mapping
    with open(os.path.join(output_dir, 'index.txt'), 'w') as f:
        for i, term in enumerate(vocab):
            f.write(f"{i}\t{term}\n")
    
    # 7. seed_taxo.txt - initial taxonomy with congressional categories
    with open(os.path.join(output_dir, 'seed_taxo.txt'), 'w') as f:
        f.write("Root\n")
        categories = [
            "Agriculture_and_Food",
            "Animals",
            "Armed_Forces_and_National_Security",
            "Arts_Culture_Religion",
            "Civil_Rights_and_Liberties_Minority_Issues",
            "Crime_and_Law_Enforcement",
            "Economics_and_Public_Finance",
            "Education",
            "Emergency_Management",
            "Energy",
            "Environmental_Protection",
            "Families",
            "Finance_and_Financial_Sector",
            "Foreign_Trade_and_International_Finance",
            "Government_Operations_and_Politics",
            "Health",
            "Housing_and_Community_Development",
            "Immigration",
            "International_Affairs",
            "Labor_and_Employment",
            "Law",
            "Native_Americans",
            "Private_Legislation",
            "Public_Lands_and_Natural_Resources",
            "Science_Technology_Communications",
            "Social_Sciences_and_History",
            "Social_Welfare",
            "Sports_and_Recreation",
            "Taxation",
            "Transportation_and_Public_Works",
            "Water_Resources_Development"
        ]
        for category in categories:
            f.write(f"\t{category}\n")

    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Process congressional speeches into TaxoCom format')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to speeches_114.txt file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for processed files')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                      help='BERT model to use for embeddings')
    
    args = parser.parse_args()
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    process_speeches(args.input, args.output, args.model)

if __name__ == "__main__":
    main()