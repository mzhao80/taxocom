import argparse
import re

def clean_phrase(phrase):
    """Clean and normalize a phrase"""
    # Remove special characters except underscore
    cleaned = re.sub(r'[^\w\s_-]', ' ', phrase)
    # Replace spaces with underscores for multi-word phrases
    cleaned = '_'.join(cleaned.split())
    return cleaned.lower()

def process_autophrase_output(segmentation_file, autophrase_file, docs_output, terms_output, min_phrase_freq=5):
    # First pass: collect phrases and their frequencies
    phrase_counts = {}
    
    # Regular expression to match AutoPhrase quality scores and phrases
    phrase_pattern = r'<phrase_q_([0-9.]+)_([^>]+)>([^<]+)</phrase>'
    
    with open(segmentation_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Find all phrases with their quality scores
            phrases = re.finditer(phrase_pattern, line)
            for match in phrases:
                score = float(match.group(1))
                phrase_text = match.group(3).strip()
                if score >= 0.5:  # Only collect high-quality phrases
                    cleaned_phrase = clean_phrase(phrase_text)
                    if cleaned_phrase:
                        phrase_counts[cleaned_phrase] = phrase_counts.get(cleaned_phrase, 0) + 1

    # Get additional quality phrases from AutoPhrase.txt
    with open(autophrase_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                score, phrase = line.strip().split('\t')
                score = float(score)
                if score >= 0.5:  # Quality threshold
                    cleaned_phrase = clean_phrase(phrase)
                    if cleaned_phrase:
                        phrase_counts[cleaned_phrase] = phrase_counts.get(cleaned_phrase, 0) + 1
            except:
                continue

    # Write terms.txt - include frequent quality phrases
    with open(terms_output, 'w', encoding='utf-8') as f:
        for phrase, count in phrase_counts.items():
            if count >= min_phrase_freq:
                f.write(phrase + '\n')

    # Write docs.txt - tokenized documents with phrases
    with open(docs_output, 'w', encoding='utf-8') as f_out:
        with open(segmentation_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                # Replace quality phrases with their cleaned versions
                def replace_phrase(match):
                    phrase_text = match.group(3).strip()
                    return clean_phrase(phrase_text)
                
                # Process the line
                processed_line = re.sub(phrase_pattern, replace_phrase, line)
                # Clean remaining text (non-phrases)
                processed_line = ' '.join(token.strip() for token in processed_line.split())
                processed_line = clean_phrase(processed_line)
                
                if processed_line:
                    f_out.write(processed_line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create docs.txt and terms.txt from AutoPhrase output')
    parser.add_argument('--segmentation', required=True, help='Path to AutoPhrase segmentation output')
    parser.add_argument('--autophrase', required=True, help='Path to AutoPhrase.txt')
    parser.add_argument('--docs_output', required=True, help='Path to output docs.txt')
    parser.add_argument('--terms_output', required=True, help='Path to output terms.txt')
    parser.add_argument('--min_freq', type=int, default=5, help='Minimum frequency for phrases')
    
    args = parser.parse_args()
    
    process_autophrase_output(
        args.segmentation,
        args.autophrase,
        args.docs_output,
        args.terms_output,
        args.min_freq
    )
