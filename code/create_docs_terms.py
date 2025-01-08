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
    with open(segmentation_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract phrases within <phrase>...</phrase> tags
            phrases = re.findall(r'<phrase>([^<]+)</phrase>', line)
            for phrase in phrases:
                cleaned_phrase = clean_phrase(phrase)
                if cleaned_phrase:
                    phrase_counts[cleaned_phrase] = phrase_counts.get(cleaned_phrase, 0) + 1

    # Get quality phrases from AutoPhrase output
    quality_phrases = set()
    with open(autophrase_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                score, phrase = line.strip().split('\t')
                score = float(score)
                if score >= 0.5:  # Quality threshold
                    cleaned_phrase = clean_phrase(phrase)
                    if cleaned_phrase:
                        quality_phrases.add(cleaned_phrase)
            except:
                continue

    # Write terms.txt - include frequent phrases and quality phrases
    written_terms = set()
    with open(terms_output, 'w', encoding='utf-8') as f:
        # Write frequent phrases
        for phrase, count in phrase_counts.items():
            if count >= min_phrase_freq and phrase not in written_terms:
                f.write(phrase + '\n')
                written_terms.add(phrase)
        
        # Add quality phrases from AutoPhrase
        for phrase in quality_phrases:
            if phrase not in written_terms:
                f.write(phrase + '\n')
                written_terms.add(phrase)

    # Write docs.txt - tokenized documents with phrases marked
    with open(docs_output, 'w', encoding='utf-8') as f_out:
        with open(segmentation_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                # Replace phrases with underscore-connected versions
                def replace_phrase(match):
                    phrase = match.group(1)
                    return clean_phrase(phrase)
                
                # Process the line
                processed_line = re.sub(r'<phrase>([^<]+)</phrase>', replace_phrase, line)
                # Clean remaining text (non-phrases)
                processed_line = ' '.join(clean_phrase(token) for token in processed_line.split())
                
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
