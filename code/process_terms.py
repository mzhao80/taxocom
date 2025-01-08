import sys
import re

def clean_phrase(phrase):
    """Clean and normalize a phrase"""
    # Remove special characters except underscore and hyphen
    cleaned = re.sub(r'[^\w\s_-]', ' ', phrase)
    # Replace spaces with underscores for multi-word phrases
    if ' ' in cleaned:
        cleaned = '_'.join(cleaned.split())
    return cleaned.lower()

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_terms.py autophrase_file output_file")
        sys.exit(1)

    autophrase_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(autophrase_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    score, phrase = line.strip().split('\t')
                    cleaned_phrase = clean_phrase(phrase)
                    if cleaned_phrase:
                        f_out.write(cleaned_phrase + '\n')
                except:
                    continue

if __name__ == '__main__':
    main()
