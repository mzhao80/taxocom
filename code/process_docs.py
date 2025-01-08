import re
import sys
from collections import defaultdict

def process_line(line):
    # Function to handle phrase replacements
    phrases = []  # Collect phrases for terms.txt
    
    def replace_phrase(match):
        phrase = match.group(1)
        # Store the phrase for terms.txt
        if ' ' in phrase:
            phrases.append('_'.join(phrase.lower().split()))
        else:
            phrases.append(phrase.lower())
        # If phrase contains spaces, replace them with underscores
        if ' ' in phrase:
            return '_'.join(phrase.split())
        return phrase

    # Remove the phrase tags and handle multi-word phrases
    # Pattern matches <phrase_Q=X.XXX>text</phrase>
    pattern = r'<phrase_Q=[0-9.]+>([^<]+)</phrase>'
    processed = re.sub(pattern, replace_phrase, line)
    return processed.lower(), phrases

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_docs.py input_file docs_output terms_output")
        sys.exit(1)

    input_file = sys.argv[1]
    docs_output = sys.argv[2]
    terms_output = sys.argv[3]

    # Collect all unique phrases and their frequencies
    phrase_counts = defaultdict(int)
    
    # First pass: process docs and collect phrases
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(docs_output, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                processed_line, phrases = process_line(line.strip())
                if processed_line:
                    f_out.write(processed_line + '\n')
                # Count phrase occurrences
                for phrase in phrases:
                    phrase_counts[phrase] += 1

    # Write terms.txt with phrases that appear at least 5 times
    with open(terms_output, 'w', encoding='utf-8') as f_out:
        for phrase, count in sorted(phrase_counts.items()):
            if count >= 5:  # Minimum frequency threshold
                f_out.write(phrase + '\n')

if __name__ == '__main__':
    main()
