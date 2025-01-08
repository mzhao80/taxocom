import re
import sys

def process_line(line):
    """Process a line from segmented speeches to clean format"""
    # Function to handle phrase replacements
    def replace_phrase(match):
        phrase = match.group(1)
        # Always convert multi-word phrases to use underscores
        if ' ' in phrase:
            return '_'.join(phrase.split())
        return phrase

    # Remove the phrase tags and handle multi-word phrases
    # Pattern matches <phrase_Q=X.XXX>text</phrase>
    pattern = r'<phrase_Q=[0-9.]+>([^<]+)</phrase>'
    processed = re.sub(pattern, replace_phrase, line)
    return processed.lower()

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_docs.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                processed_line = process_line(line.strip())
                if processed_line:
                    f_out.write(processed_line + '\n')

if __name__ == '__main__':
    main()
