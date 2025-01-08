import sys

def extract_term_integrity(autophrase_file, output_file):
    """Extract term integrity scores from AutoPhrase output"""
    with open(autophrase_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # AutoPhrase output format is: score<tab>phrase
                try:
                    score, phrase = line.strip().split('\t')
                    # Convert score to float and validate
                    score = float(score)
                    if 0 <= score <= 1:
                        # Write in format: phrase<tab>score
                        f_out.write(f'{phrase}\t{score}\n')
                except:
                    continue

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_term_integrity.py <autophrase_file> <output_file>")
        sys.exit(1)
    
    autophrase_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_term_integrity(autophrase_file, output_file)
