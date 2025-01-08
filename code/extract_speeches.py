import csv
import sys
import os

def extract_speeches(input_csv, output_file):
    with open(input_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for row in reader:
                # Extract the speech field and write to output
                speech = row['speech'].strip()
                if speech and not speech.startswith('The clerk'):  
                    # Only write non-empty speeches
                    # remove speech if it starts with The clerk
                    f_out.write(speech + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_speeches.py <input_csv> <output_file>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} does not exist")
        sys.exit(1)
        
    extract_speeches(input_csv, output_file)
    print(f"Successfully extracted speeches to {output_file}")
