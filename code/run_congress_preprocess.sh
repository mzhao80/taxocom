#!/bin/bash

# Directory setup
CONGRESS_DIR="../data/congress"
RAW_DIR="$CONGRESS_DIR/raw"
INPUT_DIR="$CONGRESS_DIR/input"
AUTOPHRASE_DIR="../AutoPhrase-master"
CODE_DIR="../code"

echo "Step 1: Cleaning crec2023.csv..."
# Use Python script to properly extract speeches from CSV
python extract_speeches.py $RAW_DIR/crec2023.csv $RAW_DIR/cleaned_speeches.txt

echo "Step 2: Running AutoPhrase for tokenization..."
# Copy cleaned speeches to AutoPhrase input and set environment variables
cp $RAW_DIR/cleaned_speeches.txt $AUTOPHRASE_DIR/data/EN/input.txt

# Run AutoPhrase with correct input file
cd $AUTOPHRASE_DIR
./auto_phrase.sh
./phrasal_segmentation.sh

# Get the segmented text
cp models/DBLP/segmentation.txt $RAW_DIR/segmented_speeches.txt

# Return to code directory
cd $CODE_DIR

echo "Step 3: Processing segmented speeches..."
# Run our preprocessing script to convert segmented speeches to docs.txt and terms.txt
python preprocess_congress.py \
    --input $RAW_DIR/segmented_speeches.txt \
    --output_dir $INPUT_DIR \
    --raw_dir $RAW_DIR

echo "Step 4: Extracting term integrity scores..."
# Extract term integrity scores from AutoPhrase output
python extract_term_integrity.py $AUTOPHRASE_DIR/models/DBLP/AutoPhrase.txt $INPUT_DIR/term_integrity.txt

echo "Step 5: create docs.txt and terms.txt"
# Process the segmented speeches to create docs.txt
python process_docs.py $RAW_DIR/segmented_speeches.txt $RAW_DIR/docs.txt

# Process AutoPhrase output to create terms.txt
python process_terms.py $AUTOPHRASE_DIR/models/DBLP/AutoPhrase.txt $RAW_DIR/terms.txt

echo "Step 6: Running standard preprocessing..."
# Run the standard preprocessing pipeline
bash run_preprocess.sh congress

echo "Done! Files are ready in $INPUT_DIR"
