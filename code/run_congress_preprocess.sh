#!/bin/bash

# Directory setup
CONGRESS_DIR="../data/congress"
RAW_DIR="$CONGRESS_DIR/raw"
INPUT_DIR="$CONGRESS_DIR/input"
AUTOPHRASE_DIR="../AutoPhrase-master"

# Create necessary directories
mkdir -p $RAW_DIR
mkdir -p $INPUT_DIR

echo "Step 1: Cleaning crec2023.csv..."
# Extract the speech field from CSV
tail -n +2 $RAW_DIR/crec2023.csv | cut -d',' -f2 > $RAW_DIR/cleaned_speeches.txt

echo "Step 2: Running AutoPhrase for tokenization..."
# Copy cleaned speeches to AutoPhrase input and set environment variables
cp $RAW_DIR/cleaned_speeches.txt $AUTOPHRASE_DIR/data/input.txt

# Run AutoPhrase with correct input file
cd $AUTOPHRASE_DIR
./auto_phrase.sh
./phrasal_segmentation.sh

# Get the segmented text
cp models/DBLP/segmentation.txt $RAW_DIR/segmented_speeches.txt

# Return to code directory
cd - > /dev/null

echo "Step 3: Processing segmented speeches..."
# Run our preprocessing script to convert segmented speeches to docs.txt and terms.txt
python preprocess_congress.py \
    --input $RAW_DIR/segmented_speeches.txt \
    --output_dir $INPUT_DIR \
    --raw_dir $RAW_DIR

echo "Step 4: Running standard preprocessing..."
# Run the standard preprocessing pipeline
bash run_preprocess.sh congress

echo "Done! Files are ready in $INPUT_DIR"
