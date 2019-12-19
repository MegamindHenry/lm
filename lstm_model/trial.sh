#!/bin/bash

#make trained
#download nltk punkt
#import nltk
#nltk.download('punkt')

echo "Preparing sequence for training..."
prepare_sequences="python3 prepare_sequences.py"
$prepare_sequences

echo "Making tokenization..."
tokenization="python3 tokenization.py"
$tokenization

echo "Training..."
train="python3 train.py"
$train

echo "Producing probability table..."
production="python3 production.py"
$production