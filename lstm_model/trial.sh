#!/bin/bash

echo "Preparing sequence for training..."
prepare_sequences="python3 prepare_sequences.py"
$prepare_sequences

echo "Making tokenization..."
tokenization="python3 tokenization.py"
$tokenization

echo "Training..."
train="train.py"
$train

echo "Producing probability table..."
production="python3 production.py"
$production