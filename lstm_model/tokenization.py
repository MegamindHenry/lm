"""tokenize sequences and transfer to np array
"""
import lib_path

import argparse
from keras.preprocessing.text import Tokenizer
from pickle import dump
import numpy as np
from lm_lib.read import load_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-wn', action="store", dest="word_num", type=int, default=20000, help='only this number of word will be keept')
    parser.add_argument('-f', action="store", dest="file", type=str, default='../trained/sequences_5.txt', help='set save path')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='../trained/', help='set tokenizer save path')

    args = parser.parse_args()

    file = args.file
    word_num = args.word_num
    tokenizer_path = args.tokenizer_path

    lines = load_seq(file)

    print('Preparing tokenizer...')
    tokenizer = Tokenizer(word_num, filters='', oov_token='UNK')
    tokenizer.fit_on_texts(lines)

    vocab_size = len(tokenizer.word_index) + 1
    print('vocab_size: {}'.format(vocab_size))

    print('Transforming to sequences...')
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = np.array(sequences)
    print('sequences shape: {}'.format(sequences.shape))

    print('Saving files...')
    dump(tokenizer, open(tokenizer_path + 'tokenizer.pkl', 'wb'))
    dump(sequences, open(tokenizer_path + 'sequences.pkl', 'wb'))