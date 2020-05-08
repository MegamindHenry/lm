"""train keras model
"""
import lib_path

import argparse
from pickle import load
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping
import numpy as np
from math import ceil
from lm_lib.read import load_seq
import pandas
from tqdm import tqdm


def read(file_path):
    df = pandas.read_csv(file_path, index_col=0)
    df['prev_prob'] = 0.0
    df['next_prob'] = 0.0
    return df


def write(df):
    df.to_csv('output/homophone_in_sequence_predicted.csv')


def predict(model, df, tokenizer):
    prev_words = df['prev_word'].to_list()
    print(prev_words)
    lines = [f'{w}' for w in prev_words]
    lines = [str(w) for w in prev_words]
    lines = [w for w in prev_words]
    for i, l in enumerate(lines):
        if type(l) == type(''):
            continue
        print(l, i)
        print(type(l))
    sequences = tokenizer.texts_to_sequences(lines)
    print(sequences)


def predict_by_line(model, df, tokenizer):
    for i, row in tqdm(df.iterrows()):
        try:
            prev_word = row['prev_word']
            prev_word = tokenizer.texts_to_sequences([prev_word])
            next_word = row['next_word']
            next_word = tokenizer.texts_to_sequences([next_word])
            homophone = row['homophone']
            homophone = tokenizer.texts_to_sequences([homophone])

            prev_prediction = model.predict(prev_word)[0][homophone[0]]
            next_prediction = model.predict(next_word)[0][homophone[0]]
            
            # print(df[i]['prev_prob'])
            row['prev_prob'] = prev_prediction
            row['next_prob'] = next_prediction
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='trained/tokenizer.pkl', help='tokenizer save path')
    # parser.add_argument('-sp', action="store", dest="sequences_path", type=str, default='trained/sequences.pkl', help='sequences save path')
    # parser.add_argument('-ed', action="store", dest="embed_dim", type=int,
    #     default=300, help='embedding layer dim')
    # parser.add_argument('-lc', action="store", dest="lstm_cells", type=int,
    #     default=128, help='lstm cells')
    # parser.add_argument('-bs', action="store", dest="batch_size", type=int,
    #     default=256, help='batch_size')

    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
    # sequences_path = args.sequences_path
    # embed_dim = args.embed_dim
    # lstm_cells = args.lstm_cells
    # batch_size = args.batch_size

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))

    model = load_model('trained/model.h5')

    # vocab_size = len(tokenizer.word_index) + 1
    vocab_size = min((tokenizer.num_words, len(tokenizer.word_index) + 1))

    df = read('output/homophone_in_sequence.csv')
    predict_by_line(model, df, tokenizer)
    write(df)