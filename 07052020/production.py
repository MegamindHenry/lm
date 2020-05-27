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
    return df


def write(df):
    # df.to_csv('output/play_predicted.csv')
    df.to_csv('output/homophone_in_sequence_predicted.csv')


def step_generator(total, step):
    n = 0
    while (n+1)*step < total:
        yield n*step, (n+1)*step
        n += 1
    yield n*step, total


def predict_step(prev_model, next_model, df, tokenizer, start, end,\
    replace_start_end=True):

    df_batch = df[start:end].fillna(
        {'prev_word':'<s>', 'next_word':'</s>'})
    prev_words = df_batch['prev_word'].to_list()
    next_words = df_batch['next_word'].to_list()
    homophones = df_batch['homophone'].to_list()
    prev_words_sequences = np.array(tokenizer.texts_to_sequences(prev_words))
    next_words_sequences = np.array(tokenizer.texts_to_sequences(next_words))
    homophones_sequences = np.array(tokenizer.texts_to_sequences(homophones)).flatten()

    prev_prediction_raw = prev_model.predict(prev_words_sequences)
    next_prediction_raw = next_model.predict(next_words_sequences)

    length = len(homophones_sequences)

    prev_prediction = prev_prediction_raw[np.arange(length), homophones_sequences]
    next_prediction = next_prediction_raw[np.arange(length), homophones_sequences]
    
    df.loc[start:end-1, 'prev_prob'] = prev_prediction
    df.loc[start:end-1, 'next_prob'] = next_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str,
        default='trained/tokenizer.pkl', help='tokenizer save path')
    parser.add_argument('-s', action='store', dest='step', type=int,
        default=16384, help='how much steps per batch when predicting')

    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
    step = args.step

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))

    prev_model = load_model('trained/prev_model.h5')
    next_model = load_model('trained/next_model.h5')

    vocab_size = min((tokenizer.num_words, len(tokenizer.word_index) + 1))

    # df = read('output/play.csv')
    df = read('output/homophone_in_sequence.csv')
    total = len(df)
    pbar = tqdm(total=total)
    for s, e in step_generator(total, step):
        predict_step(prev_model, next_model, df, tokenizer, s, e)
        pbar.update(step)
    pbar.close()
    write(df)