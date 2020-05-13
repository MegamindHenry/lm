"""train keras model
"""
import lib_path

import argparse
from pickle import load
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping
import numpy as np
from math import ceil
from lm_lib.read import load_seq


def model(cell, vocab_size, embed_dim, input_length):
    """create model
    
    Args:
        cell (int): lstm unit
        vocab_size (int): vocab size
        embed_dim (int): embed dim
        input_length (int): input length
    
    Returns:
        keras.model: model
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=input_length))
    model.add(LSTM(cell, return_sequences=True))
    model.add(LSTM(cell))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

    model.summary()

    return model


def data_generator(X, y, input_length, batch_size, vocab_size):
    """data generator for 
    
    Args:
        X (nparray): X
        y (nparray): y
        input_length (int): input length
        batch_size (int): batch size
        vocab_size (int): batch size
    
    Yields:
        X, y: X, y
    """
    while True:
        iter = 0
        while iter < input_length:
            end = iter+batch_size
            if end > input_length:
                yield X[iter:], to_categorical(y[iter:], num_classes=vocab_size)
                iter = end
                break
            yield X[iter:end], to_categorical(y[iter:end], num_classes=vocab_size)
            iter = end


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='trained/tokenizer.pkl', help='tokenizer save path')
    parser.add_argument('-sp', action="store", dest="sequences_path", type=str, default='trained/sequences.pkl', help='sequences save path')
    parser.add_argument('-ed', action="store", dest="embed_dim", type=int,
        default=300, help='embedding layer dim')
    parser.add_argument('-lc', action="store", dest="lstm_cells", type=int,
        default=128, help='lstm cells')
    parser.add_argument('-bs', action="store", dest="batch_size", type=int,
        default=256, help='batch_size')

    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
    sequences_path = args.sequences_path
    embed_dim = args.embed_dim
    lstm_cells = args.lstm_cells
    batch_size = args.batch_size

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))
    print('Load sequences...')
    sequences = load(open(sequences_path, 'rb'))

    # vocab_size = len(tokenizer.word_index) + 1
    vocab_size = min((tokenizer.num_words, len(tokenizer.word_index) + 1))

    X, y = sequences[:, 1:], sequences[:, 0]
    total_samples = len(X)
    seq_length = X.shape[1]

    train_dev_split = total_samples//2
    train_samples = train_dev_split
    dev_samples = total_samples - train_samples

    samples_per_epoch = ceil(train_samples/batch_size)
    validation_steps = ceil(dev_samples/batch_size)

    X_train, X_dev = X[:train_dev_split], X[train_dev_split:]
    y_train, y_dev = y[:train_dev_split], y[train_dev_split:]

    model = model(lstm_cells, vocab_size, embed_dim, seq_length)

    train_generator = data_generator(X_train, y_train, train_samples, batch_size, vocab_size)
    dev_generator = data_generator(X_dev, y_dev, dev_samples, batch_size, vocab_size)

    es = EarlyStopping(patience=20, restore_best_weights=True)

    model.fit_generator(generator=train_generator, validation_data=dev_generator, samples_per_epoch=samples_per_epoch, validation_steps=validation_steps, nb_epoch=100, callbacks=[es])

    model.save('trained/model.h5')