import argparse
from pickle import load
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np
from math import ceil


def load_seq(file):
    fp = open(file, 'r', encoding='utf8')
    text = fp.read()
    sequences = text.split('\n')
    return sequences


def model(cell, vocab_size, embed_dim, input_length):
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
    while True:
        # for i in range(steps):
        #     X_out = np.array([X[i]])
        #     y_out = np.array([to_categorical(y[i], num_classes=vocab_size)])
        #     yield X_out, y_out

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
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='../trained/tokenizer.pkl', help='tokenizer save path')
    parser.add_argument('-sp', action="store", dest="sequences_path", type=str, default='../trained/sequences.pkl', help='sequences save path')
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

    X, y = sequences[:, :-1], sequences[:, -1]
    input_length = len(X)
    seq_length = X.shape[1]
    samples_per_epoch = ceil(input_length/batch_size)

    model = model(lstm_cells, vocab_size, embed_dim, seq_length)

    model.fit_generator(data_generator(X, y, input_length, batch_size, vocab_size), samples_per_epoch=samples_per_epoch, nb_epoch=2)

    model.save('../trained/demo_model.h5')