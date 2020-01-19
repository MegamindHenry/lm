import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from lm_lib.read import load_seq, read_gentleinput_list
from pickle import load
import numpy as np
from keras.models import load_model
from tqdm import tqdm


def eval(model, X, gold, gi):
    total = 0
    correct = 0

    predict_raw = model.predict(X)

    predicts = np.argmax(predict_raw, axis=-1)

    for i in range(len(X)):
        if predicts[i] == gold[i]:
            correct += 1
        total += 1
        
    accuracy = correct/total
    return '{}: {}'.format(gi, accuracy)


if __name__ == '__main__':
    seq_path = '../trained/gentleinput_seq/'
    tokenizer_path = '../trained/tokenizer.pkl'
    model_path = '../trained/demo_model.h5'
    gentle_inputs_path = '../trained/gentleinput_list.txt'

    print('Read gentleinput list ...')
    gentle_inputs = read_gentleinput_list(gentle_inputs_path)
    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))
    print('Load model...')
    model = load_model(model_path)


    results = []
    for gi in tqdm(gentle_inputs):
        # print('Load sequences [{}] ...'.format(gi))
        lines = load_seq(seq_path + gi + '_5.txt')
        sequences = tokenizer.texts_to_sequences(lines)
        sequences = np.array(sequences)

        X, y = sequences[:, :-1], sequences[:, -1]

        results.append(eval(model, X, y, gi))

    print(results)