import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from lm_lib.read import load_seq, read_gentleinput_list
from lm_lib.math import argmax_top_k
from pickle import load
import numpy as np
from keras.models import load_model
from tqdm import tqdm


def eval(model, X, gold, gi, k, tokenizer):
    total = 0
    correct = 0

    predict_raw = model.predict(X)
    predicts = argmax_top_k(predict_raw, k)

    for i in range(len(X)):
        # candidates = [tokenizer.index_word[j] for j in predicts[i]]
        # gold_label = tokenizer.index_word[gold[i]]
        # print(candidates)
        # print(gold_label)

        if gold[i] != 1 and gold[i] in predicts[i]:
            correct += 1
        total += 1
        
    accuracy = correct/total
    result = '{}: {}'.format(gi, accuracy)
    print(result)

    return result


if __name__ == '__main__':
    seq_path = '../trained/gentleinput_seq/'
    tokenizer_path = '../trained/tokenizer.pkl'
    model_path = '../trained/demo_model.h5'
    gentle_inputs_path = '../trained/gentleinput_list.txt'
    k = 100

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

        results.append(eval(model, X, y, gi, k, tokenizer))

    print(results)
