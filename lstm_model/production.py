import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from keras.models import load_model
import argparse
from lm_lib.read import read_tasa, prepare_sequences_tasatext
from lm_lib.text import TasaTextEncoder, TasaTextProb
from pickle import load
import numpy as np
import json

def make_prob(predicts_raw, tokenizer, top_num):
    prob_list = []
    for predict in predicts_raw:
        prob = dict()

        top_index = np.argsort(predict)[::-1]
        for i in range(top_num):
            index = top_index[i]
            if index == 0:
                prob['||'] = str(predict[index])
                continue

            word = tokenizer.index_word[index]
            prob[word] = str(predict[index])

        prob_list.append(prob)

    # hard code here
    return [prob_list, prob_list]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='produce prob')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-t', action="store", dest="target", type=str, default='../data/tasaS.txt', help='set target path')
    parser.add_argument('-mp', action="store", dest="model_path", type=str, default='../trained/demo_model.h5', help='set trained model path')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='../trained/tokenizer.pkl', help='tokenizer save path')
    parser.add_argument('-tn', action="store", dest="top_num", type=int, default=100, help='top number of candidate to keep')

    args = parser.parse_args()

    context_win = args.context_win
    target = args.target
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    top_num = args.top_num

    print('Load model...')
    model = load_model(model_path)

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))

    tts = read_tasa(target)

    lines = prepare_sequences_tasatext(tts, context_win)

    sequences = tokenizer.texts_to_sequences(lines)
    sequences = np.array(sequences)

    X, y = sequences[:, :-1], sequences[:, -1]

    predicts_raw = model.predict(X)

    prob_list = make_prob(predicts_raw, tokenizer, top_num)

    length = len(tts)

    for i in range(length):
        prob_table = []
        position_length = len(prob_list[i])
        for j in range(position_length):
            ttp = TasaTextProb(j, prob_list[i][j])
            prob_table.append(ttp)

        tts[i].prob_table = prob_table

    output = json.dumps(tts, cls=TasaTextEncoder, indent=4)

    with open('../trained/output.json', 'w+', encoding='utf8') as fp:
        fp.write(output)
    