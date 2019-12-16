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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='produce prob')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-t', action="store", dest="target", type=str, default='../data/tasaS.txt', help='set target path')
    parser.add_argument('-mp', action="store", dest="model_path", type=str, default='../trained/demo_model.h5', help='set trained model path')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='../trained/tokenizer.pkl', help='tokenizer save path')

    args = parser.parse_args()

    context_win = args.context_win
    target = args.target
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path

    print('Load model...')
    model = load_model(model_path)

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))

    tts = read_tasa(target)

    # for tt in tts:
    #     tt.demo_prob_table()
    #     print("=====")
    #     print(json.dumps(tt, cls=TasaTextEncoder, indent=4))
    # quit()

    lines = prepare_sequences_tasatext(tts, context_win)
    # words = [l.split() for l in lines]

    # # print(words[1][:-1])
    # # quit()
    # contexts, golds = words[:][:-1], words[:][-1]

    # print(contexts[2], golds[2])
    # quit()

    sequences = tokenizer.texts_to_sequences(lines)
    sequences = np.array(sequences)

    X, y = sequences[:, :-1], sequences[:, -1]

    # print(X.shape)
    # print(y.shape)

    predicts_raw = model.predict(X)
    # print(predicts_raw.shape)

    aa = []
    for predict in predicts_raw:
        a = dict()
        for i, j in enumerate(predict):
            if i == 0:
                a['||'] = str(j)
                continue

            word = tokenizer.index_word[i]
            a[word] = str(j)
        aa.append(a)

    bb = [aa, aa]

    # for i, (aa, tt, gold) in enumerate(zip(bb, tts, golds)):
    #     # tt.demo_prob_table()
    #     # tt.
    #     # print("=====")
    #     # print(json.dumps(tt, cls=TasaTextEncoder, indent=4))
    #     pass

    length = len(tts)

    for i in range(length):
        # print(i, bb[i], tts[i], golds[i])
        # print("-----------")
        prob_table = []
        position_length = len(bb[i])
        for j in range(position_length):
            ttp = TasaTextProb(j, bb[i][j])
            prob_table.append(ttp)

        tts[i].prob_table = prob_table

    output = json.dumps(tts, cls=TasaTextEncoder, indent=4)

    with open('../trained/output.json', 'w+', encoding='utf8') as fp:
        fp.write(output)
    