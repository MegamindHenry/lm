"""Produce prob tables for TaxeText
"""
import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

import argparse
from keras.models import load_model
from pickle import load
from lm_lib.read import read_tasa
from lm_lib.text import TasaTextEncoder
from tqdm import tqdm
import json
import pandas as pd


def to_pd(tt, top_num):
    tt_csv = {
        "targets": [],
    }
    for i in range(top_num):
        tt_csv["{}_pred".format(i+1)] = []
        tt_csv["{}_prob".format(i+1)] = []

    for prob in tt.prob_table_list:
        tt_csv["targets"].append(prob["target"])
        # for i, pred in enumerate(prob.prob_table):
        for i in range(top_num):
            tt_csv["{}_pred".format(i+1)].append(
                    prob["prob_table"][i]["candidate"]
                )
            tt_csv["{}_prob".format(i+1)].append(
                    prob["prob_table"][i]["probability"]
                )

    df = pd.DataFrame(tt_csv)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='produce prob')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-t', action="store", dest="target", type=str, default='tasaTest.txt', help='set target path')
    parser.add_argument('-mp', action="store", dest="model_path", type=str, default='../trained/demo_model.h5', help='set trained model path')
    parser.add_argument('-tp', action="store", dest="tokenizer_path", type=str, default='../trained/tokenizer.pkl', help='tokenizer save path')
    parser.add_argument('-tn', action="store", dest="top_num", type=int, default=100, help='top number of candidate to keep')

    args = parser.parse_args()

    context_win = args.context_win
    target = '../data/{}'.format(args.target)
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    top_num = args.top_num

    print('Load model...')
    model = load_model(model_path)

    print('Load tokenizer...')
    tokenizer = load(open(tokenizer_path, 'rb'))

    tts = read_tasa(target)

    for tt in tqdm(tts):
        tt.construct_prob_table(model, context_win, tokenizer, top_num)

        #json format
        output = json.dumps(tt, cls=TasaTextEncoder, indent=4)
        output_path = '../lstm_output/{}.json'.format(tt.name)
        with open(output_path, 'w+', encoding='utf8') as fp:
            fp.write(output)

        #csv format
        df = to_pd(tt, top_num)
        output = df.to_csv(index=False)
        output_path = '../lstm_output/{}.csv'.format(tt.name)
        with open(output_path, 'w+', encoding='utf8') as fp:
            fp.write(output)