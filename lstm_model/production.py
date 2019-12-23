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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='produce prob')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-t', action="store", dest="target", type=str, default='../data/tasaTest.txt', help='set target path')
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

    for tt in tqdm(tts):
        tt.construct_prob_table(model, context_win, tokenizer, top_num)

        output = json.dumps(tt, cls=TasaTextEncoder, indent=4)
        output_path = '../output/{}.json'.format(tt.name)
        with open(output_path, 'w+', encoding='utf8') as fp:
            fp.write(output)