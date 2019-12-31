import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from pickle import load
import argparse
from lm_lib.read import read_tasa
from tqdm import tqdm
import json
from lm_lib.text import TasaTextEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='production nltk')
    parser.add_argument('-sp', action="store", dest="model_path", type=str, default='../trained/nltk_model.pkl', help='set model path')
    parser.add_argument('-t', action="store", dest="target", type=str, default='tasaTest.txt', help='set target path')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')

    args = parser.parse_args()
    model_path = args.model_path
    target = '../data/{}'.format(args.target)
    context_win = args.context_win

    print('Loading trained model...')
    model = load(open(model_path, 'rb'))

    print('Reading target text...')
    tts = read_tasa(target)

    for tt in tts:
        print('tasaText: {}\n'.format(tt.name))

        # hard code for all candidates
        all_candidates = [["language", "tool", "the", ".", ","] for _ in range(tt.length)]

        tt.construct_prob_table_nltk(model, all_candidates, context_win)

        output = json.dumps(tt, cls=TasaTextEncoder, indent=4)
        output_path = '../nltk_output/{}.json'.format(tt.name)
        with open(output_path, 'w+', encoding='utf8') as fp:
            fp.write(output)

