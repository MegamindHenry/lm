import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

import argparse
from lm_lib.read import read_tasa
from tqdm import tqdm
from pickle import dump


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare tokenized sentences for tasa corpus')
    parser.add_argument('-sp', action="store", dest="sents_path", type=str, default='../trained/', help='set save path')
    parser.add_argument('-c', action="store", dest="corpus", type=str, default='tasaTrain.txt', help='set train corpus path')


    args = parser.parse_args()
    sents_path = '{}sents.pkl'.format(args.sents_path)
    corpus = '../data/{}'.format(args.corpus)

    print('Loading corpus...')
    tts = read_tasa(corpus)
    tts_sents = [tt.to_sents() for tt in tqdm(tts)]

    print('Saving files...')
    dump(tts_sents, open(sents_path, 'wb'))