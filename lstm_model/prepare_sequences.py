import sys, os
from nltk.tokenize import word_tokenize
import argparse

path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from lm_lib.read import read_tasa, prepare_sequences_tasatext


def save_seq(path, sequences):
    
    text = '\n'.join(sequences)
    with open(path, 'w+', encoding='utf8') as fp:
    	fp.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-sp', action="store", dest="save_path", type=str, default='../trained/', help='set save path')

    args = parser.parse_args()
    context_win = args.context_win
    file_name = 'sequences_{}.txt'.format(context_win)
    save_path = args.save_path + file_name

    tts = read_tasa('../data/tasaDocs.txt')
    sequences = prepare_sequences_tasatext(tts, context_win)
    save_seq(save_path, sequences)


    
