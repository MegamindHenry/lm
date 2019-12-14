import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from nltk.tokenize import word_tokenize
import argparse
from lm_lib.read import read_tasa, prepare_sequences_tasatext


def save_seq(path, sequences):
    
    text = '\n'.join(sequences)
    with open(path, 'w+', encoding='utf8') as fp:
    	fp.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-sp', action="store", dest="save_path", type=str, default='../trained/', help='set save path')
    parser.add_argument('-sn', action="store", dest="seq_name", type=str, default='sequences', help='seq file name')
    parser.add_argument('-c', action="store", dest="corpus", type=str, default='tasaDocs.txt', help='set save path')


    args = parser.parse_args()
    context_win = args.context_win 
    seq_name = args.seq_name
    file_name = '{}_{}.txt'.format(seq_name, context_win)
    save_path = args.save_path + file_name
    corpus = '../data/{}'.format(args.corpus)

    tts = read_tasa(corpus)
    sequences = prepare_sequences_tasatext(tts, context_win)
    save_seq(save_path, sequences)


    
