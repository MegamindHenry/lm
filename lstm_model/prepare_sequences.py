"""Prepare sequences for tokenizer and training
"""
import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

import argparse
from lm_lib.text import TasaText
from lm_lib.read import read_tasa
from tqdm import tqdm


def save_seq(path, tts_sequences):
    """save sequences in a file
    
    Args:
        path (str): save path
        tts_sequences (list): list of list of sequences
    """
    text = '\n'.join(sequence for tt_sequences in tts_sequences for sequence in tt_sequences)
    
    with open(path, 'w+', encoding='utf8') as fp:
        fp.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare context words for tasa corpus')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set context windows')
    parser.add_argument('-sp', action="store", dest="save_path", type=str, default='../trained/', help='set save path')
    parser.add_argument('-sn', action="store", dest="seq_name", type=str, default='sequences', help='seq file name')
    parser.add_argument('-c', action="store", dest="corpus", type=str, default='tasaTrain.txt', help='set train corpus path')


    args = parser.parse_args()
    context_win = args.context_win 
    seq_name = args.seq_name
    file_name = '{}_{}.txt'.format(seq_name, context_win)
    save_path = args.save_path + file_name
    corpus = '../data/{}'.format(args.corpus)

    tts = read_tasa(corpus)
    tts_sequences = [tt.to_sequences() for tt in tqdm(tts)]
    save_seq(save_path, tts_sequences)

