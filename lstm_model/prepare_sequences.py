import sys, os
from nltk.tokenize import word_tokenize

path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from lm_lib.read import read_tasa, prepare_sequences_tasatext


def save_seq(path, sequences):
	text = '\n'.join(sequences)
	with open(path, 'w+', encoding='utf8') as fp:
		fp.write(text)


if __name__ == '__main__':
    context_win = 5
    seq_path = '../data/sequences.txt'

    tts = read_tasa('../data/tasaDocs.txt')
    sequences = prepare_sequences_tasatext(tts, context_win)
    save_seq(seq_path, sequences)


    
