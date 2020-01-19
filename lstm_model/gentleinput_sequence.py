import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from pathlib import Path
from tqdm import tqdm
import string
from lm_lib.read import read_gentleinput_list
import argparse

def read_gentle(file, remove_punc=True):
    fp = open(file, 'r', encoding='utf8')
    segments = fp.read().split('\n')
    fp.close()

    if remove_punc:
        return [segment for segment in segments
            if segment not in string.punctuation]

    return segments

def to_sequences(segments, context_win):
    length = context_win + 1
    sequences = list()

    # append open tags
    for i in range(context_win):
        open_tag = "<s>"
        # construct open tags
        seq = [open_tag] * (length-i-1) + segments[:i+1]
        # convert into a line
        line = ' '.join(seq)
        sequences.append(line)

    for i in range(length, len(segments)+1):
        # select sequence of tokens
        seq = segments[i - length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)

    # append closed tags 
    closed_tag = "</s>"
    seq = segments[-length+1:] + [closed_tag]
    line = ' '.join(seq)
    sequences.append(line)

    return sequences

def save_sequences(file_path, sequences):
    with open(file_path, 'w+', encoding='utf8') as fp:
        fp.write('\n'.join(sequences))


if __name__ == '__main__':
    gentle_inputs_path = '../trained/gentleinput_list.txt'
    data_path = '/mnt/shared/projects/RedHen/well_aligned/'
    save_path = '../trained/gentleinput_seq/'
    context_win = 5

    parser = argparse.ArgumentParser(description='find all gentelinput and output a sequences of all gentleinput files')
    parser.add_argument("-tl", help="test in local env", action="store_true", dest="test_local")


    args = parser.parse_args()
    test_local = args.test_local
    if test_local:
        data_path = '../test_data/'


    gentle_inputs = read_gentleinput_list(gentle_inputs_path)

    Path('../trained/gentleinput_seq/').mkdir(parents=True, exist_ok=True)

    # sequence_list = []
    for file in tqdm(gentle_inputs):
        file_path = data_path + file + '.gentleinput'
        segments = read_gentle(file_path, True)
        sequences = to_sequences(segments, context_win)
        seq_path = '{}{}_{}.txt'.format(save_path, file, context_win)
        save_sequences(seq_path, sequences)

