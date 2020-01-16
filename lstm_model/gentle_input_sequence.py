from tqdm import tqdm
import string

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

def save_sequences(file_path, sequence_list):
    lines = ['\n'.join(sequences) for sequences in sequence_list]
    with open(file_path, 'w+', encoding='utf8') as fp:
        fp.write('\n'.join(lines))


if __name__ == '__main__':
    gentle_inputs_path = '../trained/gentleinput_list.txt'
    # data_path = '../test_data/'
    data_path = '/mnt/shared/projects/RedHen/well_aligned/'
    save_path = '../trained/gentleinput_seq.txt'
    context_win = 5

    with open(gentle_inputs_path, 'r', encoding='utf8') as fp:
        gentle_inputs = fp.read().split('\n')

    sequence_list = []
    for file in tqdm(gentle_inputs):
        file_path = data_path + file + '.gentleinput'
        segments = read_gentle(file_path, True)
        sequences = to_sequences(segments, context_win)
        sequence_list.append(sequences)

    save_sequences(save_path, sequence_list)

