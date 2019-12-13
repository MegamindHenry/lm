from lm_lib.text import TasaText
from nltk.tokenize import word_tokenize
import progressbar


def read_tasa(file):
    """Read tasa txt file and return

    :param file: (str) tasa file path
    :return: (list)(TasaText) Returning a list of TasaText
    """
    with open(file, 'r', encoding='utf8') as fp:
        corpus = fp.read()
        # texts = corpus.split('\n\n')
        # texts = [TasaText.from_text(text) if  for text in corpus.split('\n\n')]

        tts = []
        for text in corpus.split('\n\n'):
            tt = TasaText.from_text(text)
            if tt:
                tts.append(tt)

        return tts


def prepare_sequences_tasatext(tts):
    sequences = []
    length = len(tts)
    with progressbar.ProgressBar(max_value=length) as bar:
        for i in range(length):
            text = ' '.join(tts[i].sents)
            sequences += construct_sequences(text)
            bar.update(i)

    return sequences
        


def construct_sequences(text, context_win=5):
    tokens = word_tokenize(text)

    length = context_win + 1
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i - length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    return sequences