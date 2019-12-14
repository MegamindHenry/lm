from lm_lib.text import TasaText
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def read_tasa(file):
    """Read tasa txt file and return

    :param file: (str) tasa file path
    :return: (list)(TasaText) Returning a list of TasaText
    """
    with open(file, 'r', encoding='utf8') as fp:
        corpus = fp.read()
        
        tts = []
        for text in corpus.split('\n\n'):
            tt = TasaText.from_text(text)
            if tt:
                tts.append(tt)

        return tts


def prepare_sequences_tasatext(tts, context_win=5):
    sequences = []

    for tt in tqdm(tts):
        text = ' '.join(tt.sents)
        sequences += construct_sequences(text, context_win)

    return sequences
        


def construct_sequences(text, context_win):
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


def load_seq(file):
    fp = open(file, 'r', encoding='utf8')
    text = fp.read()
    sequences = text.split('\n')
    return sequences