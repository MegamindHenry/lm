"""helper methods for read tasa files and sequences
"""
from lm_lib.text import TasaText
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def read_tasa(file, remove_punc=True):
    """helper methods for read tasa file
    
    Args:
        file (str): file path of a tasa corpus
    
    Returns:
        list<-tasaText: list of tasaText objects
    """
    with open(file, 'r', encoding='utf8') as fp:
        corpus = fp.read()
        
        tts = []
        for text in tqdm(corpus.split('\n\n')):
            tt = TasaText.from_text(text, remove_punc)
            if tt:
                tts.append(tt)

        return tts


def load_seq(file):
    """read sequences for a file
    
    Args:
        file (str): file path
    
    Returns:
        list<-str: list of sequences
    """
    fp = open(file, 'r', encoding='utf8')
    text = fp.read()
    sequences = text.split('\n')
    return sequences