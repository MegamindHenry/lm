import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from pickle import load, dump
import argparse
from nltk.lm import WittenBellInterpolated, MLE, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train with nltk')
    parser.add_argument('-sp', action="store", dest="sents_path", type=str, default='../trained/sents.pkl', help='set save path')
    parser.add_argument('-mp', action="store", dest="model_path", type=str, default='../trained/nltk_model.pkl', help='set save model path')
    parser.add_argument('-mn', action="store", dest="model_name", type=str, default='KneserNeyInterpolated', help='set n-gram model')
    parser.add_argument('-cw', action="store", dest="context_win", type=int, default=5, help='set n-gram number')
    parser.add_argument('-co', action="store", dest="cutoff", type=int, default=8, help='set unknown words cutoff')


    args = parser.parse_args()
    sents_path = args.sents_path
    model_name = args.model_name
    ngram = args.context_win + 1
    model_path = args.model_path
    cutoff = args.cutoff

    print('Load tokenizer...')
    tokenized_texts = load(open(sents_path, 'rb'))

    tokens = [token for text in tokenized_texts for token in text]

    vocab = Vocabulary(tokens, unk_cutoff=cutoff, unk_label='<UNK>')

    train_data, _ = padded_everygram_pipeline(ngram, tokenized_texts)

    if model_name == 'WittenBellInterpolated':
        model = WittenBellInterpolated(ngram)
    elif model_name == 'KneserNeyInterpolated':
        model = KneserNeyInterpolated(ngram)
    elif model_name == 'MLE':
        model = MLE(ngram)
    else:
        model = KneserNeyInterpolated(ngram)

    print('Training...')
    model.fit(train_data, vocab)
    print('Training done')

    print('Saving files...')
    dump(model, open(model_path, 'wb'))
    print('Saving done')
