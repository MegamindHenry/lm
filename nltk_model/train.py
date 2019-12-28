import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

from pickle import load, dump
import argparse
from nltk.lm import WittenBellInterpolated, MLE, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train with nltk')
    parser.add_argument('-sp', action="store", dest="sents_path", type=str, default='../trained/sents.pkl', help='set save path')
    parser.add_argument('-mp', action="store", dest="model_path", type=str, default='../trained/nltk_model.pkl', help='set save model path')
    parser.add_argument('-mn', action="store", dest="model_name", type=str, default='KneserNeyInterpolated', help='set n-gram model')
    parser.add_argument('-n', action="store", dest="ngram", type=int, default=6, help='set n-gram number')


    args = parser.parse_args()
    sents_path = args.sents_path
    model_name = args.model_name
    ngram = args.ngram
    model_path = args.model_path

    print('Load tokenizer...')
    tokenized_text = load(open(sents_path, 'rb'))

    train_data, padded_sents = padded_everygram_pipeline(ngram, tokenized_text)

    if model_name == 'WittenBellInterpolated':
        model = WittenBellInterpolated(ngram)
    elif model_name == 'KneserNeyInterpolated':
        model = KneserNeyInterpolated(ngram)
    elif model_name == 'MLE':
        model = MLE(ngram)
    else:
        model = KneserNeyInterpolated(ngram)

    print('Training...')
    model.fit(train_data, padded_sents)
    print('Done')

    print('Saving files...')
    dump(model, open(model_path, 'wb'))

    print(model.logscore("tool", "language is never a good".split()))
