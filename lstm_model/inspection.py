import lib_path

import argparse
from keras.models import load_model
import pickle
import json
import numpy as np
from lm_lib import math


def acc(gold, speech, combine):
    total = 0
    correct_s = 0
    correct_c = 0

    for g, s, c in zip(gold, speech, combine):
        total += 1
        if g == s:
            correct_s += 1
        if g == c:
            correct_c += 1


        if g == c and g == s:
            print(g)

    return total, correct_s, correct_c



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate accuracy")

    args = parser.parse_args()


    model_path = "../trained/demo_model.h5"
    tokenizer_path = "../trained/tokenizer.pkl"
    df_context_path = "../trained/top10-dataframe-context.json"
    context_win = 5

    print("Load model...")
    model = load_model(model_path)

    print("Load tokenizer...")
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    print("Load df...")
    data = json.load(open(df_context_path, 'r'))

    for f, v in data.items():
        lines = [" ".join(t["context"][:context_win]) for t in v]
        sequences = tokenizer.texts_to_sequences(lines)
        sequences = np.array(sequences)

        candidates = [" ".join(t["candidates"]) for t in v]

        gold = [t["target"] for t in v]
        gold = tokenizer.texts_to_sequences(gold)
        gold = np.array(gold).flatten()

        predicts_speech = [t["candidates"][0] for t in v]
        predicts_speech = tokenizer.texts_to_sequences(predicts_speech)
        predicts_speech = np.array(predicts_speech).flatten()

        candidates_index = tokenizer.texts_to_sequences(candidates)
        candidates_index = np.array(candidates_index)

        predicts_raw = model.predict(sequences)

        predicts_combine = [c[np.argmax(p[c])]\
            for c, p in zip(candidates_index, predicts_raw)]
        predicts_combine = np.array(predicts_combine)





        predicts_top_k = math.argmax_top_k(predicts_raw, 10)

        for g, pk, c, pr, ci, pc in zip(gold, predicts_top_k, candidates,
                predicts_raw, candidates_index, predicts_combine):
            g = tokenizer.index_word[g]
            # pk = [tokenizer.index_word[x] for x in pk]
            pk_d = {tokenizer.index_word[x]: pr[x] for x in pk}
            c_d = {c.split(" ")[x]: pr[ci][x] for x in range(10)}
            pc = tokenizer.index_word[pc]

            # if g in pk:
            print("\n"*1)
            print("="*10)
            print("Gold: {}".format(g))
            print("Prediction Combine: {}".format(pc))
            print("Prediction: {}".format(pk_d))
            print("Candidates: {}".format(c))
            print("{}".format(c_d))

        # total, correct_s, correct_c = acc(gold,
        #     predicts_speech, predicts_combine)

        # print("Total: {}, Correct_s: {}, Correct_c: {}, Acc: N/A".format(
        #     total, correct_s, correct_c))
        


        