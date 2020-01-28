import sys, os
path = os.path.dirname(sys.path[0])
sys.path.insert(0, path)

import pickle
from tqdm import tqdm
from lm_lib import process_df
from pathlib import Path
import textgrids
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='find context for words in the top10-dataframe.pickle')
    parser.add_argument('--dev', action="store_true", dest="dev",
        help='set true for dev env')

    args = parser.parse_args()

    if args.dev:
        top10_df_path = "../trained/top10-dataframe.pickle"
        textgrid_file_path = "../playground/textgrid/"
    else:
        top10_df_path = "/mnt/shared/people/elnaz/language-model/top10-dataframe.pickle"
        textgrid_file_path = "/mnt/shared/projects/RedHen/well_aligned_clean/"

    file_list_path = "../trained/file_list.txt"
    context_win = 5

    df = pickle.load(open(top10_df_path, "rb"))

    files = process_df.load_list(file_list_path)

    df_context = dict()

    for f in tqdm(files):
        df_sub = df.loc[df['File'] == "{}{}".format(f, "wav")]

        n_rows = len(df_sub)

        textgrid_file = "{}{}TextGrid".format(textgrid_file_path, f)

        words = textgrids.TextGrid(textgrid_file)['words']

        j = 0
        context = ["<s>"]*(context_win+1)

        context_list = []
        for i in range(n_rows):
            word_df = df_sub.iloc[i]
            word_tg = words[j]

            while word_tg.text != word_df["wordtoken"] and\
                word_tg.xmin != word_df["start"] and\
                word_tg.xmax != word_df["end"]:

                j += 1
                word_tg = words[j]

                if word_tg.text != "":
                    context.pop(0)
                    context.append(word_tg.text)

                    context_list.append({
                        "target": word_tg.text,
                        "context": context[:context_win+1],
                        "candidates": [
                            word_df["rank{}".format(m+1)] for m in range(10)
                            ]
                        })

        df_context[f] = context_list

    df_output = json.dumps(df_context, indent=4)
    
    with open("top10-dataframe-context.json", "w") as fp:
        fp.write(df_output)