import pickle
import argparse

def replace(file_name):
    return file_name.replace("clean.wav", "clean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get file list from top10-dataframe.pickle')
    parser.add_argument('--dev', action="store_true", dest="dev",
        help='set true for dev env')

    args = parser.parse_args()

    if args.dev:
        df_path = "../trained/top10-dataframe.pickle"
        df = pickle.load(open(df_path, "rb"))
        df = df.head(1000)
    else:
        df_path = "/mnt/shared/people/elnaz/language-model/top10-dataframe.pickle"
        df = pickle.load(open(df_path, "rb"))

    file_list = df.File.unique().tolist()
    file_list = list(map(replace, file_list))
    file_list.sort()

    with open("../trained/file_list.txt", "w+") as fp:
        fp.write("\n".join(file_list))
