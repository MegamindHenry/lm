import pickle

def replace(file_name):
    return file_name.replace("clean.wav", "clean.")

df = pickle.load(open("../trained/top10-dataframe.pickle", "rb"))

file_list = df.File.unique().tolist()
file_list = list(map(replace, file_list))
file_list.sort()

with open("../trained/file_list.txt", "w+") as fp:
    fp.write("\n".join(file_list))
