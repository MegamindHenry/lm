def load_list(file):
    with open(file, "r") as fp:
        files = fp.read().split("\n")
        return files