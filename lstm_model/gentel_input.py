import glob
import os
import argparse

def findFiles(path):
    return glob.glob(path)

def save(path, files):
    with open(path, 'w+', encoding='utf8') as fp:
        output = '\n'.join(files)
        fp.write(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find all gentelinput file')
    parser.add_argument('-dp', action="store", dest="data_path", type=str, default='/mnt/shared/projects/RedHen/well_aligned/', help='set context windows')
    parser.add_argument('-sp', action="store", dest="save_path", type=str, default='../trained/', help='set save path')

    args = parser.parse_args()
    data_path = args.data_path + '*.gentelinput'
    save_path = args.save_path + 'gentelinput_list.txt'

    files = []

    for filename in findFiles(data_path):
        file = os.path.splitext(os.path.basename(filename))[0]
        files.append(file)

    save(save_path, files)