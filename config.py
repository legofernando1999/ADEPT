'''
objective: project configurations file
- sets the path of the ROOT directory and DATA directory

'''
#import os
from pathlib import Path

# read directory from os library (returns string)
#ROOT_DIR = os.getcwd()
#DATA_DIR = os.path.join(ROOT_DIR, 'data')

# read directory from pathlib library (returns PosixPath object)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'


def main():
    pass

if __name__ == '__main__':
    main()
