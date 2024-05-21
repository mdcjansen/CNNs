#!/usr/bin/env python

import os

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "21/05/2024"

# Folder paths
input_dir = r"D:\MDCJ\10X-dataset\9-oversampeled"


def main():
    for _, dirs, _ in os.walk(input_dir):
        for d in dirs:
            for i in range(1, 5):
                if d == f"Fold-{i}":
                    fold_list_train = set([])
                    fold_list_val = set([])
                    for _, _, files in os.walk(f"{input_dir}\\Fold-{i}\Train"):
                        for f in files:
                            fold_list_train.add(f.split("_")[1])
                    for _, dirs, _ in os.walk(f"{input_dir}\\Fold-{i}\Val"):
                        for f in files:
                            fold_list_val.add(f.split("_")[1])
                    print(f"Fold {i} Train:\t{fold_list_train}")
                    print(f"Fold {i} Val:\t{fold_list_val}")                    
                if d == "Test":
                    fold_list = set([])
                    for _, dirs, _ in os.walk(f"{input_dir}\\Test"):
                        for f in files:
                            fold_list.add(f.split("_")[1])
                    print(f"Test:\t{fold_list}")


if __name__ == '__main__':
    main()
