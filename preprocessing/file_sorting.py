#!/usr/bin/env python

import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.2"
__date__ = "26/10/2023"

if __name__ == "__main__":
    identifier = set([])
    for file in os.listdir():
        if file.endswith(".txt"):
            identifier.add(file.split()[0])
    for i in identifier:
        os.makedirs(i)
        files = [file for file in os.listdir() if file.endswith(".txt") if file.startswith(i)]
        for f in range(0, len(files)):
            shutil.move(files[f], i)
