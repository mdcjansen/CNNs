#!/usr/bin/env python

import os
import sys
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.6"
__date__ = "21/05/2024"

# Folder paths
sorted_folder = r"D:\path\to\sorted_folder"
folder_to_sort = r"D:\path\to\folder_to_sort"
destination = r"D:\path\to\destination_folder"

# Variables
suffix_sf = ".suffix_sorted_folder"
suffix_fts = ".suffix_folder_to_sort"

if __name__ == "__main__":
    sorted_files = []
    if not os.path.exists(destination):
        os.makedirs(destination)
    for (dirpath, dirnames, filenames) in os.walk(sorted_folder):
        sorted_files.extend(filenames)
    sorted_files_sc = [suffix.replace(suffix_sf, suffix_fts) for suffix in sorted_files]
    for (dirpath, dirnames, filenames) in os.walk(folder_to_sort):
        if len(dirnames) == 0:
            for i in range(0, len(filenames)):
                if filenames[i] in sorted_files_sc:
                    if filenames[i] != "exclude_specific_file":
                        print("MATCHEDFILE:\t{dp}\{dr}".format(dp=dirpath, dr=filenames[i]))
                        shutil.move("{dp}\{dr}".format(dp=dirpath, dr=filenames[i]), destination)
    sys.exit(0)
