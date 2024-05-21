from datetime import datetime
import multiprocessing
import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.6"
__date__ = "21/05/2024"

# Folder paths
filtered_folder = r"D:\previously\filtered\folder"
sorted_folder = r"D:\previously\sorted\folder"
extract_folder = r"D:\folder\to\extract\from"
filter_dest = r"D:\folder\to\move\files\to"

# Variables
suffix_filtered = ".suffix"
suffix_sorted = ".suffix"
suffix_extract = ".suffix"
num_workers = 12


def extract(file_info, source_folder, dest_folder):
    file_id, cx, cy, cw, ch = file_info
    for root, dirs, filenames in os.walk(source_folder):
        for filename in filenames:
            if filename == "file.file":
                continue
            if file_id in filename:
                extract_cx = int(filename.split(",")[1].replace("x=", ""))
                extract_cy = int(filename.split(",")[2].replace("y=", ""))
                extract_cw = int(filename.split(",")[3].replace("w=", ""))
                extract_ch = int(filename.split(",")[4].replace("h=", "").replace(suffix_extract, ""))
                extract_cx_end = extract_cx + extract_cw
                extract_cy_end = extract_cy + extract_ch
                if (extract_cx <= cx < extract_cx_end) and \
                        (extract_cx < cx + cw <= extract_cx_end):
                    if (extract_cy <= cy < extract_cy_end) and \
                            (extract_cy < cy + ch <= extract_cy_end):
                        source_path = os.path.join(root, filename)
                        dest_path = os.path.join(dest_folder, filename)
                        shutil.move(source_path, dest_path)


def process_filtered(filename):
    if filename == "file.file":
        return
    elif filename == "file.file":
        return
    elif filename.endswith(".suffix"):
        return
    file_info = filename.split()[0], \
        int(filename.split(",")[1].replace("x=", "")), \
        int(filename.split(",")[2].replace("y=", "")), \
        int(filename.split(",")[3].replace("w=", "")), \
        int(filename.split(",")[4].replace("h=", "").replace(suffix_filtered, ""))
    extract(file_info, extract_folder, filter_dest)


def process_sorted(filename):
    if filename == "file.file":
        return
    elif filename == "file.file":
        return
    file_info = filename.split()[0], \
        int(filename.split(",")[1].replace("x=", "")), \
        int(filename.split(",")[2].replace("y=", "")), \
        0.5 * int(filename.split(",")[3].replace("w=", "")), \
        0.5 * int(filename.split(",")[4].replace("h=", "").replace(suffix_sorted, ""))
    extract(file_info, extract_folder, filter_dest)


def main():
    print("\33[96m[{time}][INFO]:\33[94m\tStart Extraction".format(time=datetime.now().strftime("%H:%M:%S")))
    filtered_files = []
    sorted_files = []
    for root, dirs, filenames in os.walk(filtered_folder):
        filtered_files.extend(filenames)
    print("[{time}][INFO]:\tListed all files from filterd_folder"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    for root, dirs, filenames in os.walk(sorted_folder):
        sorted_files.extend(filenames)
    print("[{time}][INFO]:\tListed all files from sorted_folder"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    pool = multiprocessing.Pool(processes=num_workers)
    print("[{time}][INFO]:\tStart extraction using filtered files"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    pool.map(process_filtered, filtered_files)
    print("\33[92m[{time}][INFO]:\33[1m\tCompleted extraction from filtered files"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    print("[{time}][INFO]:\tStart extraction using sorted files"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    pool.map(process_sorted, sorted_files)
    print("\33[92m[{time}][INFO]:\33[1m\tCompleted extraction from sorted files"
          .format(time=datetime.now().strftime("%H:%M:%S")))
    pool.close()
    pool.join()
    print("\33[96m[{time}][INFO]:\33[94m\tExtraction completed"
          .format(time=datetime.now().strftime("%H:%M:%S")))


if __name__ == "__main__":
    main()
