#!/usr/bin/env python

import os
import shutil

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.2"
__date__ = "21/05/2024"

# Folder paths
csv_path = r"D:\path\to\CNN\output\paramater_data.csv"
model_dir = r"D:\path\to\CNN\output\produced_models"


def main():   
    with open(csv_path, "r") as meta_file:
        for i in meta_file.readlines():
            model = i.split(',')[0]
            CNN = i.split(',')[10].strip("model_name=").strip("\n")
            batch_norm = i.split(',')[9].strip("batch_norm=").upper()
            output = f"{model_dir}/{CNN}/{fold}"
            os.makedirs(output, exist_ok=True)
            with open(f"{output}\{CNN}_{fold}_short_param.csv", 'a+') as model_csv:
                model_csv.write(f"{model},{batch_norm}\n")
            model_csv.close()
            for root, dirs, file in os.walk(model_dir):
                for f in file:
                    if len(dirs) != 0:
                        f_modelname = f.split("_")[4]
                        if model == f_modelname:
                            f_move = f"{model_dir}\{f}"
                            shutil.move(f_move, output)
    meta_file.close()


if __name__ == '__main__':
    fold = csv_path.split("_")[4].strip(".csv")
    main()
