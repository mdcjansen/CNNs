import os
import pandas as pd
import numpy as np
import cv2
import shutil
import random

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "14/05/2024"

# Parameter file path
train_save_dir = r"D:\path\to\save\directory"
excel_file = r"D:\path\to\binary.xlsx"
output_dir = r"D:\path\to\output\directory"


def random_flip_save(image_list, output_dir, counter):
    while True:
        img_path = np.random.choice(image_list, 1)[0]
        img = cv2.imread(img_path)
        if (f"{img_path}_hflip" in flip_history) and (f"{img_path}_vflip" in flip_history):
            continue
        rand_flip_no = np.random.randint(-1, 1)
        if rand_flip_no == -1:
            flip_type = 'hvflip'
            flip_key = f"{img_path}_{flip_type}"
            img = cv2.flip(img, -1)
        elif rand_flip_no == 0:
            flip_type = 'vflip'
            flip_key = f"{img_path}_{flip_type}"
            img = cv2.flip(img, 0)
        elif rand_flip_no == 1:
            flip_type = 'hflip'
            flip_key = f"{img_path}_{flip_type}"
            img = cv2.flip(img, 1)
        flip_history.add(flip_key)
        flipped_img_name = f"{os.path.basename(img_path).replace('.jpg', '')}_{flip_type}_{counter}.jpg"
        output_path = os.path.join(output_dir, flipped_img_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True


if __name__ == '__main__':
    df = pd.read_excel(excel_file)
    images = []
    labels = []
    majority_class = 0
    minority_class = 1
    counter = 0
    flip_history = set()
    study_id_to_label = dict(zip(df["study_id"], df["label"]))
    for root, _, files in os.walk(train_save_dir):
        for file in files:
            if file.endswith('.jpg'):
                study_id = int(file.split('_')[1])
                if study_id in study_id_to_label:
                    labels.append(study_id_to_label[study_id])
                    images.append(os.path.join(root, file))
    class_counts = {label: labels.count(label) for label in set(labels)}
    print(f"Class distribution before oversampling: {class_counts}")
    num_oversample_majority = int(class_counts[majority_class] * 0.0)
    num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])
    majority_images = [img for img, label in zip(images, labels) if label == majority_class]
    minority_images = [img for img, label in zip(images, labels) if label == minority_class]
    oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
    oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)
    for _ in oversampled_majority:
        output_folder = os.path.join(output_dir, str(majority_class))
        random_flip_save(majority_images, output_folder, counter)
        counter += 1
    for _ in oversampled_minority:
        output_folder = os.path.join(output_dir, str(minority_class))
        random_flip_save(minority_images, output_folder, counter)
        counter += 1
    for img_path in images:
        output_path = img_path.replace(train_save_dir, output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(img_path, output_path)
    class_counts[majority_class] += num_oversample_majority
    class_counts[minority_class] += num_oversample_minority
    print(f"Class distribution after oversampling: {class_counts}")
