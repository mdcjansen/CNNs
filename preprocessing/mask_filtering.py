import os
import shutil
from PIL import Image

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "21/05/2024"

# Folder paths
source_directory = r"D:\path\to\input\directory"
target_directory = r"D:\path\to\output\directory"

# Variables
background_threshold = 0.75


def move_image_with_structure(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)


def has_excess_white_background(img_path, threshold):
    with Image.open(img_path) as img:
        pixels = img.convert('RGB').getdata()
        white_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
        total_pixels = img.width * img.height
        return (white_count / total_pixels) > threshold


def process_images(root_dir, target_dir, threshold):
    print(f"Starting processing for directory: {root_dir}")
    for foldername, subfolders, filenames in os.walk(root_dir):
        print(f"Processing folder: {foldername}")
        for filename in filenames:
            if filename.endswith('.png'):
                png_path = os.path.join(foldername, filename)
                print(f"Processing PNG: {png_path}")
                if has_excess_white_background(png_path, threshold):
                    jpg_filename = filename.replace('.png', '.jpg')
                    jpg_path = os.path.join(foldername, jpg_filename)
                    if os.path.exists(jpg_path):
                        new_png_path = png_path.replace(root_dir, target_dir)
                        new_jpg_path = jpg_path.replace(root_dir, target_dir)
                        move_image_with_structure(png_path, new_png_path)
                        move_image_with_structure(jpg_path, new_jpg_path)


if __name__ == "__main__":
    process_images(source_directory, target_directory, background_threshold)
    print("Processing completed.")
