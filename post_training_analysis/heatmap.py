#!/usr/bin/env python

import os
import re
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "21/05/2024"

# Folder paths
images_dir = r'D:\path\to\input\images'
csv_path = r'D:\path\to\CNN\test\summary.csv'

# Variables
chunk_size = 100


def string_to_list(string):
    return [item.strip() for item in string.split(',')]


def string_to_list_image(string):
    return [item.strip() for item in string.split('_norm.jpg.jpg,')]


def extract_info(filename):
    matches = re.findall(r'x=(\d+),y=(\d+),w=(\d+),h=(\d+)', filename)
    if matches:
        x, y, w, h = matches[0]
        return int(x), int(y), int(w), int(h)
    else:
        return None


def heatmap(figname, value, form, min_x, min_y, max_x, max_y, cmap, cmap_bin, data):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    if form == 'large':
        ax.set_axis_off()
        plt.title('Large heatmap')
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
    elif form == 'coords':
        plt.title('Coordinate Heatmap')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
    for i, row in data.iterrows():
        for j in range(len(row[value])):
            info = extract_info(row['image'][j])
            x, y, w, h = info
            heat = row[value]
            if value == 'probs':
                color = cmap(heat)
            elif value == 'predictions':
                color = cmap_bin(heat)
            sqr_color = tuple(color[j])
            rect = patches.Rectangle((x, y), w, h, linewidth=0, edgecolor='none', facecolor=sqr_color, alpha=0.9)
            ax.add_patch(rect)
        plt.gca().invert_yaxis()
        plt.savefig(figname, bbox_inches='tight', pad_inches=0)
        ax.clear()
    
    
def overlay_heatmap(image_path, heatmap_path, output_path, img_lay, heat_lay):
    Image.MAX_IMAGE_PIXELS = None
    img = plt.imread(image_path)
    heatmap = plt.imread(heatmap_path)
    height, width, _ = img.shape
    overlay_img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            img_chunk = img[y:y+chunk_size, x:x+chunk_size]
            heatmap_chunk = heatmap[y:y+chunk_size, x:x+chunk_size]
            overlay_chunk = img_lay * img_chunk + heat_lay * heatmap_chunk
            overlay_chunk = np.clip(overlay_chunk, 0, 255).astype(np.uint8)
            overlay_img[y:y+chunk_size, x:x+chunk_size] = overlay_chunk
    plt.imsave(output_path, overlay_img)


def main():
    df = pd.read_csv(csv_path, converters={'predictions': lambda x: [float(value) for value in x.split(',')],
                                           'probs': lambda x: [float(value) for value in x.split(',')],
                                           'labels': string_to_list,
                                           'image': string_to_list_image
                                           }
                     )
    print("Determining colour space")
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['cyan', 'red'])
    cmap_bin = mcolors.LinearSegmentedColormap.from_list("", ['cyan', 'red'], N=2)
    for sid, data in df.groupby('study_id'):
        print(f"\nProcessing SID: {sid}")
        for i, row in data.iterrows():
            for j in range(len(row['image'])):
                info = extract_info(row['image'][j])
                sid_dir = f"CZ_{row['image'][j].split(' ')[0]}"
        sid_dir = os.path.join(output, sid_dir)
        os.makedirs(sid_dir, exist_ok=True)
        heat_folders = ['Large', 'Coordinates', 'Overlay']
        for i in heat_folders:
            os.makedirs(os.path.join(sid_dir, i), exist_ok=True)
        print("\tDetermining canvas size")
        for i, row in data.iterrows():
            for j in range(len(row['image'])):
                min_x = min([extract_info(row['image'][j])[0] for j in range(len(row['image']))]) - 4500
                min_y = min([extract_info(row['image'][j])[1] for j in range(len(row['image']))]) - 4500
                max_x = max([extract_info(row['image'][j])[0] for j in range(len(row['image']))]) + 4500
                max_y = max([extract_info(row['image'][j])[1] for j in range(len(row['image']))]) + 4500
        width = max_x - min_x
        height = max_y - min_y
        
        print("\tGenerating canvas")
        heatmap_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        heatmap_prob = np.ones((height, width, 3), dtype=np.uint8) * 255
        heatmap_predict = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        print("\tGenerating large heatmap")
        heatmap(f'{sid_dir}\Large\CZ_{sid}_heatmap_large_probs.jpg', 'probs', 'large',
                min_x, min_y, max_x, max_y, cmap, cmap_bin, data)
        heatmap(f'{sid_dir}\Large\CZ_{sid}_heatmap_large_predict.jpg', 'predictions', 'large',
                min_x, min_y, max_x, max_y, cmap, cmap_bin, data)
        
        print("\tGenerating cropped heatmap with coordinates")
        heatmap(f'{sid_dir}\Coordinates\CZ_{sid}_heatmap_coords_probs.jpg',  'probs', 'coords',
                min_x, min_y, max_x, max_y, cmap, cmap_bin, data)
        heatmap(f'{sid_dir}\Coordinates\CZ_{sid}_heatmap_coords_predict.jpg', 'predictions', 'coords',
                min_x, min_y, max_x, max_y, cmap, cmap_bin, data)
        
        print("\tGenerating probability image heatmaps")
        for i, row in data.iterrows():
            for j in range(len(row['image'])):
                info = extract_info(row['image'][j])
                x, y, w, h = info
                full_sid = row['image'][j].split(' ')[0]
                if row['image'][j].endswith("_norm.jpg.jpg"):
                    img_path = os.path.join(images_dir, f"CZ_{full_sid}\CZ_{row['image'][j]}")
                else:
                  img_path = os.path.join(images_dir, f"CZ_{full_sid}\CZ_{row['image'][j]}_norm.jpg.jpg")
                img = plt.imread(img_path)
                img_resized = np.array(Image.fromarray(img).resize((w,h)))
                heatmap_img[y - min_y:y - min_y + h, x - min_x:x - min_x + w] = img_resized
                prob = row['probs']
                predict = row['predictions']
                rgba_color_prob = cmap(prob)
                rgba_color_predict = cmap_bin(predict)
                heatmap_prob[y - min_y:y - min_y + h, x - min_x:x - min_x + w] = \
                    (rgba_color_prob[j][:3] * 255).astype(np.uint8)
                heatmap_predict[y - min_y:y - min_y + h, x - min_x:x - min_x + w] = \
                    (rgba_color_predict[j][:3] * 255).astype(np.uint8)
        heatmap_img = np.clip(heatmap_img, 0, 255).astype(np.uint8)
        
        Image.MAX_IMAGE_PIXELS = None
        prob_map = False
        predict_map = False
        reconstruct_map = False
        
        print("\tSaving probability heatmap colours")
        try:
            plt.imsave(f'{sid_dir}\Overlay\CZ_{sid}_heatmap_probability_colours.tif', heatmap_prob)
            heatmap_probs_path = os.path.join(root, f"{sid_dir}\Overlay\CZ_{sid}_heatmap_probability_colours.tif")
            output_probs = os.path.join(root, f"{sid_dir}\Overlay\CZ_{sid}_heatmap_probability.tif")
        except MemoryError or OSError:
            prob_map = True
            continue
    
        try:
            print("\tSaving prediction heatmap colours")
            plt.imsave(f'{sid_dir}\Overlay\CZ_{sid}_heatmap_prediction_colours.tif', heatmap_predict)
            heatmap_predict_path = os.path.join(root, f"{sid_dir}\Overlay\CZ_{sid}_heatmap_prediction_colours.tif")
            output_predict = os.path.join(root, f"{sid_dir}\Overlay\CZ_{sid}_heatmap_prediction.tif")
        except MemoryError or OSError:
            predict_map = True
            continue
        try:
            print("\tSaving reconstructed image")
            plt.imsave(f'{sid_dir}\Overlay\CZ_{sid}_reconstructed_image.tif', heatmap_img)
            image_path = os.path.join(root, f"{sid_dir}\Overlay\CZ_{sid}_Reconstructed_image.tif")
        except MemoryError or OSError:
            reconstruct_map = True
            continue
        
        if prob_map or reconstruct_map is True:
            continue
        else:
            try:
                print("\tCreating probability heatmap")
                overlay_heatmap(image_path, heatmap_probs_path, output_probs, 0.6, 0.4)
            except MemoryError or OSError:
                continue
        if predict_map or reconstruct_map is True:
            continue
        else:
            try:
                print("\tCreating prediction heatmap")
                overlay_heatmap(image_path, heatmap_predict_path, output_predict, 0.8, 0.2)
            except MemoryError or OSError:
                continue


if __name__ == '__main__':
    print("\nStart generating heatmaps")
    root = os.path.dirname(os.path.realpath(__file__))
    output = os.path.join(root, "output_heatmaps")
    os.makedirs(output, exist_ok=True)
    main()
    print("\nAll heatmaps generated")
