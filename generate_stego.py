import cv2
import os
import numpy as np

cover_path = "dataset/cover"
stego_path = "dataset/stego"

if not os.path.exists(stego_path):
    os.makedirs(stego_path)

for category in os.listdir(cover_path):
    cover_category_path = os.path.join(cover_path, category)
    stego_category_path = os.path.join(stego_path, category)

    if not os.path.exists(stego_category_path):
        os.makedirs(stego_category_path)

    for file in os.listdir(cover_category_path):
        img_path = os.path.join(cover_category_path, file)
        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        stego_img = img.copy()
        flat = stego_img.flatten()

        total_pixels = len(flat)
        embed_pixels = 15000   # moderate but strong

        step = total_pixels // embed_pixels

        for i in range(0, total_pixels, step):
            flat[i] = flat[i] ^ 1

        stego_img = flat.reshape(stego_img.shape)

        cv2.imwrite(os.path.join(stego_category_path, file), stego_img)

print("Stego generation completed!")