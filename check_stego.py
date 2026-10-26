import cv2
import os
import numpy as np

cover_folder = "dataset/cover/animals"
stego_folder = "dataset/stego/animals"

# Pick first image automatically
filename = os.listdir(cover_folder)[0]

cover_path = os.path.join(cover_folder, filename)
stego_path = os.path.join(stego_folder, filename)

cover = cv2.imread(cover_path, 0)
stego = cv2.imread(stego_path, 0)

difference = np.sum(cover != stego)

print("Checking file:", filename)
print("Number of different pixels:", difference)