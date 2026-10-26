import os

base = "dataset/cover"
total = 0

for folder in os.listdir(base):
    folder_path = os.path.join(base, folder)
    total += len(os.listdir(folder_path))

print("Total cover images:", total)