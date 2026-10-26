import cv2
import os

# INPUT: where her Flowers folder is
input_folder = r"C:\Users\Afshan Zakir\OneDrive\Desktop\Flowers"

# OUTPUT: where resized images should go
output_folder = r"C:\Users\Afshan Zakir\OneDrive\Desktop\MINI PROJECT\dataset\cover\flowers"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is not None:
                resized_img = cv2.resize(img, (256, 256))

                folder_name = os.path.basename(root)
                new_name = folder_name.replace(" ", "") + "" + file

                save_path = os.path.join(output_folder, new_name)
                cv2.imwrite(save_path, resized_img)

print("Resizing completed successfully!")
print("Total images:", len(os.listdir(output_folder)))