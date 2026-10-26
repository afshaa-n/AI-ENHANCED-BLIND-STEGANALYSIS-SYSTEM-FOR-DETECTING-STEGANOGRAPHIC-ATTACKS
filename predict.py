import cv2
import numpy as np
import joblib
import sys


def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: Could not read image.")
        return None

    img = img.astype(np.float32)

    mean = np.mean(img)
    std = np.std(img)

    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    lsb = img % 2
    lsb_ratio = np.sum(lsb) / lsb.size

    horizontal_diff = np.sum(lsb[:, :-1] != lsb[:, 1:])
    horizontal_ratio = horizontal_diff / (img.shape[0] * (img.shape[1] - 1))

    vertical_diff = np.sum(lsb[:-1, :] != lsb[1:, :])
    vertical_ratio = vertical_diff / ((img.shape[0] - 1) * img.shape[1])

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = img - blur
    noise_var = np.var(noise)

    even_pixels = np.sum((img % 2) == 0)
    odd_pixels = np.sum((img % 2) == 1)
    even_odd_ratio = even_pixels / (odd_pixels + 1)

    return np.array([[mean, std, entropy, lsb_ratio,
                      horizontal_ratio, vertical_ratio,
                      noise_var, even_odd_ratio]])


# Load trained model
model = joblib.load("stego_model.pkl")

# Get image path from user
image_path = input("Enter image path: ")

features = extract_features(image_path)

if features is not None:
    prediction = model.predict(features)[0]

    print("\n===== Detection Result =====")

    if prediction == 0:
        print("Result: This image appears to be a COVER image.")
    else:
        print("Result: This image is likely a STEGO image.")

    print("============================")