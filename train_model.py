import os
import cv2
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
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

    return [
        mean,
        std,
        entropy,
        lsb_ratio,
        horizontal_ratio,
        vertical_ratio,
        noise_var,
        even_odd_ratio
    ]


# -----------------------------
# Load Dataset
# -----------------------------
X = []
y = []

for category in os.listdir("dataset/cover"):
    folder_path = os.path.join("dataset/cover", category)
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            features = extract_features(os.path.join(folder_path, file))
            if features is not None:
                X.append(features)
                y.append(0)

for category in os.listdir("dataset/stego"):
    folder_path = os.path.join("dataset/stego", category)
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            features = extract_features(os.path.join(folder_path, file))
            if features is not None:
                X.append(features)
                y.append(1)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))


# -----------------------------
# ML Pipeline
# -----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced'))
])


# -----------------------------
# Hyperparameter Grid
# -----------------------------
param_grid = {
    'svm__C': [1, 5, 10, 20, 50],
    'svm__gamma': ['scale', 0.01, 0.1, 1],
    'svm__kernel': ['rbf']
}


# -----------------------------
# Cross Validation + Grid Search
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=1
)

grid.fit(X, y)


# -----------------------------
# Results
# -----------------------------
print("\nBest Parameters Found:")
print(grid.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid.best_score_)

# Evaluate final model on full dataset predictions
y_pred = grid.predict(X)

print("\nConfusion Matrix (Full Data):")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

import joblib

joblib.dump(grid.best_estimator_, "stego_model.pkl")

print("\nModel saved as stego_model.pkl")