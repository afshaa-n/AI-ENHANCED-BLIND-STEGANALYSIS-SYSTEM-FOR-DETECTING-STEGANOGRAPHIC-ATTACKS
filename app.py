from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("stego_model.pkl")

def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = np.mean(gray)
    std = np.std(gray)

    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0,256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    lsb = gray % 2
    lsb_ratio = np.sum(lsb) / lsb.size

    horizontal_diff = np.sum(lsb[:, :-1] != lsb[:, 1:])
    horizontal_ratio = horizontal_diff / (gray.shape[0]*(gray.shape[1]-1))

    vertical_diff = np.sum(lsb[:-1, :] != lsb[1:, :])
    vertical_ratio = vertical_diff / ((gray.shape[0]-1)*gray.shape[1])

    blur = cv2.GaussianBlur(gray,(5,5),0)
    noise = gray - blur
    noise_var = np.var(noise)

    even_pixels = np.sum((gray%2)==0)
    odd_pixels = np.sum((gray%2)==1)
    even_odd_ratio = even_pixels/(odd_pixels+1)

    return np.array([[mean,std,entropy,lsb_ratio,
                      horizontal_ratio,vertical_ratio,
                      noise_var,even_odd_ratio]])

@app.route("/", methods=["GET","POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        features = extract_features(img)

        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "COVER IMAGE"
        else:
            result = "STEGO IMAGE"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)