from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load("celestial_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods = ["POST"])
def predict():
    alpha = float(request.form["alpha"])
    delta = float(request.form["delta"])
    u = float(request.form["u"])
    g = float(request.form["g"])
    r = float(request.form["r"])
    i = float(request.form["i"])
    z = float(request.form["z"])
    cam_col = float(request.form["cam_col"])
    redshift = float(request.form["redshift"])
    input_data = np.array([[
        alpha, delta, u, g, r, i, z, cam_col, redshift
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_class = encoder.inverse_transform(prediction)[0]
    return render_template(
        "index.html",
        prediction = predicted_class
    )
if __name__ == "__main__":
    app.run(debug = True)
