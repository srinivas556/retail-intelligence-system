import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from model import analyze_dataset, predict_sales, generate_recommendations

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "retail-intelligence-demo-key"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_uploaded_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a CSV or Excel file.")
        return redirect(url_for("index"))

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        flash("Only CSV, XLSX, and XLS files are supported.")
        return redirect(url_for("index"))

    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / safe_name
    file.save(path)

    try:
        df = load_uploaded_file(path)
        result = analyze_dataset(df, OUTPUT_DIR)
        recs = generate_recommendations(df, result["forecast_rows"])

        return render_template(
            "dashboard.html",
            result=result,
            recommendations=recs,
            file_name=file.filename,
        )
    except Exception as exc:
        flash(f"Could not analyze the file: {exc}")
        return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        quantity = float(request.form["quantity"])
        unit_price = float(request.form["unit_price"])
        discount = float(request.form["discount"])
        result = predict_sales(quantity, unit_price, discount)
        return render_template("prediction.html", result=result)
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))


@app.route("/sample")
def sample():
    sample_path = BASE_DIR / "data" / "retail_sales.csv"
    return send_from_directory(sample_path.parent, sample_path.name, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
