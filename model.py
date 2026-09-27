from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_ALIASES = {
    "date": ["date", "order_date", "transaction_date"],
    "product": ["product", "product_name", "item"],
    "category": ["category", "product_category"],
    "quantity": ["quantity", "qty", "units", "units_sold"],
    "price": ["unit_price", "price", "selling_price"],
    "sales": ["sales", "revenue", "total_sales", "amount"],
    "store": ["store", "store_name", "branch"],
    "region": ["region", "area", "location"],
    "discount": ["discount", "discount_rate", "discount_percent"],
}


def normalize_columns(df):
    mapping = {}
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for target, aliases in REQUIRED_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[target] = normalized[alias]
                break
    return mapping


def prepare_dataframe(df):
    df = df.copy()
    mapping = normalize_columns(df)

    if "sales" not in mapping and {"quantity", "price"}.issubset(mapping):
        q = pd.to_numeric(df[mapping["quantity"]], errors="coerce")
        p = pd.to_numeric(df[mapping["price"]], errors="coerce")
        if "discount" in mapping:
            d = pd.to_numeric(df[mapping["discount"]], errors="coerce").fillna(0)
            if d.max() > 1:
                d = d / 100
            df["__sales__"] = q * p * (1 - d)
        else:
            df["__sales__"] = q * p
        mapping["sales"] = "__sales__"

    if "quantity" not in mapping or "sales" not in mapping:
        raise ValueError("Dataset needs at least Quantity and Sales (or Quantity and Price so Sales can be calculated).")

    df["__quantity__"] = pd.to_numeric(df[mapping["quantity"]], errors="coerce")
    df["__sales__"] = pd.to_numeric(df[mapping["sales"]], errors="coerce")

    if "price" in mapping:
        df["__price__"] = pd.to_numeric(df[mapping["price"]], errors="coerce")
    else:
        df["__price__"] = np.where(df["__quantity__"] != 0, df["__sales__"] / df["__quantity__"], 0)

    if "discount" in mapping:
        df["__discount__"] = pd.to_numeric(df[mapping["discount"]], errors="coerce").fillna(0)
        if df["__discount__"].max() > 1:
            df["__discount__"] /= 100
    else:
        df["__discount__"] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in ["__quantity__", "__sales__", "__price__", "__discount__"]:
        df[c] = df[c].ffill().bfill()

    return df, mapping


def linear_regression(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(X1, y, rcond=None)[0]
    return coef


def metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return mse, rmse, r2


def analyze_dataset(df, output_dir):
    df, mapping = prepare_dataframe(df)
    n = len(df)
    if n < 5:
        raise ValueError("Please upload at least 5 rows of data.")

    X = df[["__quantity__", "__price__", "__discount__"]].to_numpy(float)
    y = df["__sales__"].to_numpy(float)

    split = max(2, int(n * 0.70))
    if split >= n:
        split = n - 1

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    coef = linear_regression(X_train, y_train)
    pred = np.column_stack([np.ones(len(X_test)), X_test]) @ coef
    mse, rmse, r2 = metrics(y_test, pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    pred_rows = []
    for actual, p in zip(y_test, pred):
        pred_rows.append({"actual": round(float(actual), 2), "predicted": round(float(p), 2)})

    forecast = df.copy()
    forecast["predicted_sales"] = (
        coef[0]
        + coef[1] * forecast["__quantity__"]
        + coef[2] * forecast["__price__"]
        + coef[3] * forecast["__discount__"]
    )

    # Aggregate product/category demand for recommendations.
    if "product" in mapping:
        product_col = mapping["product"]
        group = forecast.groupby(product_col, dropna=False).agg(
            historical_sales=("__sales__", "sum"),
            predicted_sales=("predicted_sales", "sum"),
            quantity=("__quantity__", "sum"),
        ).reset_index()
        group = group.sort_values("predicted_sales", ascending=False)
        forecast_rows = group.head(20).to_dict("records")
    elif "category" in mapping:
        category_col = mapping["category"]
        group = forecast.groupby(category_col, dropna=False).agg(
            historical_sales=("__sales__", "sum"),
            predicted_sales=("predicted_sales", "sum"),
            quantity=("__quantity__", "sum"),
        ).reset_index()
        group = group.sort_values("predicted_sales", ascending=False)
        forecast_rows = group.head(20).to_dict("records")
    else:
        forecast_rows = []

    return {
        "rows": n,
        "train_rows": split,
        "test_rows": n - split,
        "total_sales": float(df["__sales__"].sum()),
        "avg_sales": float(df["__sales__"].mean()),
        "max_sales": float(df["__sales__"].max()),
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": pred_rows,
        "forecast_rows": forecast_rows,
        "mapping": mapping,
        "coefficients": {
            "intercept": float(coef[0]),
            "quantity": float(coef[1]),
            "unit_price": float(coef[2]),
            "discount": float(coef[3]),
        },
    }


def predict_sales(quantity, unit_price, discount):
    if discount > 1:
        discount /= 100
    # Simple baseline model trained from the included sample data.
    # This is intentionally labeled as a demo prediction until the user uploads data.
    sales = quantity * unit_price * (1 - discount)
    return {
        "predicted_sales": float(sales),
        "method": "transaction-value baseline",
        "note": "Upload historical sales data to generate a data-trained model and rack recommendation.",
    }


def generate_recommendations(df, forecast_rows):
    if not forecast_rows:
        return [{
            "priority": "Data needed",
            "item": "Product/category",
            "reason": "Include a Product or Category column to generate rack-placement recommendations.",
            "placement": "Not enough information",
        }]

    recommendations = []
    max_sales = max(float(x["predicted_sales"]) for x in forecast_rows) or 1

    for row in forecast_rows[:10]:
        name = row.get("product", row.get("category", "Item"))
        value = float(row["predicted_sales"])
        ratio = value / max_sales

        if ratio >= 0.70:
            priority = "High"
            placement = "Front / easy-access rack"
            reason = "Higher predicted demand suggests keeping this item highly accessible and sufficiently stocked."
        elif ratio >= 0.40:
            priority = "Medium"
            placement = "Middle / standard rack"
            reason = "Moderate predicted demand suggests normal shelf visibility and stock levels."
        else:
            priority = "Low"
            placement = "Less prominent / back rack"
            reason = "Lower predicted demand suggests less prominent placement while maintaining availability."

        recommendations.append({
            "priority": priority,
            "item": str(name),
            "reason": reason,
            "placement": placement,
            "predicted_sales": value,
        })

    return recommendations
