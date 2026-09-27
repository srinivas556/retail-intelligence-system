# Retail Intelligence — Sales Forecasting & Rack Recommendation

A Flask-based retail analytics and decision-support application.

## Features
- Upload CSV/XLSX/XLS historical sales data.
- Automatic column recognition for common retail column names.
- Missing-value handling.
- 70/30 train-test split.
- NumPy-based Linear Regression (chosen to avoid platform-specific scikit-learn DLL issues).
- MSE, RMSE and R² evaluation.
- Product/category demand aggregation.
- Transparent rack-placement suggestions based on predicted demand.
- Quick transaction sales estimate.
- Included sample dataset for demonstration.

## Expected data
Ideally include:
`Date, Product, Category, Quantity, Unit_Price, Discount, Sales, Store, Region`

The app can also derive Sales when Quantity and Price are available.

## Important
The included CSV is a recreated sample dataset for this portfolio project, not a claim that it is the user's original lost dataset. Replace it with a real historical dataset when available.

Rack recommendations are decision-support suggestions based on the uploaded data and model output. They are not guarantees of future sales.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

## Deployment
Suitable for Render or another Python/Flask host. Use:
`gunicorn app:app`
and add `gunicorn` to requirements.txt for deployment.
