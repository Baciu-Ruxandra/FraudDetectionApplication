import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logger

app = Flask(__name__)

# Load prediction pipeline once
predictor = PredictionPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    try:
        # Read CSV directly into memory
        df = pd.read_csv(file)
        logger.info(f"Uploaded file with {len(df)} rows and {len(df.columns)} columns.")

        # ✅ Use the DataFrame prediction method directly
        if hasattr(predictor, "predict_df"):
            results = predictor.predict_df(df)
        else:
            # fallback (older version of PredictionPipeline)
            from tempfile import NamedTemporaryFile
            tmp = NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            results = predictor.predict(tmp.name)

        # Format fraud probability nicely
        if "fraud_probability" in results.columns:
            results["fraud_probability"] = (results["fraud_probability"] * 100).round(2)

        # Summary info
        total = len(results)
        high_count = (results["is_fraud_predicted"] == 1).sum()
        risk_pct = round((high_count / total) * 100, 2) if total > 0 else 0

        summary = {
            "total": total,
            "high_count": high_count,
            "risk_pct": risk_pct
        }

        return render_template(
            "index.html",
            tables=results.to_dict(orient="records"),
            columns=results.columns.tolist(),
            summary=summary
        )

    except Exception as e:
        logger.exception(e)
        return f"<h3>Error processing file:</h3><pre>{e}</pre>"

if __name__ == "__main__":
    app.run(debug=True)
