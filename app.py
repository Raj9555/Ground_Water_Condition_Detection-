import os
import json
import joblib
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))



load_dotenv()

MODEL_PATH = "final_isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
DB_PATH = "predictions_history.db"

app = Flask(__name__, template_folder="templates")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ---------------- EMAIL ALERT FUNCTION ---------------- #
def send_email_alert(subject, message, to_email):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        print("Email Error: Missing EMAIL_SENDER or EMAIL_PASSWORD")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        print(f"Email sent successfully to {to_email}")

    except Exception as e:
        print("Email sending failed:", e)


# ---------------- DB FUNCTIONS ---------------- #
def get_db():
    if "_db" not in g:
        g._db = sqlite3.connect(DB_PATH)
        g._db.row_factory = sqlite3.Row
    return g._db


def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            state TEXT,
            district TEXT,
            latitude REAL,
            longitude REAL,
            features_json TEXT,
            raw_prediction INTEGER,
            decision_score REAL,
            label TEXT
        )
    """)
    db.commit()


@app.teardown_appcontext
def close_db(error):
    db = g.pop("_db", None)
    if db:
        db.close()


# ---------------- FEATURE ORDER ---------------- #
features_order = [
    "Recharge from rainfall During Monsoon Season",
    "Recharge from other sources During Monsoon Season",
    "Recharge from rainfall During Non Monsoon Season",
    "Recharge from other sources During Non Monsoon Season",
    "Total Natural Discharges",
    "Current Annual Ground Water Extraction For Irrigation",
    "Current Annual Ground Water Extraction For Domestic & Industrial Use",
    "Net Ground Water Availability for future use",
    "Stage of Ground Water Extraction (%)"
]

# ---------------- THRESHOLDS ---------------- #
critical_thresholds = {
    "Recharge from rainfall During Monsoon Season": 40075,
    "Recharge from other sources During Monsoon Season": 9366,
    "Recharge from rainfall During Non Monsoon Season": 4850,
    "Recharge from other sources During Non Monsoon Season": 12124,
    "Total Natural Discharges": 5597,
    "Current Annual Ground Water Extraction For Irrigation": 35530,
    "Current Annual Ground Water Extraction For Domestic & Industrial Use": 4215,
    "Net Ground Water Availability for future use": 26498,
    "Stage of Ground Water Extraction (%)": 60.7
}


# ---------------- HOME PAGE ---------------- #
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- PREDICT ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Build vector
    X = np.array([float(data[f]) for f in features_order]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    raw_pred = int(model.predict(X_scaled)[0])
    score = float(model.decision_function(X_scaled)[0])

    threshold_trigger = any(float(data[f]) > critical_thresholds[f] for f in critical_thresholds)
    final_label = "CRITICAL" if raw_pred == -1 or threshold_trigger else "SAFE"

    # ---- EMAIL ALERT (FROM .env ONLY) ---- #
    receiver_email = os.getenv("ALERT_EMAIL")

    if receiver_email:
        subject = f"Groundwater Alert: {final_label}"
        message = f"""
Groundwater condition at ({data.get('latitude')}, {data.get('longitude')}) is {final_label}.
Decision Score: {score}
""".strip()
        send_email_alert(subject, message, receiver_email)

    # ---- SAVE TO DB ---- #
    db = get_db()
    db.execute("""
        INSERT INTO predictions
        (timestamp, state, district, latitude, longitude, features_json,
         raw_prediction, decision_score, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(IST).isoformat(),
        None,
        None,
        float(data["latitude"]) if data.get("latitude") else None,
        float(data["longitude"]) if data.get("longitude") else None,
        json.dumps({f: data[f] for f in features_order}),
        raw_pred,
        score,
        final_label
    ))
    db.commit()

    return jsonify({
        "success": True,
        "label": final_label,
        "raw_prediction": raw_pred,
        "decision_score": score
    })


# ---------------- HISTORY ---------------- #
@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100").fetchall()

    history_list = []
    for r in rows:
        label = r["label"] if r["label"] else ("CRITICAL" if r["raw_prediction"] == -1 else "SAFE")

        history_list.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "features": json.loads(r["features_json"]),
            "raw_prediction": r["raw_prediction"],
            "decision_score": r["decision_score"],
            "label": label
        })

    return jsonify({"success": True, "history": history_list})


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)