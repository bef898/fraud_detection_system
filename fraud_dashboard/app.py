from flask import Flask, jsonify, render_template
import pandas as pd
from dashboard import app as dash_app  # Import Dash app

app = Flask(__name__)

# Initialize and attach Dash app
dash_app.init_app(app)

# Include other Flask routes as in Step 3 here

if __name__ == '__main__':
    app.run(debug=True)

# Load fraud data from CSV
def load_data():
    df = pd.read_csv(r'C:\Users\befekadum\Documents\10x acadamy\week8and9\fraud_detection_system\fraud_dashboard\data\encoded_data1.csv')
    return df

# API endpoint for summary statistics
@app.route('/api/summary')
def summary():
    df = load_data()
    total_transactions = len(df)
    total_fraud_cases = df['is_fraud'].sum()
    fraud_percentage = (total_fraud_cases / total_transactions) * 100
    summary_stats = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }
    return jsonify(summary_stats)

# API endpoint for fraud cases over time
@app.route('/api/fraud_over_time')
def fraud_over_time():
    df = load_data()
    fraud_over_time = df.groupby('date')['is_fraud'].sum().reset_index()
    return fraud_over_time.to_json(orient='records')

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
