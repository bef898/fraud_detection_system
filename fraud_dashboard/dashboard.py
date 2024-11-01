# dashboard.py
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import requests
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = Dash(__name__, server=False)

# Helper functions to fetch data from Flask API
def fetch_summary_data():
    response = requests.get('http://127.0.0.1:5000/api/summary')
    return response.json()

def fetch_fraud_over_time():
    response = requests.get('http://127.0.0.1:5000/api/fraud_over_time')
    return pd.read_json(response.json())

# Dash layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary boxes
    html.Div(id="summary-boxes"),
    
    # Time Series Chart
    html.Div([
        dcc.Graph(id='fraud-over-time')
    ])
])

# Callback to update summary statistics
@app.callback(
    Output('summary-boxes', 'children'),
    Input('fraud-over-time', 'value')
)
def update_summary_boxes(n):
    data = fetch_summary_data()
    return [
        html.Div(f"Total Transactions: {data['total_transactions']}"),
        html.Div(f"Total Fraud Cases: {data['total_fraud_cases']}"),
        html.Div(f"Fraud Percentage: {data['fraud_percentage']:.2f}%")
    ]

# Callback for fraud cases over time chart
@app.callback(
    Output('fraud-over-time', 'figure'),
    Input('fraud-over-time', 'value')
)
def update_fraud_over_time(n):
    df = fetch_fraud_over_time()
    fig = px.line(df, x='date', y='is_fraud', title="Fraud Cases Over Time")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
