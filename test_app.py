#!/usr/bin/env python3
"""
Simple test Flask app to verify basic functionality
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Create Flask app
app = Flask(__name__)
CORS(app)

# Simple HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EV Route Optimizer - Test</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background-color: #d4edda; color: #155724; }
        .info { background-color: #d1ecf1; color: #0c5460; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗⚡ EV Route Optimizer - Test Interface</h1>
        
        <div class="status success">
            <strong>✅ Frontend Test:</strong> HTML rendering is working!
        </div>
        
        <div class="status info">
            <strong>ℹ️ Backend Test:</strong> Click buttons below to test API endpoints
        </div>
        
        <h2>Backend API Tests</h2>
        <button onclick="testStatus()">Test Status API</button>
        <button onclick="testPrediction()">Test Energy Prediction</button>
        
        <div id="results" style="margin-top: 20px;"></div>
        
        <script>
            function testStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('results').innerHTML = 
                            '<div class="status success"><strong>Status API:</strong> ' + 
                            JSON.stringify(data, null, 2) + '</div>';
                    })
                    .catch(error => {
                        document.getElementById('results').innerHTML = 
                            '<div class="status" style="background-color: #f8d7da; color: #721c24;"><strong>Error:</strong> ' + 
                            error + '</div>';
                    });
            }
            
            function testPrediction() {
                const testData = {
                    speed_limit_kmh: 50,
                    temperature: 20,
                    traffic_density: 0.3,
                    elevation_gradient: 0.0
                };
                
                fetch('/api/energy/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(testData)
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('results').innerHTML = 
                        '<div class="status success"><strong>Prediction API:</strong> ' + 
                        JSON.stringify(data, null, 2) + '</div>';
                })
                .catch(error => {
                    document.getElementById('results').innerHTML = 
                        '<div class="status" style="background-color: #f8d7da; color: #721c24;"><strong>Error:</strong> ' + 
                        error + '</div>';
                });
            }
        </script>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Render test dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """Simple status endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'EV Route Optimizer Test API is running',
        'backend_test': 'success',
        'timestamp': '2024-01-01T12:00:00Z',
        'version': '1.0.0-test'
    })

@app.route('/api/energy/predict', methods=['POST'])
def predict_energy():
    """Simple energy prediction endpoint"""
    try:
        # Simulate energy prediction
        return jsonify({
            'status': 'success',
            'prediction': {
                'energy_consumption_wh_per_km': 180.5,
                'model_used': 'test_model',
                'confidence': 0.85
            },
            'message': 'Test prediction successful'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting EV Route Optimizer Test Server...")
    print("📍 URL: http://localhost:5000")
    print("🧪 This is a test version to verify basic functionality")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    ) 