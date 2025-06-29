#!/usr/bin/env python3
"""
Flask Web Application for EV Route Optimization System
Provides REST API endpoints for route optimization, energy prediction, and data visualization
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_integration.data_manager import DataManager
from ml_pipeline import EVMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize system components
config = Config()
data_manager = DataManager(config)
ml_pipeline = EVMLPipeline(config)

# Global variables to store system state
system_state = {
    'initialized': False,
    'models_loaded': False,
    'last_update': None,
    'status': 'initializing'
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get system status"""
    try:
        # Get real-time data summary
        data_summary = data_manager.get_real_time_data_summary()
        
        # Get ML pipeline status
        ml_status = 'ready' if ml_pipeline.trained_models else 'not_trained'
        
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'data_sources': data_summary.get('data_sources', {}),
            'ml_status': ml_status,
            'system_state': system_state
        })
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the ML pipeline and load models"""
    try:
        logger.info("🚀 Initializing EV Route Optimization System...")
        
        # Run ML pipeline with synthetic data for demo
        results = ml_pipeline.run_complete_pipeline(use_synthetic_data=True)
        
        # Update system state
        system_state['initialized'] = True
        system_state['models_loaded'] = True
        system_state['last_update'] = datetime.now().isoformat()
        system_state['status'] = 'ready'
        
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'results': {
                'best_model': results['pipeline_summary']['best_model'],
                'best_score': results['pipeline_summary']['best_r2_score'],
                'models_trained': results['pipeline_summary']['models_trained']
            }
        })
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        system_state['status'] = 'error'
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/route/optimize', methods=['POST'])
def optimize_route():
    """Optimize route for energy efficiency"""
    try:
        data = request.get_json()
        
        # Extract route parameters
        origin = data.get('origin')
        destination = data.get('destination')
        vehicle_params = data.get('vehicle_params', {})
        preferences = data.get('preferences', {})
        
        if not origin or not destination:
            return jsonify({
                'status': 'error',
                'error': 'Origin and destination are required'
            }), 400
        
        # Get comprehensive road network
        road_network = data_manager.get_comprehensive_road_network()
        
        if road_network.empty:
            return jsonify({
                'status': 'error',
                'error': 'No road network data available'
            }), 500
        
        # Calculate energy-efficient route (simplified for demo)
        route_segments = road_network.head(10)  # Demo: take first 10 segments
        
        # Calculate route statistics
        total_distance = route_segments.geometry.length.sum() * 111  # Convert to km
        total_energy = route_segments['segment_energy_wh'].sum() / 1000  # Convert to kWh
        avg_energy_per_km = route_segments['final_energy_wh_per_km'].mean()
        
        # Create route response
        route_data = {
            'route_id': f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'origin': origin,
            'destination': destination,
            'total_distance_km': round(total_distance, 2),
            'total_energy_kwh': round(total_energy, 2),
            'avg_energy_per_km': round(avg_energy_per_km, 2),
            'estimated_time_minutes': round(total_distance / 50 * 60, 0),  # Assume 50 km/h avg
            'route_segments': [
                {
                    'segment_id': i,
                    'distance_km': round(row.geometry.length * 111, 3),
                    'energy_wh_per_km': round(row['final_energy_wh_per_km'], 2),
                    'speed_limit': row.get('speed_limit_kmh', 50),
                    'road_type': row.get('road_type', 'unknown'),
                    'coordinates': [[row.geometry.bounds[1], row.geometry.bounds[0]], 
                                  [row.geometry.bounds[3], row.geometry.bounds[2]]]
                }
                for i, (_, row) in enumerate(route_segments.iterrows())
            ]
        }
        
        return jsonify({
            'status': 'success',
            'route': route_data
        })
        
    except Exception as e:
        logger.error(f"Route optimization failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/energy/predict', methods=['POST'])
def predict_energy():
    """Predict energy consumption for given parameters"""
    try:
        if not ml_pipeline.trained_models:
            return jsonify({
                'status': 'error',
                'error': 'ML models not loaded. Please initialize the system first.'
            }), 400
        
        data = request.get_json()
        
        # Create input DataFrame from request data
        input_data = pd.DataFrame([{
            'speed_limit_kmh': data.get('speed_limit_kmh', 50),
            'elevation_gradient': data.get('elevation_gradient', 0.0),
            'weather_temperature_celsius': data.get('temperature', 20),
            'weather_humidity_percent': data.get('humidity', 60),
            'weather_wind_speed_ms': data.get('wind_speed', 5),
            'weather_precipitation_mm': data.get('precipitation', 0),
            'traffic_density': data.get('traffic_density', 0.3),
            'traffic_average_speed_kmh': data.get('traffic_speed', 45),
            'ev_battery_capacity_kwh': data.get('battery_capacity', 75),
            'ev_efficiency_wh_per_km': data.get('base_efficiency', 150),
            'ev_battery_level_percent': data.get('battery_level', 0.8),
            'ev_vehicle_weight_kg': data.get('vehicle_weight', 1800),
            'energy_efficiency_factor': data.get('efficiency_factor', 1.0),
            'temperature_impact': data.get('temp_impact', 1.0),
            'traffic_energy_impact': data.get('traffic_impact', 1.1),
            'elevation_energy_impact': data.get('elevation_impact', 1.0),
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'is_rush_hour': 1 if datetime.now().hour in [7, 8, 9, 17, 18, 19] else 0,
            'speed_elevation_interaction': data.get('speed_limit_kmh', 50) * data.get('elevation_gradient', 0.0),
            'temperature_traffic_interaction': data.get('temperature', 20) * data.get('traffic_density', 0.3),
            'weather_traffic_interaction': data.get('precipitation', 0) * data.get('traffic_density', 0.3),
            'speed_limit_squared': data.get('speed_limit_kmh', 50) ** 2,
            'elevation_gradient_squared': data.get('elevation_gradient', 0.0) ** 2,
            'temperature_squared': data.get('temperature', 20) ** 2,
            'speed_to_limit_ratio': data.get('traffic_speed', 45) / data.get('speed_limit_kmh', 50),
            'energy_efficiency_ratio': data.get('base_efficiency', 150) / 22500,  # Normalized
            'highway_encoded': 1 if data.get('road_type') == 'highway' else 0,
            'road_type_encoded': {'highway': 0, 'arterial': 1, 'local': 2}.get(data.get('road_type', 'local'), 2),
            'surface_encoded': 0,  # Default paved
            'weather_condition_encoded': 0,  # Default clear
            'traffic_congestion_level_encoded': 1,  # Default medium
            'distance_to_nearest_charger_km': data.get('charger_distance', 2.0),
            'charger_density_per_km2': data.get('charger_density', 0.5),
        }])
        
        # Make prediction
        prediction = ml_pipeline.predict_energy_consumption(input_data)
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'energy_consumption_wh_per_km': round(float(prediction[0]), 2),
                'confidence': 'high' if ml_pipeline.trained_models['comparison']['best_score'] > 0.7 else 'medium',
                'model_used': ml_pipeline.trained_models['comparison']['best_model'],
                'factors': {
                    'base_efficiency': data.get('base_efficiency', 150),
                    'weather_impact': round((data.get('temperature', 20) - 20) * 0.01, 3),
                    'traffic_impact': round(data.get('traffic_density', 0.3) * 0.2, 3),
                    'elevation_impact': round(data.get('elevation_gradient', 0.0) * 0.1, 3)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Energy prediction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/data/road-network')
def get_road_network():
    """Get road network data for visualization"""
    try:
        road_network = data_manager.get_comprehensive_road_network()
        
        if road_network.empty:
            return jsonify({
                'status': 'error',
                'error': 'No road network data available'
            }), 404
        
        # Convert to GeoJSON format (simplified)
        features = []
        for idx, row in road_network.head(50).iterrows():  # Limit for demo
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[row.geometry.bounds[0], row.geometry.bounds[1]], 
                                  [row.geometry.bounds[2], row.geometry.bounds[3]]]
                },
                'properties': {
                    'segment_id': idx,
                    'speed_limit': row.get('speed_limit_kmh', 50),
                    'road_type': row.get('road_type', 'unknown'),
                    'energy_per_km': round(row['final_energy_wh_per_km'], 2),
                    'efficiency_factor': round(row['energy_efficiency_factor'], 3)
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return jsonify({
            'status': 'success',
            'data': geojson,
            'summary': {
                'total_segments': len(road_network),
                'avg_energy_per_km': round(road_network['final_energy_wh_per_km'].mean(), 2),
                'coverage_area_km2': data_manager._calculate_coverage_area(road_network)
            }
        })
        
    except Exception as e:
        logger.error(f"Road network data retrieval failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/data/charging-stations')
def get_charging_stations():
    """Get charging stations data"""
    try:
        charging_stations = data_manager.get_charging_stations_with_availability()
        
        if charging_stations.empty:
            return jsonify({
                'status': 'error',
                'error': 'No charging stations data available'
            }), 404
        
        # Convert to JSON format
        stations = []
        for idx, row in charging_stations.iterrows():
            station = {
                'id': idx,
                'latitude': float(row.geometry.y),
                'longitude': float(row.geometry.x),
                'name': row.get('name', f'Station {idx}'),
                'charging_type': row['charging_type'],
                'power_kw': float(row['charging_power_kw']),
                'is_available': bool(row['is_available']),
                'price_per_kwh': float(row['price_per_kwh']),
                'connector_types': row.get('connector_types', ['Type2']),
                'network': row.get('network', 'Unknown')
            }
            stations.append(station)
        
        return jsonify({
            'status': 'success',
            'data': stations,
            'summary': {
                'total_stations': len(charging_stations),
                'available_stations': int(charging_stations['is_available'].sum()),
                'fast_charging_stations': len(charging_stations[charging_stations['charging_type'] == 'fast']),
                'avg_price_per_kwh': round(float(charging_stations['price_per_kwh'].mean()), 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Charging stations data retrieval failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/weather/current')
def get_current_weather():
    """Get current weather data"""
    try:
        weather_data = data_manager.weather_collector.get_current_weather()
        
        return jsonify({
            'status': 'success',
            'data': weather_data
        })
        
    except Exception as e:
        logger.error(f"Weather data retrieval failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/ml/models')
def get_ml_models():
    """Get ML model information"""
    try:
        if not ml_pipeline.trained_models:
            return jsonify({
                'status': 'error',
                'error': 'No models trained yet'
            }), 404
        
        summary = ml_pipeline.get_pipeline_summary()
        
        return jsonify({
            'status': 'success',
            'data': summary
        })
        
    except Exception as e:
        logger.error(f"ML models data retrieval failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/export/data')
def export_data():
    """Export system data for analysis"""
    try:
        exported_files = data_manager.export_data_for_analysis("demo_exports")
        
        return jsonify({
            'status': 'success',
            'exported_files': exported_files,
            'message': 'Data exported successfully'
        })
        
    except Exception as e:
        logger.error(f"Data export failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize system on startup
    try:
        logger.info("🚀 Starting EV Route Optimization Web Application")
        system_state['status'] = 'starting'
        
        # Run in debug mode for development
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        system_state['status'] = 'failed' 