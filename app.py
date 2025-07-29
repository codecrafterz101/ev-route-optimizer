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
from shapely.geometry import Point

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
        
        # Convert addresses to coordinates (simplified geocoding for Berlin)
        origin_coords = _geocode_address(origin)
        dest_coords = _geocode_address(destination)
        
        logger.info(f"Geocoded coordinates - Origin: {origin} -> {origin_coords}, Destination: {destination} -> {dest_coords}")
        
        if not origin_coords or not dest_coords:
            return jsonify({
                'status': 'error',
                'error': 'Could not geocode origin or destination addresses'
            }), 400
        
        # Find the closest road segments to origin and destination
        origin_segment = _find_closest_segment(road_network, origin_coords)
        dest_segment = _find_closest_segment(road_network, dest_coords)
        
        if origin_segment is None or dest_segment is None:
            return jsonify({
                'status': 'error',
                'error': 'Could not find road segments near origin or destination'
            }), 400
        
        # Calculate route between origin and destination
        route_segments = _calculate_route(road_network, origin_segment, dest_segment)
        
        if not route_segments:
            return jsonify({
                'status': 'error',
                'error': 'Could not calculate route between specified points'
            }), 400
        
        # Calculate route statistics
        total_distance = sum(seg.geometry.length * 111 for seg in route_segments)  # Convert to km
        # Calculate total energy by summing individual segment energies
        total_energy = 0
        for seg in route_segments:
            segment_distance_km = seg.geometry.length * 111
            # Handle pandas Series properly
            if hasattr(seg, 'get'):
                segment_energy_wh_per_km = seg.get('final_energy_wh_per_km', 150)
            else:
                segment_energy_wh_per_km = 150  # Default if not available
            segment_energy_wh = segment_distance_km * segment_energy_wh_per_km
            total_energy += segment_energy_wh
        total_energy = total_energy / 1000  # Convert to kWh
        
        # Calculate average energy per km
        energy_values = []
        for seg in route_segments:
            if hasattr(seg, 'get'):
                energy_values.append(seg.get('final_energy_wh_per_km', 150))
            else:
                energy_values.append(150)
        avg_energy_per_km = sum(energy_values) / len(route_segments) if route_segments else 150
        
        # Create route response with improved coordinate data
        route_data = {
            'route_id': f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'origin': origin,
            'destination': destination,
            'total_distance_km': round(total_distance, 2),
            'total_energy_kwh': round(total_energy, 2),
            'avg_energy_per_km': round(avg_energy_per_km, 2),
            'estimated_time_minutes': round(total_distance / 50 * 60, 0),  # Assume 50 km/h avg
            'route_segments': []
        }
        
        # Generate continuous path coordinates
        all_coordinates = []
        for i, seg in enumerate(route_segments):
            # Get segment coordinates
            if hasattr(seg.geometry, 'coords'):
                coords = list(seg.geometry.coords)
                # Shapely coordinates are (lon, lat), convert to [lat, lon] for Leaflet
                segment_coords = [[lat, lon] for lon, lat in coords]
            else:
                # Fallback to bounds if coords not available
                # bounds are (minx, miny, maxx, maxy) where x=lon, y=lat
                segment_coords = [
                    [seg.geometry.bounds[1], seg.geometry.bounds[0]],  # [lat, lon] from bounds
                    [seg.geometry.bounds[3], seg.geometry.bounds[2]]
                ]
            
            # Add intermediate points for smoother path
            if i > 0 and len(all_coordinates) > 0:
                # Connect to previous segment
                last_coord = all_coordinates[-1]
                first_coord = segment_coords[0]
                
                # Add intermediate point
                mid_lat = (last_coord[0] + first_coord[0]) / 2
                mid_lon = (last_coord[1] + first_coord[1]) / 2
                all_coordinates.append([mid_lat, mid_lon])
            
            all_coordinates.extend(segment_coords)
            
            # Safely get segment values
            energy_wh_per_km = seg.get('final_energy_wh_per_km', 150) if hasattr(seg, 'get') else 150
            speed_limit = seg.get('speed_limit_kmh', 50) if hasattr(seg, 'get') else 50
            road_type = seg.get('road_type', 'unknown') if hasattr(seg, 'get') else 'unknown'
            traffic_density = seg.get('traffic_density', 0.3) if hasattr(seg, 'get') else 0.3
            weather_condition = seg.get('weather_condition', 'clear') if hasattr(seg, 'get') else 'clear'
            
            route_data['route_segments'].append({
                'segment_id': i,
                'distance_km': round(seg.geometry.length * 111, 3),
                'energy_wh_per_km': round(energy_wh_per_km, 2),
                'speed_limit': speed_limit,
                'road_type': road_type,
                'segment_energy_wh': round(energy_wh_per_km * seg.geometry.length * 111, 2),
                'estimated_time_minutes': round((seg.geometry.length * 111) / (speed_limit / 60), 1),
                'traffic_density': traffic_density,
                'weather_condition': weather_condition,
                'coordinates': segment_coords
            })
        
        # Add continuous path coordinates to route data
        route_data['continuous_path'] = all_coordinates
        
        logger.info(f"Generated route with {len(all_coordinates)} coordinates")
        logger.info(f"First few coordinates: {all_coordinates[:3]}")
        logger.info(f"Last few coordinates: {all_coordinates[-3:]}")
        
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

def _geocode_address(address):
    """Convert address to coordinates (simplified for Berlin)"""
    # Simplified geocoding for Berlin landmarks (restricted to small area)
    berlin_landmarks = {
        # Points within our small bounds (52.5195-52.5205, 13.411-13.412)
        'mitte center': (52.520, 13.4115),
        'alexanderplatz': (52.522, 13.413),
        'museum island': (52.521, 13.397),
        'museumsinsel': (52.521, 13.397),
        'brandenburg gate': (52.516, 13.377),
        'brandenburger tor': (52.516, 13.377),
        'reichstag': (52.518, 13.376),
        'victory column': (52.514, 13.350),
        'siegessäule': (52.514, 13.350),
        'holocaust memorial': (52.517, 13.378),
        'memorial to the murdered jews': (52.517, 13.378),
        'gendarmenmarkt': (52.514, 13.393),
        'unter den linden': (52.517, 13.389),
        'friedrichstraße': (52.520, 13.387),
        'hackescher markt': (52.523, 13.402),
        'rosenthaler platz': (52.527, 13.407),
        'weinmeisterstraße': (52.525, 13.405),
        'jannowitzbrücke': (52.515, 13.420),
        'ostbahnhof': (52.510, 13.434),
        'friedrichshain': (52.515, 13.454),
        'kreuzberg': (52.497, 13.388),
        'prenzlauer berg': (52.535, 13.420),
        'wedding': (52.545, 13.355),
        'moabit': (52.525, 13.340),
        'tiergarten': (52.514, 13.350),
        'schöneberg': (52.483, 13.355),
        'wilmersdorf': (52.490, 13.320),
        'charlottenburg': (52.520, 13.296),
        'charlottenburg palace': (52.520, 13.296),
        'schloss charlottenburg': (52.520, 13.296),
        'berlin hauptbahnhof': (52.525, 13.369),
        'hauptbahnhof': (52.525, 13.369),
        'potsdamer platz': (52.509, 13.375),
        'checkpoint charlie': (52.507, 13.390),
        'kaiser wilhelm memorial church': (52.505, 13.335),
        'gedächtniskirche': (52.505, 13.335),
        'kurfürstendamm': (52.504, 13.327),
        'zoologischer garten': (52.507, 13.337),
        'berlin zoo': (52.507, 13.337),
        'east side gallery': (52.505, 13.443),
        'berlin wall memorial': (52.535, 13.389),
        'kottbusser tor': (52.499, 13.419),
        'hermannplatz': (52.483, 13.424),
        'warschauer straße': (52.500, 13.450),
        'frankfurter tor': (52.515, 13.454),
        'neukölln': (52.483, 13.424),
        'spandau': (52.535, 13.200),
        'reinickendorf': (52.590, 13.320),
        'pankow': (52.570, 13.410),
        'lichtenberg': (52.520, 13.480),
        'marzahn': (52.545, 13.545),
        'hohenschönhausen': (52.545, 13.480),
        'treptow': (52.489, 13.456),
        'köpenick': (52.445, 13.575),
        'steglitz': (52.455, 13.320),
        'zehlendorf': (52.430, 13.230),
        'tempelhof': (52.473, 13.404),
        'tempelhof airport': (52.473, 13.404),
        'flughafen tempelhof': (52.473, 13.404),
    }
    
    # Normalize address for matching
    normalized_address = address.lower().strip()
    
    # Try exact match first
    if normalized_address in berlin_landmarks:
        return berlin_landmarks[normalized_address]
    
    # Try partial matches
    for landmark, coords in berlin_landmarks.items():
        if landmark in normalized_address or normalized_address in landmark:
            return coords
    
    # If no match found, return a default location in Berlin center
    logger.warning(f"Could not geocode address: {address}, using default location")
    return (52.520, 13.405)  # Berlin center

def _find_closest_segment(road_network, coords):
    """Find the closest road segment to given coordinates"""
    
    # coords is (lat, lon) from geocoding, but Point expects (lon, lat)
    point = Point(coords[1], coords[0])  # Convert to (lon, lat)
    min_distance = float('inf')
    closest_segment = None
    
    for idx, segment in road_network.iterrows():
        distance = point.distance(segment.geometry)
        if distance < min_distance:
            min_distance = distance
            closest_segment = segment
    
    return closest_segment

def _calculate_route(road_network, origin_segment, dest_segment):
    """Calculate route between origin and destination segments"""
    # Create a continuous path using interpolation between origin and destination
    
    origin_point = origin_segment.geometry.centroid
    dest_point = dest_segment.geometry.centroid
    
    # Calculate distance between origin and destination
    distance_km = origin_point.distance(dest_point) * 111  # Convert to km
    
    # Create a continuous path with interpolated points
    route_segments = []
    
    # Add origin segment
    route_segments.append(origin_segment)
    
    # Calculate number of intermediate points based on distance
    num_intermediate_points = max(3, min(8, int(distance_km / 2)))
    
    # Create interpolated route points
    for i in range(1, num_intermediate_points + 1):
        # Interpolate between origin and destination
        t = i / (num_intermediate_points + 1)
        interpolated_lon = origin_point.x + t * (dest_point.x - origin_point.x)
        interpolated_lat = origin_point.y + t * (dest_point.y - origin_point.y)
        
        # Find the closest road segment to this interpolated point
        interpolated_point = Point(interpolated_lon, interpolated_lat)
        closest_segment = None
        min_distance = float('inf')
        
        for idx, segment in road_network.iterrows():
            distance = interpolated_point.distance(segment.geometry)
            if distance < min_distance:
                min_distance = distance
                closest_segment = segment
        
        if closest_segment is not None and closest_segment.name not in [seg.name for seg in route_segments]:
            route_segments.append(closest_segment)
    
    # Add destination segment
    route_segments.append(dest_segment)
    
    # Ensure we have at least 5 segments for a good visualization
    segment_indices = set()
    for seg in route_segments:
        segment_indices.add(seg.name)
    
    # If we don't have enough segments, add some nearby segments
    while len(route_segments) < 5:
        # Find segments near the middle of our route
        if len(route_segments) > 0:
            mid_point = route_segments[len(route_segments)//2].geometry.centroid
        else:
            mid_point = origin_point
        
        # Find closest segment to mid point that we haven't used
        closest_to_mid = None
        min_distance = float('inf')
        
        for idx, segment in road_network.iterrows():
            if segment.name not in segment_indices:
                distance = mid_point.distance(segment.geometry.centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_to_mid = segment
        
        if closest_to_mid is not None:
            route_segments.append(closest_to_mid)
            segment_indices.add(closest_to_mid.name)
        else:
            break
    
    return route_segments

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
            # Add missing EV features that the model expects
            'ev_aerodynamic_coefficient': 0.3,  # Default value
            'ev_elevation_impact_factor': 0.1,  # Default value
            'ev_regenerative_braking_efficiency': 0.3,  # Default value
            'ev_regenerative_braking_factor': 0.3,  # Default value
            'ev_temperature_impact_factor': 0.15,  # Default value
            'ev_traffic_impact_factor': 0.25,  # Default value
        }])
        
        # Make prediction
        try:
            prediction = ml_pipeline.predict_energy_consumption(input_data)
            energy_consumption = round(float(prediction[0]), 2)
        except Exception as e:
            logger.warning(f"ML prediction failed, using fallback calculation: {e}")
            # Fallback: Simple energy calculation
            base_energy = data.get('base_efficiency', 150)
            speed_factor = (data.get('speed_limit_kmh', 50) / 50) ** 1.5
            temp_factor = 1.0 + (data.get('temperature', 20) - 20) * 0.01
            traffic_factor = 1.0 + data.get('traffic_density', 0.3) * 0.2
            elevation_factor = 1.0 + abs(data.get('elevation_gradient', 0.0)) * 0.1
            
            energy_consumption = round(base_energy * speed_factor * temp_factor * traffic_factor * elevation_factor, 2)
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'energy_consumption_wh_per_km': energy_consumption,
                'confidence': 'high' if ml_pipeline.trained_models and ml_pipeline.trained_models['comparison']['best_score'] > 0.7 else 'medium',
                'model_used': ml_pipeline.trained_models['comparison']['best_model'] if ml_pipeline.trained_models else 'fallback',
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
            # Handle different geometry types (Point vs Polygon)
            if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                # Point geometry
                lat = float(row.geometry.y)
                lon = float(row.geometry.x)
            elif hasattr(row.geometry, 'centroid'):
                # Polygon geometry - use centroid
                lat = float(row.geometry.centroid.y)
                lon = float(row.geometry.centroid.x)
            else:
                # Fallback - skip this station
                continue
            
            # Handle NaN values in name field
            station_name = row.get('name', f'Station {idx}')
            if pd.isna(station_name):
                station_name = f'Charging Station {idx}'
            
            station = {
                'id': idx,
                'latitude': lat,
                'longitude': lon,
                'name': station_name,
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