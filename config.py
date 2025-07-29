"""
Configuration settings for EV Route Optimization System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class"""
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/ev_routing')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'ev_routing')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    
    # API Keys
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
    
    # Berlin-specific coordinates (bounding box) - Small central Berlin area
    BERLIN_BOUNDS = {
        'north': 52.5205,
        'south': 52.5195,
        'east': 13.412,
        'west': 13.411
    }
    
    # EV Parameters
    EV_PARAMS = {
        'battery_capacity_kwh': 75.0,  # Tesla Model 3 Long Range
        'efficiency_wh_per_km': 150.0,  # Base efficiency
        'regenerative_braking_factor': 0.3,
        'temperature_impact_factor': 0.15,  # 15% efficiency loss in cold
        'traffic_impact_factor': 0.25,  # 25% efficiency loss in heavy traffic
        'elevation_impact_factor': 0.1,  # 10% efficiency change per 100m elevation
    }
    
    # Weather impact factors
    WEATHER_IMPACT = {
        'temperature': {
            'cold_threshold': 5,  # Celsius
            'hot_threshold': 30,  # Celsius
            'cold_penalty': 0.2,  # 20% efficiency loss
            'hot_penalty': 0.1,   # 10% efficiency loss
        },
        'precipitation': {
            'rain_penalty': 0.05,  # 5% efficiency loss
            'snow_penalty': 0.15,  # 15% efficiency loss
        },
        'wind': {
            'headwind_penalty': 0.08,  # 8% efficiency loss per 10 m/s
            'tailwind_benefit': 0.05,  # 5% efficiency gain per 10 m/s
        }
    }
    
    # Traffic impact factors
    TRAFFIC_IMPACT = {
        'congestion_levels': {
            'low': 0.05,      # 5% efficiency loss
            'medium': 0.15,   # 15% efficiency loss
            'high': 0.25,     # 25% efficiency loss
            'severe': 0.35,   # 35% efficiency loss
        }
    }
    
    # Charging station parameters
    CHARGING_STATIONS = {
        'fast_charging_power': 150,  # kW
        'standard_charging_power': 22,  # kW
        'slow_charging_power': 7,    # kW
        'min_battery_threshold': 0.1,  # 10% battery remaining
        'optimal_charging_range': 0.2,  # 20-80% battery range
    }
    
    # Machine Learning Model Parameters
    ML_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'neural_network': {
            'layers': [64, 32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
        }
    }
    
    # System Performance
    PERFORMANCE = {
        'max_route_length_km': 100,  # Maximum route length to process
        'max_processing_time': 30,  # Maximum processing time in seconds
        'batch_size': 1000,  # Batch size for data processing
        'use_synthetic_data': True,  # Skip OSM downloads, use synthetic data for speed
        'osm_timeout_seconds': 10,  # Timeout for OSM queries
    }
    
    # API Rate Limits
    RATE_LIMITS = {
        'openweather': 60,  # calls per minute
        'tomtom': 100,      # calls per minute
        'google_maps': 100, # calls per minute
    } 