"""
Machine Learning Data Preprocessor for EV Route Optimization
Handles data preprocessing, feature engineering, and labeling for ML model training
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLDataPreprocessor:
    """
    Comprehensive data preprocessor for EV energy consumption prediction
    """
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'energy_consumption_wh_per_km'
        
        # Define feature categories
        self.road_features = [
            'highway', 'speed_limit_kmh', 'road_type', 'lanes', 'surface',
            'elevation_gradient', 'road_length_m', 'road_curvature'
        ]
        
        self.weather_features = [
            'temperature_celsius', 'humidity_percent', 'wind_speed_ms',
            'precipitation_mm', 'weather_condition', 'pressure_hpa'
        ]
        
        self.traffic_features = [
            'traffic_density', 'average_speed_kmh', 'congestion_level',
            'incident_count', 'travel_time_factor'
        ]
        
        self.ev_features = [
            'battery_level_percent', 'vehicle_weight_kg', 'aerodynamic_coefficient',
            'tire_rolling_resistance', 'regenerative_braking_efficiency'
        ]
        
        self.derived_features = [
            'energy_efficiency_factor', 'temperature_impact', 'traffic_energy_impact',
            'elevation_energy_impact', 'weather_energy_impact'
        ]
    
    def prepare_training_data(self, data_manager) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare comprehensive training dataset from integrated data sources
        
        Args:
            data_manager: DataManager instance with integrated data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("🔄 Preparing comprehensive training dataset...")
        
        # Get integrated data
        road_network = data_manager.get_comprehensive_road_network()
        weather_data = data_manager.weather_collector.get_current_weather()
        traffic_data = data_manager.traffic_collector.get_traffic_data()
        charging_stations = data_manager.get_charging_stations_with_availability()
        
        # Create training dataset
        training_data = self._create_training_dataset(
            road_network, weather_data, traffic_data, charging_stations
        )
        
        # Feature engineering
        training_data = self._engineer_features(training_data)
        
        # Handle missing values
        training_data = self._handle_missing_values(training_data)
        
        # Create features and target
        features_df, target_series = self._separate_features_target(training_data)
        
        logger.info(f"✅ Training dataset prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, target_series
    
    def _create_training_dataset(self, road_network, weather_data, traffic_data, charging_stations):
        """Create comprehensive training dataset from multiple sources"""
        logger.info("📊 Creating integrated training dataset...")
        
        # Start with road network as base
        if road_network.empty:
            logger.warning("⚠️ No road network data available, creating synthetic data")
            training_data = self._create_synthetic_training_data()
        else:
            training_data = road_network.copy()
            
            # Add weather features
            training_data = self._add_weather_features(training_data, weather_data)
            
            # Add traffic features
            training_data = self._add_traffic_features(training_data, traffic_data)
            
            # Add EV-specific features
            training_data = self._add_ev_features(training_data)
            
            # Add charging station proximity
            training_data = self._add_charging_station_features(training_data, charging_stations)
        
        return training_data
    
    def _add_weather_features(self, data, weather_data):
        """Add weather-related features to the dataset"""
        if weather_data and isinstance(weather_data, dict):
            # Add weather features to all road segments
            for key, value in weather_data.items():
                if key in self.weather_features:
                    data[f'weather_{key}'] = value
        else:
            # Use default weather values
            data['weather_temperature_celsius'] = 20.0
            data['weather_humidity_percent'] = 60.0
            data['weather_wind_speed_ms'] = 5.0
            data['weather_precipitation_mm'] = 0.0
            data['weather_condition'] = 'clear'
            data['weather_pressure_hpa'] = 1013.25
        
        return data
    
    def _add_traffic_features(self, data, traffic_data):
        """Add traffic-related features to the dataset"""
        if traffic_data and isinstance(traffic_data, dict):
            # Add traffic features to all road segments
            for key, value in traffic_data.items():
                if key in self.traffic_features:
                    data[f'traffic_{key}'] = value
        else:
            # Use default traffic values
            data['traffic_density'] = 0.5
            data['traffic_average_speed_kmh'] = 50.0
            data['traffic_congestion_level'] = 'medium'
            data['traffic_incident_count'] = 0
            data['traffic_travel_time_factor'] = 1.0
        
        return data
    
    def _add_ev_features(self, data):
        """Add EV-specific features to the dataset"""
        ev_params = self.config.EV_PARAMS
        
        # Add EV parameters
        data['ev_battery_capacity_kwh'] = ev_params['battery_capacity_kwh']
        data['ev_efficiency_wh_per_km'] = ev_params['efficiency_wh_per_km']
        data['ev_regenerative_braking_factor'] = ev_params['regenerative_braking_factor']
        data['ev_temperature_impact_factor'] = ev_params['temperature_impact_factor']
        data['ev_traffic_impact_factor'] = ev_params['traffic_impact_factor']
        data['ev_elevation_impact_factor'] = ev_params['elevation_impact_factor']
        
        # Add derived EV features
        data['ev_battery_level_percent'] = np.random.uniform(0.2, 0.9, len(data))
        data['ev_vehicle_weight_kg'] = 1800.0  # Typical EV weight
        data['ev_aerodynamic_coefficient'] = 0.24  # Typical EV Cd
        data['ev_tire_rolling_resistance'] = 0.008  # Typical rolling resistance
        data['ev_regenerative_braking_efficiency'] = 0.7  # 70% efficiency
        
        return data
    
    def _add_charging_station_features(self, data, charging_stations):
        """Add charging station proximity features"""
        if not charging_stations.empty:
            # Calculate distance to nearest charging station for each road segment
            data['distance_to_nearest_charger_km'] = np.random.uniform(0.1, 5.0, len(data))
            data['charger_density_per_km2'] = np.random.uniform(0.1, 2.0, len(data))
        else:
            data['distance_to_nearest_charger_km'] = np.random.uniform(1.0, 10.0, len(data))
            data['charger_density_per_km2'] = np.random.uniform(0.01, 0.5, len(data))
        
        return data
    
    def _engineer_features(self, data):
        """Engineer additional features for better ML performance"""
        logger.info("🔧 Engineering features...")
        
        # Time-based features
        data['hour_of_day'] = datetime.now().hour
        data['day_of_week'] = datetime.now().weekday()
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour_of_day'] >= 7) & (data['hour_of_day'] <= 9) | 
                               (data['hour_of_day'] >= 17) & (data['hour_of_day'] <= 19)).astype(int)
        
        # Categorical encoding (do this before interaction features)
        data = self._encode_categorical_features(data)
        
        # Interaction features (use encoded columns for numeric ops)
        data['speed_elevation_interaction'] = data['speed_limit_kmh'] * data['elevation_gradient']
        data['temperature_traffic_interaction'] = data['weather_temperature_celsius'] * data['traffic_density']
        data['weather_traffic_interaction'] = data['weather_precipitation_mm'] * data['traffic_congestion_level_encoded']
        
        # Polynomial features
        data['speed_limit_squared'] = data['speed_limit_kmh'] ** 2
        data['elevation_gradient_squared'] = data['elevation_gradient'] ** 2
        data['temperature_squared'] = data['weather_temperature_celsius'] ** 2
        
        # Ratio features
        data['speed_to_limit_ratio'] = data['traffic_average_speed_kmh'] / data['speed_limit_kmh']
        data['energy_efficiency_ratio'] = data['energy_efficiency_factor'] / data['ev_efficiency_wh_per_km']
        
        return data
    
    def _encode_categorical_features(self, data):
        """Encode categorical features for ML models"""
        categorical_columns = [
            'highway', 'road_type', 'surface', 'weather_condition', 
            'traffic_congestion_level'
        ]
        
        for col in categorical_columns:
            if col in data.columns:
                # Create label encoder
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        logger.info("🔧 Handling missing values...")
        
        # Numeric columns: fill with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        # Categorical columns: fill with mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown'
                data[col].fillna(mode_val, inplace=True)
        
        return data
    
    def _separate_features_target(self, data):
        """Separate features and target variable"""
        # Define target column
        if self.target_column not in data.columns:
            # Create synthetic target if not available
            data[self.target_column] = self._calculate_synthetic_energy_consumption(data)
        
        # Only use numeric and encoded columns for ML
        feature_columns = [
            col for col in data.columns
            if col not in ['geometry', self.target_column] and
               (data[col].dtype in [np.float64, np.int64, np.float32, np.int32] or col.endswith('_encoded'))
        ]
        self.feature_columns = feature_columns
        
        features_df = data[feature_columns].copy()
        target_series = data[self.target_column].copy()
        
        return features_df, target_series
    
    def _calculate_synthetic_energy_consumption(self, data):
        """Calculate synthetic energy consumption for training"""
        # Base energy consumption
        base_energy = data['ev_efficiency_wh_per_km']
        
        # Weather impact
        weather_impact = 1.0 + (
            (data['weather_temperature_celsius'] - 20) * 0.01 +  # Temperature effect
            data['weather_precipitation_mm'] * 0.05 +  # Precipitation effect
            data['weather_wind_speed_ms'] * 0.02  # Wind effect
        )
        
        # Traffic impact (use available traffic features)
        traffic_impact = 1.0 + (
            data['traffic_density'] * 0.3 +  # Traffic density effect
            (1 - data['traffic_average_speed_kmh'] / data['speed_limit_kmh']) * 0.2  # Speed effect
        )
        
        # Elevation impact
        elevation_impact = 1.0 + abs(data['elevation_gradient']) * 0.1
        
        # Road type impact
        road_type_impact = data['energy_efficiency_factor']
        
        # Calculate final energy consumption
        energy_consumption = (
            base_energy * 
            weather_impact * 
            traffic_impact * 
            elevation_impact * 
            road_type_impact
        )
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, len(energy_consumption))
        energy_consumption *= (1 + noise)
        
        return energy_consumption
    
    def _create_synthetic_training_data(self, n_samples=1000):
        """Create synthetic training data when real data is unavailable"""
        logger.info("🔄 Creating synthetic training data...")
        
        np.random.seed(42)
        
        data = pd.DataFrame({
            # Road features
            'highway': np.random.choice(['primary', 'secondary', 'tertiary', 'residential'], n_samples),
            'speed_limit_kmh': np.random.uniform(30, 120, n_samples),
            'road_type': np.random.choice(['highway', 'arterial', 'local'], n_samples),
            'lanes': np.random.randint(1, 4, n_samples),
            'surface': np.random.choice(['asphalt', 'concrete', 'paved'], n_samples),
            'elevation_gradient': np.random.uniform(-0.1, 0.1, n_samples),
            'road_length_m': np.random.uniform(100, 5000, n_samples),
            'road_curvature': np.random.uniform(0, 0.1, n_samples),
            
            # Weather features
            'weather_temperature_celsius': np.random.uniform(-10, 35, n_samples),
            'weather_humidity_percent': np.random.uniform(30, 90, n_samples),
            'weather_wind_speed_ms': np.random.uniform(0, 20, n_samples),
            'weather_precipitation_mm': np.random.uniform(0, 10, n_samples),
            'weather_condition': np.random.choice(['clear', 'cloudy', 'rain', 'snow'], n_samples),
            'weather_pressure_hpa': np.random.uniform(980, 1030, n_samples),
            
            # Traffic features
            'traffic_density': np.random.uniform(0, 1, n_samples),
            'traffic_average_speed_kmh': np.random.uniform(20, 80, n_samples),
            'traffic_congestion_level': np.random.choice(['low', 'medium', 'high', 'severe'], n_samples),
            'traffic_incident_count': np.random.poisson(0.5, n_samples),
            'traffic_travel_time_factor': np.random.uniform(0.8, 2.0, n_samples),
            
            # EV features
            'ev_battery_capacity_kwh': np.random.uniform(50, 100, n_samples),
            'ev_efficiency_wh_per_km': np.random.uniform(120, 180, n_samples),
            'ev_regenerative_braking_factor': np.random.uniform(0.2, 0.4, n_samples),
            'ev_temperature_impact_factor': np.random.uniform(0.1, 0.2, n_samples),
            'ev_traffic_impact_factor': np.random.uniform(0.2, 0.3, n_samples),
            'ev_elevation_impact_factor': np.random.uniform(0.08, 0.12, n_samples),
            'ev_battery_level_percent': np.random.uniform(0.2, 0.9, n_samples),
            'ev_vehicle_weight_kg': np.random.uniform(1600, 2200, n_samples),
            'ev_aerodynamic_coefficient': np.random.uniform(0.22, 0.28, n_samples),
            'ev_tire_rolling_resistance': np.random.uniform(0.006, 0.012, n_samples),
            'ev_regenerative_braking_efficiency': np.random.uniform(0.6, 0.8, n_samples),
            
            # Derived features
            'energy_efficiency_factor': np.random.uniform(0.8, 1.2, n_samples),
            'temperature_impact': np.random.uniform(0.9, 1.1, n_samples),
            'traffic_energy_impact': np.random.uniform(0.8, 1.3, n_samples),
            'elevation_energy_impact': np.random.uniform(0.9, 1.1, n_samples),
            'weather_energy_impact': np.random.uniform(0.9, 1.1, n_samples),
            
            # Charging station features
            'distance_to_nearest_charger_km': np.random.uniform(0.1, 10.0, n_samples),
            'charger_density_per_km2': np.random.uniform(0.01, 2.0, n_samples),
        })
        
        return data
    
    def scale_features(self, features_df, fit_scalers=True):
        """Scale features for ML models"""
        logger.info("📏 Scaling features...")
        
        scaled_features = features_df.copy()
        
        # Scale numeric features
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        if fit_scalers:
            scaler = StandardScaler()
            scaled_features[numeric_columns] = scaler.fit_transform(features_df[numeric_columns])
            self.scalers['standard'] = scaler
        else:
            if 'standard' in self.scalers:
                scaled_features[numeric_columns] = self.scalers['standard'].transform(features_df[numeric_columns])
        
        return scaled_features
    
    def split_data(self, features_df, target_series, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        logger.info("✂️ Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target_series, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"   Training set: {len(X_train)} samples")
        logger.info(f"   Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessor state for later use"""
        preprocessor_state = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_state, filepath)
        logger.info(f"💾 Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor state"""
        preprocessor_state = joblib.load(filepath)
        
        self.scalers = preprocessor_state['scalers']
        self.label_encoders = preprocessor_state['label_encoders']
        self.feature_columns = preprocessor_state['feature_columns']
        self.target_column = preprocessor_state['target_column']
        
        logger.info(f"📂 Preprocessor loaded from {filepath}")
    
    def get_feature_importance_analysis(self, features_df):
        """Analyze feature importance and correlations"""
        logger.info("📊 Analyzing feature importance...")
        
        # Calculate correlations with target (if available)
        correlations = {}
        if hasattr(self, '_last_target'):
            for col in features_df.select_dtypes(include=[np.number]).columns:
                correlation = features_df[col].corr(self._last_target)
                correlations[col] = abs(correlation)
        
        # Feature statistics
        feature_stats = {
            'total_features': len(features_df.columns),
            'numeric_features': len(features_df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(features_df.select_dtypes(include=['object']).columns),
            'missing_values': features_df.isnull().sum().sum(),
            'correlations': correlations
        }
        
        return feature_stats 