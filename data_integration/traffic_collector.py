"""
Traffic Data Collector for EV Route Optimization
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficCollector:
    """Collects and processes traffic data for EV energy consumption modeling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.TOMTOM_API_KEY
        self.base_url = "https://api.tomtom.com/traffic/services/4"
        self.berlin_bounds = config.BERLIN_BOUNDS
        
    def get_current_traffic(self, bounds: Dict = None) -> pd.DataFrame:
        """
        Get current traffic conditions for Berlin area
        
        Args:
            bounds: Bounding box coordinates (defaults to Berlin bounds)
            
        Returns:
            DataFrame with traffic data
        """
        if bounds is None:
            bounds = self.berlin_bounds
            
        logger.info("Fetching current traffic conditions for Berlin...")
        
        try:
            # TomTom Traffic Flow API
            url = f"{self.base_url}/flowSegmentData/relative0/10/json"
            params = {
                'key': self.api_key,
                'unit': 'KMPH',
                'style': 's3',
                'zoom': 10,
                'bbox': f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}"
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            traffic_data = response.json()
            processed_data = self._process_traffic_data(traffic_data)
            
            logger.info(f"Successfully fetched traffic data for {len(processed_data)} segments")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching traffic data: {e}")
            return self._get_simulated_traffic_data(bounds)
        except Exception as e:
            logger.error(f"Unexpected error processing traffic data: {e}")
            return self._get_simulated_traffic_data(bounds)
    
    def get_traffic_incidents(self, bounds: Dict = None) -> pd.DataFrame:
        """
        Get traffic incidents for Berlin area
        
        Args:
            bounds: Bounding box coordinates (defaults to Berlin bounds)
            
        Returns:
            DataFrame with incident data
        """
        if bounds is None:
            bounds = self.berlin_bounds
            
        logger.info("Fetching traffic incidents for Berlin...")
        
        try:
            # TomTom Traffic Incident API
            url = "https://api.tomtom.com/traffic/services/4/incidentDetails/s3"
            params = {
                'key': self.api_key,
                'bbox': f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}",
                'fields': '{incidents{type,geometry{coordinates},properties{iconCategory,from,to}}}'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            incident_data = response.json()
            processed_incidents = self._process_incident_data(incident_data)
            
            logger.info(f"Successfully fetched {len(processed_incidents)} traffic incidents")
            return processed_incidents
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching traffic incidents: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error processing incident data: {e}")
            return pd.DataFrame()
    
    def _process_traffic_data(self, traffic_data: Dict) -> pd.DataFrame:
        """Process raw traffic flow data"""
        processed_segments = []
        
        try:
            flow_segments = traffic_data.get('flowSegmentData', [])
            
            for segment in flow_segments:
                coordinates = segment.get('coordinates', {})
                flow_data = segment.get('flow', {})
                
                processed_segment = {
                    'segment_id': segment.get('id', ''),
                    'latitude': coordinates.get('coordinate', [0, 0])[0],
                    'longitude': coordinates.get('coordinate', [0, 0])[1],
                    'current_speed_kmh': flow_data.get('currentSpeed', 0),
                    'free_flow_speed_kmh': flow_data.get('freeFlowSpeed', 0),
                    'confidence': flow_data.get('confidence', 0),
                    'congestion_level': self._calculate_congestion_level(flow_data),
                    'energy_impact_factor': self._calculate_traffic_energy_impact(flow_data),
                    'timestamp': datetime.now()
                }
                
                processed_segments.append(processed_segment)
            
            return pd.DataFrame(processed_segments)
            
        except Exception as e:
            logger.error(f"Error processing traffic data: {e}")
            return pd.DataFrame()
    
    def _process_incident_data(self, incident_data: Dict) -> pd.DataFrame:
        """Process traffic incident data"""
        processed_incidents = []
        
        try:
            incidents = incident_data.get('incidents', [])
            
            for incident in incidents:
                properties = incident.get('properties', {})
                geometry = incident.get('geometry', {})
                coordinates = geometry.get('coordinates', [0, 0])
                
                processed_incident = {
                    'incident_id': incident.get('id', ''),
                    'latitude': coordinates[1] if len(coordinates) > 1 else 0,
                    'longitude': coordinates[0] if len(coordinates) > 0 else 0,
                    'incident_type': properties.get('iconCategory', 'unknown'),
                    'from_location': properties.get('from', ''),
                    'to_location': properties.get('to', ''),
                    'severity': self._classify_incident_severity(properties.get('iconCategory', '')),
                    'energy_impact_factor': self._calculate_incident_energy_impact(properties.get('iconCategory', '')),
                    'timestamp': datetime.now()
                }
                
                processed_incidents.append(processed_incident)
            
            return pd.DataFrame(processed_incidents)
            
        except Exception as e:
            logger.error(f"Error processing incident data: {e}")
            return pd.DataFrame()
    
    def _calculate_congestion_level(self, flow_data: Dict) -> str:
        """Calculate congestion level based on traffic flow data"""
        current_speed = flow_data.get('currentSpeed', 0)
        free_flow_speed = flow_data.get('freeFlowSpeed', 1)
        
        if free_flow_speed == 0:
            return 'unknown'
        
        speed_ratio = current_speed / free_flow_speed
        
        if speed_ratio >= 0.9:
            return 'low'
        elif speed_ratio >= 0.7:
            return 'medium'
        elif speed_ratio >= 0.5:
            return 'high'
        else:
            return 'severe'
    
    def _calculate_traffic_energy_impact(self, flow_data: Dict) -> float:
        """Calculate energy consumption impact from traffic conditions"""
        congestion_level = self._calculate_congestion_level(flow_data)
        config = self.config.TRAFFIC_IMPACT['congestion_levels']
        
        return config.get(congestion_level, 0.0)
    
    def _classify_incident_severity(self, incident_type: str) -> str:
        """Classify incident severity based on type"""
        high_severity = ['accident', 'road_closed', 'construction']
        medium_severity = ['congestion', 'slow_traffic', 'weather']
        low_severity = ['information', 'minor_delay']
        
        incident_type = incident_type.lower()
        
        if any(high in incident_type for high in high_severity):
            return 'high'
        elif any(medium in incident_type for medium in medium_severity):
            return 'medium'
        elif any(low in incident_type for low in low_severity):
            return 'low'
        else:
            return 'unknown'
    
    def _calculate_incident_energy_impact(self, incident_type: str) -> float:
        """Calculate energy consumption impact from traffic incidents"""
        severity = self._classify_incident_severity(incident_type)
        
        impact_factors = {
            'high': 0.4,    # 40% efficiency loss
            'medium': 0.25, # 25% efficiency loss
            'low': 0.1,     # 10% efficiency loss
            'unknown': 0.15 # 15% efficiency loss
        }
        
        return impact_factors.get(severity, 0.15)
    
    def _get_simulated_traffic_data(self, bounds: Dict) -> pd.DataFrame:
        """Generate simulated traffic data when API is unavailable"""
        logger.info("Generating simulated traffic data...")
        
        # Create a grid of simulated traffic data
        lat_points = np.linspace(bounds['south'], bounds['north'], 10)
        lon_points = np.linspace(bounds['west'], bounds['east'], 10)
        
        simulated_data = []
        
        for lat in lat_points:
            for lon in lon_points:
                # Simulate traffic conditions based on time and location
                hour = datetime.now().hour
                
                # Rush hour simulation
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    base_speed = 30 + np.random.normal(0, 10)
                    congestion_level = np.random.choice(['medium', 'high'], p=[0.6, 0.4])
                else:
                    base_speed = 50 + np.random.normal(0, 15)
                    congestion_level = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
                
                free_flow_speed = 60 + np.random.normal(0, 5)
                current_speed = max(5, min(free_flow_speed, base_speed))
                
                simulated_data.append({
                    'segment_id': f'sim_{lat:.4f}_{lon:.4f}',
                    'latitude': lat,
                    'longitude': lon,
                    'current_speed_kmh': current_speed,
                    'free_flow_speed_kmh': free_flow_speed,
                    'confidence': 0.8,
                    'congestion_level': congestion_level,
                    'energy_impact_factor': self._calculate_traffic_energy_impact({
                        'currentSpeed': current_speed,
                        'freeFlowSpeed': free_flow_speed
                    }),
                    'timestamp': datetime.now()
                })
        
        return pd.DataFrame(simulated_data)
    
    def get_traffic_forecast(self, bounds: Dict = None, hours: int = 24) -> List[pd.DataFrame]:
        """
        Get traffic forecast for the next hours
        
        Args:
            bounds: Bounding box coordinates
            hours: Number of hours to forecast
            
        Returns:
            List of DataFrames with hourly traffic forecasts
        """
        if bounds is None:
            bounds = self.berlin_bounds
            
        logger.info(f"Generating {hours}-hour traffic forecast...")
        
        forecasts = []
        
        for hour in range(hours):
            forecast_time = datetime.now() + timedelta(hours=hour)
            
            # Simulate traffic patterns based on time of day
            if 7 <= forecast_time.hour <= 9:  # Morning rush
                base_congestion = 'high'
                speed_factor = 0.6
            elif 17 <= forecast_time.hour <= 19:  # Evening rush
                base_congestion = 'high'
                speed_factor = 0.6
            elif 22 <= forecast_time.hour or forecast_time.hour <= 5:  # Night
                base_congestion = 'low'
                speed_factor = 1.2
            else:  # Daytime
                base_congestion = 'medium'
                speed_factor = 0.9
            
            # Generate forecast data
            forecast_data = self._get_simulated_traffic_data(bounds)
            forecast_data['forecast_hour'] = hour
            forecast_data['forecast_time'] = forecast_time
            
            # Adjust speeds based on forecast
            forecast_data['current_speed_kmh'] *= speed_factor
            forecast_data['congestion_level'] = base_congestion
            
            forecasts.append(forecast_data)
        
        return forecasts
    
    def get_traffic_statistics(self, traffic_data: pd.DataFrame) -> Dict:
        """Calculate traffic statistics for analysis"""
        if traffic_data.empty:
            return {}
        
        stats = {
            'total_segments': len(traffic_data),
            'average_speed_kmh': float(traffic_data['current_speed_kmh'].mean()),
            'average_congestion_level': traffic_data['congestion_level'].mode().iloc[0] if not traffic_data['congestion_level'].mode().empty else 'unknown',
            'congestion_distribution': {k: int(v) for k, v in traffic_data['congestion_level'].value_counts().to_dict().items()},
            'average_energy_impact': float(traffic_data['energy_impact_factor'].mean()),
            'timestamp': datetime.now()
        }
        
        return stats 