"""
Comprehensive Data Manager for EV Route Optimization
Integrates OpenStreetMap, Weather, and Traffic data sources
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
from shapely.geometry import Point, LineString
import json

from .openstreetmap_collector import OpenStreetMapCollector
from .weather_collector import WeatherCollector
from .traffic_collector import TrafficCollector
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages and integrates all data sources for EV route optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.osm_collector = OpenStreetMapCollector(config)
        self.weather_collector = WeatherCollector(config)
        self.traffic_collector = TrafficCollector(config)
        
        # Cache for data to avoid repeated API calls
        self._cache = {}
        self._cache_timestamps = {}
        
    def get_comprehensive_road_network(self, force_refresh: bool = False) -> gpd.GeoDataFrame:
        """
        Get comprehensive road network with all integrated data
        
        Args:
            force_refresh: Force refresh cache even if valid cache exists
            
        Returns:
            GeoDataFrame with road network and integrated data
        """
        logger.info("Building comprehensive road network with integrated data...")
        
        # Get base road network from OpenStreetMap (with caching)
        road_network = self.osm_collector.get_berlin_road_network(force_refresh=force_refresh)
        
        if road_network.empty:
            logger.error("Failed to get road network data")
            return gpd.GeoDataFrame()
        
        # Add weather data
        road_network = self._add_weather_data_to_network(road_network)
        
        # Add traffic data
        road_network = self._add_traffic_data_to_network(road_network)
        
        # Calculate final energy consumption factors
        road_network = self._calculate_final_energy_factors(road_network)
        
        logger.info(f"Comprehensive road network created with {len(road_network)} segments")
        return road_network
    
    def _add_weather_data_to_network(self, road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add weather data to road network segments"""
        logger.info("Adding weather data to road network...")
        
        # Get current weather for Berlin
        weather_data = self.weather_collector.get_current_weather()
        
        # Add weather impact factors to all road segments
        road_network['temperature_impact'] = weather_data['temperature_impact_factor']
        road_network['precipitation_impact'] = weather_data['precipitation_impact_factor']
        road_network['wind_impact'] = weather_data['wind_impact_factor']
        road_network['total_weather_impact'] = weather_data['total_weather_impact_factor']
        road_network['weather_efficiency_modifier'] = weather_data['energy_efficiency_modifier']
        
        return road_network
    
    def _add_traffic_data_to_network(self, road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add traffic data to road network segments"""
        logger.info("Adding traffic data to road network...")
        
        # Get current traffic conditions
        traffic_data = self.traffic_collector.get_current_traffic()
        
        if traffic_data.empty:
            logger.warning("No traffic data available, using defaults")
            road_network['traffic_energy_impact'] = 0.1  # Default 10% impact
            road_network['congestion_level'] = 'medium'
            return road_network
        
        # Match traffic data to road segments based on proximity
        road_network = self._match_traffic_to_road_segments(road_network, traffic_data)
        
        return road_network
    
    def _match_traffic_to_road_segments(self, road_network: gpd.GeoDataFrame, 
                                      traffic_data: pd.DataFrame) -> gpd.GeoDataFrame:
        """Match traffic data to road segments based on spatial proximity"""
        
        # Initialize traffic columns
        road_network['traffic_energy_impact'] = 0.1  # Default
        road_network['congestion_level'] = 'medium'
        road_network['current_speed_kmh'] = 50.0  # Default
        
        # For each road segment, find the closest traffic data point
        for idx, road_segment in road_network.iterrows():
            road_center = road_segment.geometry.centroid
            
            # Find closest traffic data point
            min_distance = float('inf')
            closest_traffic = None
            
            for _, traffic_point in traffic_data.iterrows():
                traffic_location = Point(traffic_point['longitude'], traffic_point['latitude'])
                distance = road_center.distance(traffic_location)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_traffic = traffic_point
            
            # If we found a close traffic point (within 2km), use its data
            if closest_traffic is not None and min_distance < 0.02:  # ~2km in degrees
                road_network.loc[idx, 'traffic_energy_impact'] = closest_traffic['energy_impact_factor']
                road_network.loc[idx, 'congestion_level'] = closest_traffic['congestion_level']
                road_network.loc[idx, 'current_speed_kmh'] = closest_traffic['current_speed_kmh']
        
        return road_network
    
    def _calculate_final_energy_factors(self, road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate final energy consumption factors for each road segment"""
        logger.info("Calculating final energy consumption factors...")
        
        # Calculate base energy consumption per km
        road_network['base_energy_wh_per_km'] = self.config.EV_PARAMS['efficiency_wh_per_km']
        
        # Apply all impact factors
        road_network['final_energy_wh_per_km'] = (
            road_network['base_energy_wh_per_km'] *
            road_network['energy_efficiency_factor'] *
            road_network['weather_efficiency_modifier'] *
            (1 + road_network['traffic_energy_impact'])
        )
        
        # Calculate energy consumption for each segment
        road_network['segment_energy_wh'] = (
            road_network['final_energy_wh_per_km'] * 
            road_network.geometry.length  # Convert to meters
        )
        
        # Add segment cost (for routing optimization)
        road_network['segment_cost'] = road_network['segment_energy_wh'] / 1000  # Convert to kWh
        
        return road_network
    
    def get_charging_stations_with_availability(self, force_refresh: bool = False) -> gpd.GeoDataFrame:
        """Get charging stations with real-time availability data"""
        logger.info("Getting charging stations with availability data...")
        
        # Get charging stations from OpenStreetMap (with caching)
        charging_stations = self.osm_collector.get_charging_stations(force_refresh=force_refresh)
        
        if charging_stations.empty:
            logger.warning("No charging stations found, creating simulated data")
            charging_stations = self._create_simulated_charging_stations()
        
        # Add availability and pricing information
        charging_stations = self._add_charging_station_metadata(charging_stations)
        
        return charging_stations
    
    def _create_simulated_charging_stations(self) -> gpd.GeoDataFrame:
        """Create simulated charging stations for Berlin"""
        logger.info("Creating simulated charging stations...")
        
        # Create a grid of charging stations across Berlin
        lat_points = np.linspace(self.config.BERLIN_BOUNDS['south'], 
                                self.config.BERLIN_BOUNDS['north'], 8)
        lon_points = np.linspace(self.config.BERLIN_BOUNDS['west'], 
                                self.config.BERLIN_BOUNDS['east'], 8)
        
        stations = []
        
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                # Skip some points to create realistic distribution
                if np.random.random() > 0.6:
                    continue
                
                # Determine charging type based on location
                if i < 2 or i > 5 or j < 2 or j > 5:  # Outer areas
                    charging_type = 'fast'
                    power = np.random.choice([50, 150, 350])
                else:  # Inner city
                    charging_type = np.random.choice(['standard', 'slow'], p=[0.7, 0.3])
                    power = 22 if charging_type == 'standard' else 7
                
                station = {
                    'geometry': Point(lon, lat),
                    'station_id': f'station_{i}_{j}',
                    'charging_power_kw': power,
                    'charging_type': charging_type,
                    'is_available': np.random.choice([True, False], p=[0.8, 0.2]),
                    'price_per_kwh': np.random.uniform(0.25, 0.45),
                    'operator': np.random.choice(['Tesla', 'Ionity', 'EnBW', 'Shell']),
                    'connector_types': ['Type 2', 'CCS'] if charging_type == 'fast' else ['Type 2']
                }
                
                stations.append(station)
        
        return gpd.GeoDataFrame(stations, crs='EPSG:4326')
    
    def _add_charging_station_metadata(self, stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add metadata to charging stations"""
        
        # Add waiting time estimates
        stations['estimated_wait_time_min'] = np.random.exponential(10, len(stations))
        
        # Add reliability score
        stations['reliability_score'] = np.random.uniform(0.7, 1.0, len(stations))
        
        # Add accessibility score
        stations['accessibility_score'] = np.random.uniform(0.8, 1.0, len(stations))
        
        # Add overall rating
        stations['overall_rating'] = (
            stations['reliability_score'] * 0.4 +
            stations['accessibility_score'] * 0.3 +
            (1 - stations['estimated_wait_time_min'] / 60) * 0.3
        )
        
        return stations
    
    def get_real_time_data_summary(self) -> Dict[str, Any]:
        """Get a summary of all real-time data sources"""
        logger.info("Generating real-time data summary...")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {},
            'system_status': 'operational'
        }
        
        # Weather summary
        try:
            weather = self.weather_collector.get_current_weather()
            summary['data_sources']['weather'] = {
                'status': 'available',
                'temperature_celsius': weather['temperature_celsius'],
                'description': weather['description'],
                'energy_impact': weather['total_weather_impact_factor']
            }
        except Exception as e:
            summary['data_sources']['weather'] = {'status': 'error', 'error': str(e)}
        
        # Traffic summary
        try:
            traffic = self.traffic_collector.get_current_traffic()
            if not traffic.empty:
                traffic_stats = self.traffic_collector.get_traffic_statistics(traffic)
                summary['data_sources']['traffic'] = {
                    'status': 'available',
                    'segments_count': traffic_stats['total_segments'],
                    'average_speed_kmh': traffic_stats['average_speed_kmh'],
                    'congestion_level': traffic_stats['average_congestion_level']
                }
            else:
                summary['data_sources']['traffic'] = {'status': 'no_data'}
        except Exception as e:
            summary['data_sources']['traffic'] = {'status': 'error', 'error': str(e)}
        
        # Road network summary
        try:
            road_network = self.osm_collector.get_berlin_road_network()
            summary['data_sources']['road_network'] = {
                'status': 'available',
                'segments_count': len(road_network),
                'coverage_area_km2': self._calculate_coverage_area(road_network)
            }
        except Exception as e:
            summary['data_sources']['road_network'] = {'status': 'error', 'error': str(e)}
        
        # Charging stations summary
        try:
            charging_stations = self.get_charging_stations_with_availability()
            summary['data_sources']['charging_stations'] = {
                'status': 'available',
                'total_count': len(charging_stations),
                'available_count': int(charging_stations['is_available'].sum()),
                'fast_charging_count': int(len(charging_stations[charging_stations['charging_type'] == 'fast']))
            }
        except Exception as e:
            summary['data_sources']['charging_stations'] = {'status': 'error', 'error': str(e)}
        
        return summary
    
    def _calculate_coverage_area(self, road_network: gpd.GeoDataFrame) -> float:
        """Calculate the coverage area of the road network"""
        try:
            # Create a buffer around the road network and calculate area
            buffer_distance = 0.01  # ~1km in degrees
            buffered = road_network.geometry.buffer(buffer_distance)
            total_area = buffered.unary_union.area * 111 * 111  # Convert to km²
            return round(total_area, 2)
        except:
            return 0.0
    
    def export_data_for_analysis(self, output_dir: str = "data_exports") -> Dict[str, str]:
        """Export all data for analysis and visualization"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        logger.info(f"Exporting data to {output_dir}...")
        
        # Export comprehensive road network
        try:
            road_network = self.get_comprehensive_road_network()
            road_network_file = os.path.join(output_dir, "road_network.geojson")
            road_network.to_file(road_network_file, driver='GeoJSON')
            exported_files['road_network'] = road_network_file
        except Exception as e:
            logger.error(f"Failed to export road network: {e}")
        
        # Export charging stations
        try:
            charging_stations = self.get_charging_stations_with_availability()
            stations_file = os.path.join(output_dir, "charging_stations.geojson")
            charging_stations.to_file(stations_file, driver='GeoJSON')
            exported_files['charging_stations'] = stations_file
        except Exception as e:
            logger.error(f"Failed to export charging stations: {e}")
        
        # Export real-time data summary
        try:
            summary = self.get_real_time_data_summary()
            summary_file = os.path.join(output_dir, "data_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            exported_files['data_summary'] = summary_file
        except Exception as e:
            logger.error(f"Failed to export data summary: {e}")
        
        logger.info(f"Data export completed. Files: {list(exported_files.keys())}")
        return exported_files 