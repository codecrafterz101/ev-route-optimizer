"""
Weather Data Collector for EV Route Optimization
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

class WeatherCollector:
    """Collects and processes weather data for EV energy consumption modeling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.OPENWEATHER_API_KEY
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.berlin_coords = (52.5200, 13.4050)  # Berlin center coordinates
        
    def get_current_weather(self, lat: float = None, lon: float = None) -> Dict:
        """
        Get current weather conditions for a location
        
        Args:
            lat: Latitude (defaults to Berlin center)
            lon: Longitude (defaults to Berlin center)
            
        Returns:
            Dictionary with weather data and energy impact factors
        """
        if lat is None or lon is None:
            lat, lon = self.berlin_coords
            
        logger.info(f"Fetching current weather for coordinates: {lat}, {lon}")
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius and m/s
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            processed_data = self._process_weather_data(weather_data)
            
            logger.info(f"Successfully fetched weather data: {processed_data['description']}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_default_weather_data()
        except Exception as e:
            logger.error(f"Unexpected error processing weather data: {e}")
            return self._get_default_weather_data()
    
    def get_weather_forecast(self, lat: float = None, lon: float = None, days: int = 5) -> List[Dict]:
        """
        Get weather forecast for a location
        
        Args:
            lat: Latitude (defaults to Berlin center)
            lon: Longitude (defaults to Berlin center)
            days: Number of days to forecast (max 5)
            
        Returns:
            List of weather data dictionaries
        """
        if lat is None or lon is None:
            lat, lon = self.berlin_coords
            
        logger.info(f"Fetching {days}-day weather forecast for coordinates: {lat}, {lon}")
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40)  # 8 forecasts per day, max 40
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            processed_forecast = []
            
            for item in forecast_data['list']:
                processed_item = self._process_weather_data(item)
                processed_forecast.append(processed_item)
            
            logger.info(f"Successfully fetched {len(processed_forecast)} forecast entries")
            return processed_forecast
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return [self._get_default_weather_data() for _ in range(days)]
        except Exception as e:
            logger.error(f"Unexpected error processing weather forecast: {e}")
            return [self._get_default_weather_data() for _ in range(days)]
    
    def _process_weather_data(self, weather_data: Dict) -> Dict:
        """Process raw weather data and add energy impact factors"""
        processed = {
            'timestamp': datetime.fromtimestamp(weather_data['dt']),
            'temperature_celsius': weather_data['main']['temp'],
            'feels_like_celsius': weather_data['main']['feels_like'],
            'humidity_percent': weather_data['main']['humidity'],
            'pressure_hpa': weather_data['main']['pressure'],
            'wind_speed_ms': weather_data['wind']['speed'],
            'wind_direction_degrees': weather_data['wind'].get('deg', 0),
            'description': weather_data['weather'][0]['description'],
            'weather_code': weather_data['weather'][0]['id'],
            'clouds_percent': weather_data['clouds']['all'],
            'visibility_meters': weather_data.get('visibility', 10000),
        }
        
        # Add energy impact factors
        processed.update(self._calculate_energy_impact_factors(processed))
        
        return processed
    
    def _calculate_energy_impact_factors(self, weather_data: Dict) -> Dict:
        """Calculate energy consumption impact factors based on weather conditions"""
        temp = weather_data['temperature_celsius']
        humidity = weather_data['humidity_percent']
        wind_speed = weather_data['wind_speed_ms']
        weather_code = weather_data['weather_code']
        
        # Temperature impact
        temp_impact = self._calculate_temperature_impact(temp)
        
        # Precipitation impact
        precip_impact = self._calculate_precipitation_impact(weather_code)
        
        # Wind impact
        wind_impact = self._calculate_wind_impact(wind_speed)
        
        # Combined weather impact
        total_weather_impact = temp_impact + precip_impact + wind_impact
        
        return {
            'temperature_impact_factor': temp_impact,
            'precipitation_impact_factor': precip_impact,
            'wind_impact_factor': wind_impact,
            'total_weather_impact_factor': total_weather_impact,
            'energy_efficiency_modifier': 1.0 + total_weather_impact
        }
    
    def _calculate_temperature_impact(self, temperature: float) -> float:
        """Calculate energy consumption impact from temperature"""
        config = self.config.WEATHER_IMPACT['temperature']
        
        if temperature < config['cold_threshold']:
            # Cold weather penalty
            cold_factor = (config['cold_threshold'] - temperature) / config['cold_threshold']
            return config['cold_penalty'] * cold_factor
        elif temperature > config['hot_threshold']:
            # Hot weather penalty
            hot_factor = (temperature - config['hot_threshold']) / config['hot_threshold']
            return config['hot_penalty'] * hot_factor
        else:
            # Optimal temperature range
            return 0.0
    
    def _calculate_precipitation_impact(self, weather_code: int) -> float:
        """Calculate energy consumption impact from precipitation"""
        config = self.config.WEATHER_IMPACT['precipitation']
        
        # Weather codes: 200-531 for rain, 600-622 for snow
        if 200 <= weather_code <= 531:
            return config['rain_penalty']
        elif 600 <= weather_code <= 622:
            return config['snow_penalty']
        else:
            return 0.0
    
    def _calculate_wind_impact(self, wind_speed: float) -> float:
        """Calculate energy consumption impact from wind"""
        config = self.config.WEATHER_IMPACT['wind']
        
        # Simplified wind impact calculation
        # In practice, wind direction relative to travel direction would be considered
        wind_factor = wind_speed / 10.0  # Normalize to 10 m/s
        
        # Assume headwind for conservative estimate
        return config['headwind_penalty'] * wind_factor
    
    def _get_default_weather_data(self) -> Dict:
        """Return default weather data when API is unavailable"""
        return {
            'timestamp': datetime.now(),
            'temperature_celsius': 15.0,
            'feels_like_celsius': 15.0,
            'humidity_percent': 60.0,
            'pressure_hpa': 1013.0,
            'wind_speed_ms': 5.0,
            'wind_direction_degrees': 0,
            'description': 'Unknown',
            'weather_code': 800,
            'clouds_percent': 0,
            'visibility_meters': 10000,
            'temperature_impact_factor': 0.0,
            'precipitation_impact_factor': 0.0,
            'wind_impact_factor': 0.0,
            'total_weather_impact_factor': 0.0,
            'energy_efficiency_modifier': 1.0
        }
    
    def get_weather_grid(self, bounds: Dict) -> pd.DataFrame:
        """
        Get weather data for a grid of points within bounds
        
        Args:
            bounds: Dictionary with north, south, east, west coordinates
            
        Returns:
            DataFrame with weather data for grid points
        """
        logger.info("Creating weather grid for Berlin area...")
        
        # Create a grid of points
        lat_points = np.linspace(bounds['south'], bounds['north'], 5)
        lon_points = np.linspace(bounds['west'], bounds['east'], 5)
        
        grid_data = []
        
        for lat in lat_points:
            for lon in lon_points:
                weather = self.get_current_weather(lat, lon)
                weather['latitude'] = lat
                weather['longitude'] = lon
                grid_data.append(weather)
        
        return pd.DataFrame(grid_data)
    
    def get_historical_weather(self, lat: float, lon: float, days_back: int = 30) -> List[Dict]:
        """
        Get historical weather data (requires OpenWeatherMap One Call API)
        
        Args:
            lat: Latitude
            lon: Longitude
            days_back: Number of days to look back
            
        Returns:
            List of historical weather data
        """
        logger.info(f"Fetching historical weather data for {days_back} days...")
        
        try:
            # Note: This requires the One Call API which is paid
            # For now, return simulated historical data
            historical_data = []
            
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                # Simulate historical weather based on seasonal patterns
                temp = 15 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                historical_data.append({
                    'date': date,
                    'temperature_celsius': temp,
                    'humidity_percent': 60 + np.random.normal(0, 10),
                    'wind_speed_ms': 5 + np.random.exponential(2),
                    'precipitation_mm': np.random.exponential(2),
                    'energy_efficiency_modifier': 1.0 + np.random.normal(0, 0.1)
                })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return [] 