# EV Route Optimizer API Documentation

## Overview

The EV Route Optimizer provides a RESTful API for energy-efficient route planning and energy consumption prediction for electric vehicles. The API integrates machine learning models with real-time data to optimize EV routes based on energy efficiency.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, no authentication is required for API access.

## Endpoints

### System Management

#### GET `/api/status`

Get the current system status and health information.

**Response:**
```json
{
  "status": "online",
  "timestamp": "2024-01-20T10:30:00.000Z",
  "data_sources": {
    "osm": "active",
    "weather": "active",
    "traffic": "active"
  },
  "ml_status": "ready",
  "system_state": {
    "initialized": true,
    "models_loaded": true,
    "last_update": "2024-01-20T10:25:00.000Z",
    "status": "ready"
  }
}
```

#### POST `/api/initialize`

Initialize the ML pipeline and load energy prediction models.

**Request:**
```json
{}
```

**Response:**
```json
{
  "status": "success",
  "message": "System initialized successfully",
  "results": {
    "best_model": "Random Forest",
    "best_score": 0.7883,
    "models_trained": 2
  }
}
```

### Route Optimization

#### POST `/api/route/optimize`

Find the most energy-efficient route between two points.

**Request:**
```json
{
  "origin": "Brandenburg Gate, Berlin",
  "destination": "Alexanderplatz, Berlin",
  "vehicle_params": {
    "type": "sedan",
    "battery_level": 80
  },
  "preferences": {
    "optimize_for": "energy",
    "avoid_highways": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "route": {
    "route_id": "route_20240120_103000",
    "origin": "Brandenburg Gate, Berlin",
    "destination": "Alexanderplatz, Berlin",
    "total_distance_km": 2.5,
    "total_energy_kwh": 0.45,
    "avg_energy_per_km": 180.0,
    "estimated_time_minutes": 8,
    "route_segments": [
      {
        "segment_id": 0,
        "distance_km": 0.5,
        "energy_wh_per_km": 175.2,
        "speed_limit": 50,
        "road_type": "arterial",
        "coordinates": [[52.5163, 13.3777], [52.5200, 13.4050]]
      }
    ]
  }
}
```

### Energy Prediction

#### POST `/api/energy/predict`

Predict energy consumption based on driving conditions.

**Request:**
```json
{
  "speed_limit_kmh": 50,
  "temperature": 20,
  "humidity": 60,
  "wind_speed": 5,
  "precipitation": 0,
  "traffic_density": 0.3,
  "traffic_speed": 45,
  "elevation_gradient": 0.0,
  "battery_capacity": 75,
  "base_efficiency": 150,
  "battery_level": 0.8,
  "vehicle_weight": 1800,
  "road_type": "arterial"
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "energy_consumption_wh_per_km": 165.4,
    "confidence": "high",
    "model_used": "Random Forest",
    "factors": {
      "base_efficiency": 150,
      "weather_impact": 0.000,
      "traffic_impact": 0.060,
      "elevation_impact": 0.000
    }
  }
}
```

### Data Retrieval

#### GET `/api/data/road-network`

Get road network data for visualization.

**Response:**
```json
{
  "status": "success",
  "data": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "LineString",
          "coordinates": [[13.3777, 52.5163], [13.4050, 52.5200]]
        },
        "properties": {
          "segment_id": 0,
          "speed_limit": 50,
          "road_type": "arterial",
          "energy_per_km": 175.2,
          "efficiency_factor": 1.05
        }
      }
    ]
  },
  "summary": {
    "total_segments": 1250,
    "avg_energy_per_km": 168.5,
    "coverage_area_km2": 891.8
  }
}
```

#### GET `/api/data/charging-stations`

Get charging stations data with availability information.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": 0,
      "latitude": 52.5200,
      "longitude": 13.4050,
      "name": "Alexanderplatz Charging Hub",
      "charging_type": "fast",
      "power_kw": 150,
      "is_available": true,
      "price_per_kwh": 0.45,
      "connector_types": ["CCS", "Type2"],
      "network": "Ionity"
    }
  ],
  "summary": {
    "total_stations": 245,
    "available_stations": 198,
    "fast_charging_stations": 87,
    "avg_price_per_kwh": 0.42
  }
}
```

#### GET `/api/weather/current`

Get current weather data for the Berlin area.

**Response:**
```json
{
  "status": "success",
  "data": {
    "temperature_celsius": 18.5,
    "humidity_percent": 65,
    "wind_speed_ms": 3.2,
    "precipitation_mm": 0.0,
    "weather_condition": "clear",
    "timestamp": "2024-01-20T10:30:00.000Z"
  }
}
```

### Machine Learning

#### GET `/api/ml/models`

Get information about trained ML models.

**Response:**
```json
{
  "status": "success",
  "data": {
    "models_trained": 2,
    "best_model": "Random Forest",
    "best_r2_score": 0.7883,
    "model_comparison": [
      {
        "model_name": "Random Forest",
        "r2_score": 0.7883,
        "mae": 22.3566,
        "rmse": 29.4236,
        "mape": 9.01
      },
      {
        "model_name": "Neural Network",
        "r2_score": -0.0102,
        "mae": 50.5603,
        "rmse": 64.2769,
        "mape": 18.60
      }
    ],
    "feature_importance": [
      {
        "feature": "ev_efficiency_wh_per_km",
        "importance": 0.1921
      },
      {
        "feature": "energy_efficiency_factor",
        "importance": 0.0847
      }
    ]
  }
}
```

### Data Export

#### GET `/api/export/data`

Export system data for analysis.

**Response:**
```json
{
  "status": "success",
  "exported_files": [
    "demo_exports/road_network_20240120_103000.geojson",
    "demo_exports/charging_stations_20240120_103000.geojson",
    "demo_exports/weather_data_20240120_103000.json"
  ],
  "message": "Data exported successfully"
}
```

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "status": "error",
  "error": "Error description"
}
```

### Common HTTP Status Codes

- `200` - Success
- `400` - Bad Request (missing or invalid parameters)
- `404` - Not Found (resource not available)
- `500` - Internal Server Error

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

## Data Formats

### Coordinates

All geographic coordinates are provided in WGS84 format (latitude, longitude).

### Energy Consumption

Energy consumption values are provided in Wh/km (Watt-hours per kilometer).

### Distances

All distances are provided in kilometers.

### Timestamps

All timestamps are provided in ISO 8601 format with UTC timezone.

## Usage Examples

### Python

```python
import requests

# Initialize the system
response = requests.post('http://localhost:5000/api/initialize')
print(response.json())

# Predict energy consumption
data = {
    "speed_limit_kmh": 50,
    "temperature": 20,
    "traffic_density": 0.3,
    "elevation_gradient": 0.0
}
response = requests.post('http://localhost:5000/api/energy/predict', json=data)
print(response.json())

# Optimize route
route_data = {
    "origin": "Brandenburg Gate, Berlin",
    "destination": "Alexanderplatz, Berlin",
    "vehicle_params": {
        "type": "sedan",
        "battery_level": 80
    }
}
response = requests.post('http://localhost:5000/api/route/optimize', json=route_data)
print(response.json())
```

### JavaScript

```javascript
// Initialize the system
fetch('/api/initialize', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    }
})
.then(response => response.json())
.then(data => console.log(data));

// Predict energy consumption
const energyData = {
    speed_limit_kmh: 50,
    temperature: 20,
    traffic_density: 0.3,
    elevation_gradient: 0.0
};

fetch('/api/energy/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(energyData)
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
# Initialize the system
curl -X POST http://localhost:5000/api/initialize \
  -H "Content-Type: application/json"

# Predict energy consumption
curl -X POST http://localhost:5000/api/energy/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speed_limit_kmh": 50,
    "temperature": 20,
    "traffic_density": 0.3,
    "elevation_gradient": 0.0
  }'

# Get system status
curl http://localhost:5000/api/status
```

## Development Notes

- The system requires initialization before making predictions or route optimizations
- All ML models are loaded in memory for fast inference
- The system uses synthetic data for demonstration purposes
- Real-time data integration is available for weather and traffic information
- Map data is cached for improved performance

## Support

For technical support or feature requests, please refer to the project documentation or create an issue in the project repository. 