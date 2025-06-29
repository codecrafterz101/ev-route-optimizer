#!/usr/bin/env python3
"""
Demo Script for EV Route Optimizer Web Application
Tests all API endpoints and demonstrates system functionality
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List

class EVRouteOptimizerDemo:
    """Demo class for testing EV Route Optimizer web application"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {}
        
    def print_banner(self):
        """Print demo banner"""
        banner = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║     🧪 EV Route Optimizer - Demo & Testing Script 🧪        ║
        ║                                                              ║
        ║     Comprehensive API Testing and Functionality Demo        ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_step(self, step_number: int, description: str):
        """Print step header"""
        print(f"\n{'='*60}")
        print(f"STEP {step_number}: {description}")
        print('='*60)
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"❌ {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"ℹ️  {message}")
    
    def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request and handle response"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            return {"status": "error", "error": "Connection failed - is the server running?"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Invalid JSON response"}
    
    def test_server_connection(self) -> bool:
        """Test if server is running and accessible"""
        self.print_step(1, "Testing Server Connection")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.print_success("Server is running and accessible")
                return True
            else:
                self.print_error(f"Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_error("Cannot connect to server. Please ensure it's running on http://localhost:5000")
            return False
        except Exception as e:
            self.print_error(f"Connection test failed: {str(e)}")
            return False
    
    def test_system_status(self) -> Dict[str, Any]:
        """Test system status endpoint"""
        self.print_step(2, "Checking System Status")
        
        result = self.make_request('GET', '/api/status')
        
        if result.get('status') == 'online':
            self.print_success("System is online")
            self.print_info(f"ML Status: {result.get('ml_status', 'unknown')}")
            self.print_info(f"System State: {result.get('system_state', {}).get('status', 'unknown')}")
            
            # Display data sources
            data_sources = result.get('data_sources', {})
            if data_sources:
                self.print_info("Active Data Sources:")
                for source, status in data_sources.items():
                    print(f"    - {source}: {status}")
        else:
            self.print_error(f"System status check failed: {result.get('error', 'Unknown error')}")
        
        self.results['system_status'] = result
        return result
    
    def test_system_initialization(self) -> Dict[str, Any]:
        """Test system initialization"""
        self.print_step(3, "Initializing System (ML Pipeline)")
        
        self.print_info("This may take a few minutes...")
        start_time = time.time()
        
        result = self.make_request('POST', '/api/initialize')
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        if result.get('status') == 'success':
            self.print_success(f"System initialized successfully in {duration} seconds")
            
            results_data = result.get('results', {})
            self.print_info(f"Best Model: {results_data.get('best_model', 'Unknown')}")
            self.print_info(f"Best Score: {results_data.get('best_score', 'Unknown')}")
            self.print_info(f"Models Trained: {results_data.get('models_trained', 'Unknown')}")
        else:
            self.print_error(f"System initialization failed: {result.get('error', 'Unknown error')}")
        
        self.results['initialization'] = result
        return result
    
    def test_energy_prediction(self) -> Dict[str, Any]:
        """Test energy prediction endpoint with multiple scenarios"""
        self.print_step(4, "Testing Energy Prediction")
        
        # Test scenarios
        scenarios = [
            {
                "name": "City Driving - Normal Conditions",
                "data": {
                    "speed_limit_kmh": 50,
                    "temperature": 20,
                    "traffic_density": 0.3,
                    "elevation_gradient": 0.0,
                    "road_type": "arterial"
                }
            },
            {
                "name": "Highway Driving - Optimal Conditions",
                "data": {
                    "speed_limit_kmh": 120,
                    "temperature": 22,
                    "traffic_density": 0.1,
                    "elevation_gradient": 0.0,
                    "road_type": "highway"
                }
            },
            {
                "name": "Winter Driving - Heavy Traffic",
                "data": {
                    "speed_limit_kmh": 30,
                    "temperature": -5,
                    "traffic_density": 0.8,
                    "elevation_gradient": 0.02,
                    "road_type": "local"
                }
            },
            {
                "name": "Mountain Driving - Steep Uphill",
                "data": {
                    "speed_limit_kmh": 60,
                    "temperature": 15,
                    "traffic_density": 0.2,
                    "elevation_gradient": 0.08,
                    "road_type": "arterial"
                }
            }
        ]
        
        prediction_results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔍 Scenario {i}: {scenario['name']}")
            
            result = self.make_request('POST', '/api/energy/predict', scenario['data'])
            
            if result.get('status') == 'success':
                prediction = result.get('prediction', {})
                energy_consumption = prediction.get('energy_consumption_wh_per_km', 0)
                model_used = prediction.get('model_used', 'Unknown')
                confidence = prediction.get('confidence', 'Unknown')
                
                self.print_success(f"Energy Consumption: {energy_consumption} Wh/km")
                self.print_info(f"Model: {model_used} | Confidence: {confidence}")
                
                # Display factors
                factors = prediction.get('factors', {})
                if factors:
                    print("    Factors:")
                    for factor, value in factors.items():
                        print(f"      - {factor}: {value}")
                
                prediction_results.append({
                    "scenario": scenario['name'],
                    "energy_consumption": energy_consumption,
                    "model": model_used,
                    "confidence": confidence
                })
            else:
                self.print_error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        
        self.results['energy_predictions'] = prediction_results
        return prediction_results
    
    def test_route_optimization(self) -> Dict[str, Any]:
        """Test route optimization endpoint"""
        self.print_step(5, "Testing Route Optimization")
        
        # Test routes
        routes = [
            {
                "name": "Berlin City Center Route",
                "data": {
                    "origin": "Brandenburg Gate, Berlin",
                    "destination": "Alexanderplatz, Berlin",
                    "vehicle_params": {
                        "type": "sedan",
                        "battery_level": 80
                    }
                }
            },
            {
                "name": "Long Distance Route",
                "data": {
                    "origin": "Berlin Hauptbahnhof",
                    "destination": "Potsdamer Platz, Berlin",
                    "vehicle_params": {
                        "type": "suv",
                        "battery_level": 60
                    }
                }
            }
        ]
        
        route_results = []
        
        for i, route in enumerate(routes, 1):
            print(f"\n🗺️  Route {i}: {route['name']}")
            
            result = self.make_request('POST', '/api/route/optimize', route['data'])
            
            if result.get('status') == 'success':
                route_data = result.get('route', {})
                
                self.print_success("Route optimized successfully")
                self.print_info(f"Distance: {route_data.get('total_distance_km', 0)} km")
                self.print_info(f"Energy: {route_data.get('total_energy_kwh', 0)} kWh")
                self.print_info(f"Avg Energy: {route_data.get('avg_energy_per_km', 0)} Wh/km")
                self.print_info(f"Est. Time: {route_data.get('estimated_time_minutes', 0)} minutes")
                
                # Display route segments
                segments = route_data.get('route_segments', [])
                if segments:
                    print(f"    Route Segments: {len(segments)}")
                    for j, segment in enumerate(segments[:3]):  # Show first 3 segments
                        print(f"      Segment {j+1}: {segment.get('distance_km', 0)} km, "
                              f"{segment.get('energy_wh_per_km', 0)} Wh/km")
                
                route_results.append({
                    "name": route['name'],
                    "distance_km": route_data.get('total_distance_km', 0),
                    "energy_kwh": route_data.get('total_energy_kwh', 0),
                    "segments": len(segments)
                })
            else:
                self.print_error(f"Route optimization failed: {result.get('error', 'Unknown error')}")
        
        self.results['route_optimizations'] = route_results
        return route_results
    
    def test_data_endpoints(self) -> Dict[str, Any]:
        """Test data retrieval endpoints"""
        self.print_step(6, "Testing Data Endpoints")
        
        data_results = {}
        
        # Test road network data
        print("\n🛣️  Testing Road Network Data")
        result = self.make_request('GET', '/api/data/road-network')
        
        if result.get('status') == 'success':
            summary = result.get('summary', {})
            self.print_success("Road network data retrieved")
            self.print_info(f"Total Segments: {summary.get('total_segments', 0)}")
            self.print_info(f"Avg Energy/km: {summary.get('avg_energy_per_km', 0)} Wh/km")
            self.print_info(f"Coverage Area: {summary.get('coverage_area_km2', 0)} km²")
            data_results['road_network'] = summary
        else:
            self.print_error(f"Road network data failed: {result.get('error', 'Unknown error')}")
        
        # Test charging stations data
        print("\n🔌 Testing Charging Stations Data")
        result = self.make_request('GET', '/api/data/charging-stations')
        
        if result.get('status') == 'success':
            summary = result.get('summary', {})
            self.print_success("Charging stations data retrieved")
            self.print_info(f"Total Stations: {summary.get('total_stations', 0)}")
            self.print_info(f"Available Stations: {summary.get('available_stations', 0)}")
            self.print_info(f"Fast Charging: {summary.get('fast_charging_stations', 0)}")
            self.print_info(f"Avg Price: €{summary.get('avg_price_per_kwh', 0)}/kWh")
            data_results['charging_stations'] = summary
        else:
            self.print_error(f"Charging stations data failed: {result.get('error', 'Unknown error')}")
        
        # Test weather data
        print("\n🌤️  Testing Weather Data")
        result = self.make_request('GET', '/api/weather/current')
        
        if result.get('status') == 'success':
            weather = result.get('data', {})
            self.print_success("Weather data retrieved")
            self.print_info(f"Temperature: {weather.get('temperature_celsius', 0)}°C")
            self.print_info(f"Humidity: {weather.get('humidity_percent', 0)}%")
            self.print_info(f"Wind Speed: {weather.get('wind_speed_ms', 0)} m/s")
            data_results['weather'] = weather
        else:
            self.print_error(f"Weather data failed: {result.get('error', 'Unknown error')}")
        
        self.results['data_endpoints'] = data_results
        return data_results
    
    def test_ml_models_endpoint(self) -> Dict[str, Any]:
        """Test ML models information endpoint"""
        self.print_step(7, "Testing ML Models Information")
        
        result = self.make_request('GET', '/api/ml/models')
        
        if result.get('status') == 'success':
            data = result.get('data', {})
            
            self.print_success("ML models information retrieved")
            self.print_info(f"Models Trained: {data.get('models_trained', 0)}")
            self.print_info(f"Best Model: {data.get('best_model', 'Unknown')}")
            self.print_info(f"Best R² Score: {data.get('best_r2_score', 0)}")
            
            # Display model comparison
            comparison = data.get('model_comparison', [])
            if comparison:
                print("\n    Model Performance Comparison:")
                for model in comparison:
                    print(f"      {model.get('model_name', 'Unknown')}:")
                    print(f"        - R² Score: {model.get('r2_score', 0):.4f}")
                    print(f"        - MAE: {model.get('mae', 0):.2f}")
                    print(f"        - RMSE: {model.get('rmse', 0):.2f}")
            
            # Display top features
            features = data.get('feature_importance', [])
            if features:
                print("\n    Top Important Features:")
                for i, feature in enumerate(features[:5]):  # Show top 5
                    print(f"      {i+1}. {feature.get('feature', 'Unknown')}: "
                          f"{feature.get('importance', 0):.4f}")
        else:
            self.print_error(f"ML models info failed: {result.get('error', 'Unknown error')}")
        
        self.results['ml_models'] = result
        return result
    
    def test_data_export(self) -> Dict[str, Any]:
        """Test data export endpoint"""
        self.print_step(8, "Testing Data Export")
        
        result = self.make_request('GET', '/api/export/data')
        
        if result.get('status') == 'success':
            exported_files = result.get('exported_files', [])
            self.print_success(f"Data exported successfully")
            self.print_info(f"Files exported: {len(exported_files)}")
            
            if exported_files:
                print("    Exported Files:")
                for file in exported_files:
                    print(f"      - {file}")
        else:
            self.print_error(f"Data export failed: {result.get('error', 'Unknown error')}")
        
        self.results['data_export'] = result
        return result
    
    def generate_demo_report(self):
        """Generate comprehensive demo report"""
        self.print_step(9, "Generating Demo Report")
        
        report_data = {
            "demo_timestamp": datetime.now().isoformat(),
            "server_url": self.base_url,
            "test_results": self.results,
            "summary": {
                "total_tests": 8,
                "successful_tests": 0,
                "failed_tests": 0
            }
        }
        
        # Count successful tests
        successful_tests = 0
        for test_name, test_result in self.results.items():
            if isinstance(test_result, dict) and test_result.get('status') == 'success':
                successful_tests += 1
            elif isinstance(test_result, list) and test_result:  # For prediction/route results
                successful_tests += 1
        
        report_data["summary"]["successful_tests"] = successful_tests
        report_data["summary"]["failed_tests"] = 8 - successful_tests
        
        # Save report
        report_filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.print_success(f"Demo report saved: {report_filename}")
        except Exception as e:
            self.print_error(f"Failed to save report: {str(e)}")
        
        # Print summary
        print(f"\n📊 DEMO SUMMARY")
        print(f"   Total Tests: {report_data['summary']['total_tests']}")
        print(f"   Successful: {report_data['summary']['successful_tests']}")
        print(f"   Failed: {report_data['summary']['failed_tests']}")
        print(f"   Success Rate: {(successful_tests/8)*100:.1f}%")
    
    def run_full_demo(self):
        """Run complete demo sequence"""
        self.print_banner()
        
        print("🚀 Starting comprehensive EV Route Optimizer demo...")
        print(f"   Target Server: {self.base_url}")
        print(f"   Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check server connection first
        if not self.test_server_connection():
            print("\n❌ Demo aborted: Cannot connect to server")
            print("   Please ensure the server is running with: python app.py")
            return False
        
        # Run all tests
        try:
            self.test_system_status()
            self.test_system_initialization()
            self.test_energy_prediction()
            self.test_route_optimization()
            self.test_data_endpoints()
            self.test_ml_models_endpoint()
            self.test_data_export()
            self.generate_demo_report()
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"   Check the generated report for detailed results")
            return True
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            return False

def main():
    """Main function"""
    # Parse command line arguments
    base_url = "http://localhost:5000"
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("""
EV Route Optimizer Demo Script

Usage: python demo_webapp.py [options]

Options:
  --help, -h          Show this help message
  --url URL           Set custom server URL (default: http://localhost:5000)

Examples:
  python demo_webapp.py                           # Run demo on localhost:5000
  python demo_webapp.py --url http://localhost:8080  # Run demo on custom URL
            """)
            return
        elif sys.argv[1] == '--url' and len(sys.argv) > 2:
            base_url = sys.argv[2]
    
    # Create and run demo
    demo = EVRouteOptimizerDemo(base_url)
    success = demo.run_full_demo()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 