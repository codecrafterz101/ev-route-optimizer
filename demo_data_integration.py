#!/usr/bin/env python3
"""
Demonstration of Comprehensive Data Integration for EV Route Optimization
Berlin Case Study Implementation
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_integration.data_manager import DataManager

def main():
    """Main demonstration function"""
    print("🚗 EV Route Optimization - Berlin Case Study")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Initialize data manager
    print("📊 Initializing Data Manager...")
    data_manager = DataManager(config)
    
    # Demonstrate comprehensive data integration
    demonstrate_data_integration(data_manager)
    
    # Demonstrate real-time data collection
    demonstrate_real_time_data(data_manager)
    
    # Demonstrate data export and visualization
    demonstrate_data_export(data_manager)
    
    print("\n✅ Demonstration completed successfully!")

def demonstrate_data_integration(data_manager):
    """Demonstrate comprehensive data integration"""
    print("\n🔗 Comprehensive Data Integration")
    print("-" * 40)
    
    # Get comprehensive road network
    print("🛣️  Building comprehensive road network...")
    road_network = data_manager.get_comprehensive_road_network()
    
    if not road_network.empty:
        print(f"   ✅ Road network created with {len(road_network)} segments")
        print(f"   📍 Coverage area: {data_manager._calculate_coverage_area(road_network)} km²")
        
        # Show sample data
        print("\n   📋 Sample road segment data:")
        sample_columns = ['highway', 'speed_limit_kmh', 'road_type', 'energy_efficiency_factor', 
                         'temperature_impact', 'traffic_energy_impact', 'final_energy_wh_per_km']
        sample_data = road_network[sample_columns].head(3)
        print(sample_data.to_string())
        
        # Energy consumption analysis
        total_energy = road_network['segment_energy_wh'].sum() / 1000  # Convert to kWh
        avg_energy_per_km = road_network['final_energy_wh_per_km'].mean()
        print(f"\n   ⚡ Energy Analysis:")
        print(f"      Total network energy consumption: {total_energy:.2f} kWh")
        print(f"      Average energy per km: {avg_energy_per_km:.2f} Wh/km")
    else:
        print("   ❌ Failed to create road network")
    
    # Get charging stations
    print("\n🔌 Charging Station Integration...")
    charging_stations = data_manager.get_charging_stations_with_availability()
    
    if not charging_stations.empty:
        print(f"   ✅ Found {len(charging_stations)} charging stations")
        print(f"   🔋 Fast charging stations: {len(charging_stations[charging_stations['charging_type'] == 'fast'])}")
        print(f"   ✅ Available stations: {charging_stations['is_available'].sum()}")
        
        # Show charging station statistics
        avg_power = charging_stations['charging_power_kw'].mean()
        avg_price = charging_stations['price_per_kwh'].mean()
        print(f"   ⚡ Average charging power: {avg_power:.1f} kW")
        print(f"   💰 Average price: €{avg_price:.2f}/kWh")
    else:
        print("   ❌ No charging stations found")

def demonstrate_real_time_data(data_manager):
    """Demonstrate real-time data collection"""
    print("\n⏰ Real-Time Data Collection")
    print("-" * 40)
    
    # Get real-time data summary
    print("📡 Collecting real-time data...")
    summary = data_manager.get_real_time_data_summary()
    
    print(f"   🕐 Timestamp: {summary['timestamp']}")
    print(f"   🔧 System Status: {summary['system_status']}")
    
    # Display data source status
    print("\n   📊 Data Source Status:")
    for source, data in summary['data_sources'].items():
        status_emoji = "✅" if data['status'] == 'available' else "⚠️" if data['status'] == 'no_data' else "❌"
        print(f"      {status_emoji} {source.replace('_', ' ').title()}: {data['status']}")
        
        if data['status'] == 'available':
            if source == 'weather':
                print(f"         🌡️  Temperature: {data['temperature_celsius']}°C")
                print(f"         🌤️  Conditions: {data['description']}")
                print(f"         ⚡ Energy Impact: {data['energy_impact']:.2%}")
            elif source == 'traffic':
                print(f"         🚗 Average Speed: {data['average_speed_kmh']:.1f} km/h")
                print(f"         🚦 Congestion: {data['congestion_level']}")
                print(f"         📍 Segments: {data['segments_count']}")
            elif source == 'road_network':
                print(f"         🛣️  Segments: {data['segments_count']}")
                print(f"         📍 Coverage: {data['coverage_area_km2']} km²")
            elif source == 'charging_stations':
                print(f"         🔌 Total: {data['total_count']}")
                print(f"         ✅ Available: {data['available_count']}")
                print(f"         ⚡ Fast Charging: {data['fast_charging_count']}")

def demonstrate_data_export(data_manager):
    """Demonstrate data export and visualization"""
    print("\n📤 Data Export and Visualization")
    print("-" * 40)
    
    # Export data
    print("💾 Exporting data for analysis...")
    exported_files = data_manager.export_data_for_analysis("demo_exports")
    
    if exported_files:
        print("   ✅ Data exported successfully:")
        for data_type, file_path in exported_files.items():
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"      📄 {data_type}: {file_path} ({file_size:.1f} KB)")
    
    # Create simple visualization
    print("\n🗺️  Creating interactive map...")
    create_interactive_map(data_manager, exported_files)

def create_interactive_map(data_manager, exported_files):
    """Create an interactive map with the data"""
    try:
        # Get Berlin center coordinates
        berlin_center = [52.5200, 13.4050]
        
        # Create base map
        m = folium.Map(
            location=berlin_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add charging stations if available
        charging_stations = data_manager.get_charging_stations_with_availability()
        if not charging_stations.empty:
            for idx, station in charging_stations.head(20).iterrows():  # Limit to 20 for demo
                color = 'green' if station['is_available'] else 'red'
                size = 8 if station['charging_type'] == 'fast' else 5
                
                folium.CircleMarker(
                    location=[station.geometry.y, station.geometry.x],
                    radius=size,
                    popup=f"""
                    <b>Charging Station</b><br>
                    Type: {station['charging_type']}<br>
                    Power: {station['charging_power_kw']} kW<br>
                    Available: {'Yes' if station['is_available'] else 'No'}<br>
                    Price: €{station['price_per_kwh']:.2f}/kWh
                    """,
                    color=color,
                    fill=True
                ).add_to(m)
        
        # Add weather information
        weather = data_manager.weather_collector.get_current_weather()
        folium.Marker(
            location=berlin_center,
            popup=f"""
            <b>Current Weather</b><br>
            Temperature: {weather['temperature_celsius']}°C<br>
            Conditions: {weather['description']}<br>
            Energy Impact: {weather['total_weather_impact_factor']:.2%}
            """,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        # Save map
        map_file = "demo_exports/berlin_ev_map.html"
        m.save(map_file)
        print(f"   🗺️  Interactive map saved: {map_file}")
        
    except Exception as e:
        print(f"   ❌ Failed to create map: {e}")

def analyze_energy_consumption(data_manager):
    """Analyze energy consumption patterns"""
    print("\n⚡ Energy Consumption Analysis")
    print("-" * 40)
    
    road_network = data_manager.get_comprehensive_road_network()
    
    if not road_network.empty:
        # Calculate statistics
        energy_stats = {
            'total_energy_kwh': road_network['segment_energy_wh'].sum() / 1000,
            'avg_energy_per_km': road_network['final_energy_wh_per_km'].mean(),
            'max_energy_per_km': road_network['final_energy_wh_per_km'].max(),
            'min_energy_per_km': road_network['final_energy_wh_per_km'].min(),
            'total_distance_km': road_network.geometry.length.sum() * 111,  # Convert to km
        }
        
        print(f"   📏 Total network distance: {energy_stats['total_distance_km']:.1f} km")
        print(f"   ⚡ Total energy consumption: {energy_stats['total_energy_kwh']:.2f} kWh")
        print(f"   📊 Average energy per km: {energy_stats['avg_energy_per_km']:.2f} Wh/km")
        print(f"   🔥 Maximum energy per km: {energy_stats['max_energy_per_km']:.2f} Wh/km")
        print(f"   ❄️  Minimum energy per km: {energy_stats['min_energy_per_km']:.2f} Wh/km")
        
        # Road type analysis
        print("\n   🛣️  Energy consumption by road type:")
        road_type_analysis = road_network.groupby('road_type')['final_energy_wh_per_km'].agg(['mean', 'count'])
        for road_type, data in road_type_analysis.iterrows():
            print(f"      {road_type}: {data['mean']:.2f} Wh/km ({data['count']} segments)")

if __name__ == "__main__":
    main() 