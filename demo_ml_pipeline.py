#!/usr/bin/env python3
"""
Demonstration of Machine Learning Pipeline for EV Route Optimization
Shows how to train and evaluate ML models for energy consumption prediction
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline import EVMLPipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function"""
    print("🚗 EV Energy Consumption ML Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize configuration and pipeline
    config = Config()
    pipeline = EVMLPipeline(config)
    
    # Demonstrate with synthetic data (fast and reliable)
    print("\n🔄 Running ML Pipeline with Synthetic Data")
    print("-" * 40)
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(use_synthetic_data=True)
        
        # Display results
        display_pipeline_results(results)
        
        # Demonstrate model prediction
        demonstrate_model_prediction(pipeline)
        
        # Show feature importance
        demonstrate_feature_importance(results)
        
        print("\n✅ ML Pipeline demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline demonstration failed: {str(e)}")
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)

def display_pipeline_results(results):
    """Display comprehensive pipeline results"""
    print("\n📊 Pipeline Results Summary")
    print("-" * 30)
    
    summary = results['pipeline_summary']
    
    print(f"🏆 Best Model: {summary['best_model']}")
    print(f"📈 Best R² Score: {summary['best_r2_score']:.4f}")
    print(f"🤖 Models Trained: {summary['models_trained']}")
    
    if 'data_info' in summary:
        data_info = summary['data_info']
        print(f"📊 Training Data: {data_info['total_samples']} samples, {data_info['total_features']} features")
        print(f"📏 Target Mean: {data_info['target_mean']:.2f} Wh/km")
        print(f"📏 Target Std: {data_info['target_std']:.2f} Wh/km")
    
    # Display model comparison
    print("\n📋 Model Performance Comparison")
    print("-" * 35)
    print(f"{'Model':<20} {'R² Score':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 50)
    
    for model_name, details in summary['model_details'].items():
        print(f"{model_name.replace('_', ' ').title():<20} "
              f"{details['test_r2']:<10.4f} "
              f"{details['test_mae']:<10.4f} "
              f"{details['test_rmse']:<10.4f}")

def demonstrate_model_prediction(pipeline):
    """Demonstrate making predictions with trained models"""
    print("\n🔮 Model Prediction Demonstration")
    print("-" * 35)
    
    # Create sample input data
    sample_data = create_sample_input_data()
    
    print("📝 Sample Input Data:")
    print(sample_data.head(3).to_string())
    
    # Make predictions with best model
    try:
        predictions = pipeline.predict_energy_consumption(sample_data)
        
        print(f"\n🎯 Predictions (Best Model: {pipeline.trained_models['comparison']['best_model']}):")
        for i, pred in enumerate(predictions[:5]):
            print(f"   Sample {i+1}: {pred:.2f} Wh/km")
        
        # Show prediction statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"   Mean: {predictions.mean():.2f} Wh/km")
        print(f"   Std: {predictions.std():.2f} Wh/km")
        print(f"   Min: {predictions.min():.2f} Wh/km")
        print(f"   Max: {predictions.max():.2f} Wh/km")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")

def demonstrate_feature_importance(results):
    """Demonstrate feature importance analysis"""
    print("\n🔍 Feature Importance Analysis")
    print("-" * 30)
    
    if 'random_forest' in results['training_results']:
        rf_metrics = results['training_results']['random_forest']['metrics']
        
        if 'feature_importance' in rf_metrics:
            feature_importance = rf_metrics['feature_importance']
            
            print("🏆 Top 10 Most Important Features:")
            print("-" * 35)
            print(f"{'Rank':<5} {'Feature':<30} {'Importance':<10}")
            print("-" * 45)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"{i:<5} {row['feature']:<30} {row['importance']:<10.4f}")
            
            # Show feature categories
            print(f"\n📊 Feature Categories:")
            road_features = [f for f in feature_importance['feature'] if any(x in f.lower() for x in ['road', 'highway', 'speed', 'elevation'])]
            weather_features = [f for f in feature_importance['feature'] if 'weather' in f.lower()]
            traffic_features = [f for f in feature_importance['feature'] if 'traffic' in f.lower()]
            ev_features = [f for f in feature_importance['feature'] if 'ev_' in f.lower()]
            
            print(f"   🛣️  Road Features: {len(road_features)}")
            print(f"   🌤️  Weather Features: {len(weather_features)}")
            print(f"   🚗 Traffic Features: {len(traffic_features)}")
            print(f"   ⚡ EV Features: {len(ev_features)}")

def create_sample_input_data():
    """Create sample input data for prediction demonstration"""
    # Create realistic sample data
    sample_data = pd.DataFrame({
        # Road features
        'speed_limit_kmh': [50, 80, 30],
        'elevation_gradient': [0.02, -0.01, 0.05],
        'road_length_m': [500, 1000, 200],
        'road_curvature': [0.01, 0.05, 0.02],
        
        # Weather features
        'weather_temperature_celsius': [15, 25, 5],
        'weather_humidity_percent': [60, 70, 80],
        'weather_wind_speed_ms': [5, 10, 15],
        'weather_precipitation_mm': [0, 2, 5],
        
        # Traffic features
        'traffic_density': [0.3, 0.7, 0.5],
        'traffic_average_speed_kmh': [45, 65, 25],
        'traffic_travel_time_factor': [1.1, 1.5, 1.8],
        
        # EV features
        'ev_battery_capacity_kwh': [75, 75, 75],
        'ev_efficiency_wh_per_km': [150, 150, 150],
        'ev_battery_level_percent': [0.6, 0.8, 0.4],
        'ev_vehicle_weight_kg': [1800, 1800, 1800],
        
        # Derived features
        'energy_efficiency_factor': [1.0, 1.1, 0.9],
        'temperature_impact': [1.05, 1.02, 1.08],
        'traffic_energy_impact': [1.1, 1.3, 1.2],
        'elevation_energy_impact': [1.02, 0.99, 1.05],
        
        # Time features
        'hour_of_day': [8, 14, 18],
        'day_of_week': [1, 3, 5],
        'is_weekend': [0, 0, 0],
        'is_rush_hour': [1, 0, 1],
        
        # Interaction features
        'speed_elevation_interaction': [1.0, -0.8, 1.5],
        'temperature_traffic_interaction': [4.5, 17.5, 2.5],
        'weather_traffic_interaction': [0, 1.4, 2.5],
        
        # Polynomial features
        'speed_limit_squared': [2500, 6400, 900],
        'elevation_gradient_squared': [0.0004, 0.0001, 0.0025],
        'temperature_squared': [225, 625, 25],
        
        # Ratio features
        'speed_to_limit_ratio': [0.9, 0.81, 0.83],
        'energy_efficiency_ratio': [0.0067, 0.0073, 0.006],
        
        # Categorical encoded features
        'highway_encoded': [2, 0, 3],
        'road_type_encoded': [1, 0, 2],
        'surface_encoded': [0, 1, 0],
        'weather_condition_encoded': [0, 1, 2],
        'traffic_congestion_level_encoded': [1, 2, 1],
        
        # Charging station features
        'distance_to_nearest_charger_km': [1.5, 3.0, 0.8],
        'charger_density_per_km2': [0.5, 0.2, 1.2],
    })
    
    return sample_data

def demonstrate_real_data_pipeline():
    """Demonstrate pipeline with real integrated data (optional)"""
    print("\n🌍 Real Data Pipeline Demonstration")
    print("-" * 35)
    print("⚠️  This requires API keys and may take longer...")
    
    response = input("Do you want to try with real data? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            pipeline = EVMLPipeline()
            results = pipeline.run_complete_pipeline(use_synthetic_data=False)
            display_pipeline_results(results)
            print("✅ Real data pipeline completed!")
        except Exception as e:
            print(f"❌ Real data pipeline failed: {str(e)}")
            print("💡 This is expected if API keys are not configured")
    else:
        print("⏭️  Skipping real data demonstration")

if __name__ == "__main__":
    main()
    
    # Optional: Demonstrate with real data
    print("\n" + "=" * 60)
    demonstrate_real_data_pipeline()
    
    print("\n🎉 ML Pipeline demonstration completed!")
    print("📁 Check the following directories for results:")
    print("   📊 results/ - Detailed reports and metrics")
    print("   🤖 models/ - Trained models and preprocessors")
    print("   📈 logs/ - Training logs and debugging info") 