#!/usr/bin/env python3
"""
Complete Machine Learning Pipeline for EV Route Optimization
Integrates data preprocessing, model training, and evaluation
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_integration.data_manager import DataManager
from data_integration.ml_data_preprocessor import MLDataPreprocessor
from models.ml_model_trainer import MLModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EVMLPipeline:
    """
    Complete Machine Learning Pipeline for EV Energy Consumption Prediction
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.data_manager = DataManager(self.config)
        self.preprocessor = MLDataPreprocessor(self.config)
        self.trainer = MLModelTrainer(self.config)
        
        # Pipeline state
        self.training_data = None
        self.trained_models = None
        self.pipeline_results = {}
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the pipeline"""
        directories = ['models', 'data_exports', 'logs', 'results']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def run_complete_pipeline(self, use_synthetic_data=False):
        """
        Run the complete ML pipeline from data preparation to model training
        
        Args:
            use_synthetic_data: Whether to use synthetic data for training
            
        Returns:
            Dict containing all pipeline results
        """
        logger.info("🚀 Starting Complete EV ML Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Data Preparation
            logger.info("📊 Step 1: Data Preparation")
            features_df, target_series = self.prepare_training_data(use_synthetic_data)
            
            # Step 2: Data Preprocessing
            logger.info("🔧 Step 2: Data Preprocessing")
            X_train, X_test, y_train, y_test = self.preprocess_data(features_df, target_series)
            
            # Step 3: Model Training
            logger.info("🏋️ Step 3: Model Training")
            training_results = self.train_models(X_train, y_train, X_test, y_test)
            
            # Step 4: Model Evaluation
            logger.info("📈 Step 4: Model Evaluation")
            evaluation_results = self.evaluate_models(training_results, X_test, y_test)
            
            # Step 5: Save Results
            logger.info("💾 Step 5: Save Results")
            self.save_pipeline_results(training_results, evaluation_results)
            
            # Step 6: Generate Reports
            logger.info("📋 Step 6: Generate Reports")
            self.generate_reports(training_results, evaluation_results)
            
            logger.info("✅ Complete ML Pipeline finished successfully!")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'pipeline_summary': self.get_pipeline_summary()
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def prepare_training_data(self, use_synthetic_data=False):
        """Prepare training data from integrated sources"""
        logger.info("🔄 Preparing training data...")
        
        if use_synthetic_data:
            logger.info("🔄 Using synthetic training data...")
            # Create synthetic data directly
            training_data = self.preprocessor._create_synthetic_training_data(n_samples=2000)
            
            # Add target variable
            training_data['energy_consumption_wh_per_km'] = self.preprocessor._calculate_synthetic_energy_consumption(training_data)
            
            # Engineer features
            training_data = self.preprocessor._engineer_features(training_data)
            training_data = self.preprocessor._handle_missing_values(training_data)
            
            # Separate features and target
            features_df, target_series = self.preprocessor._separate_features_target(training_data)
            
        else:
            logger.info("🔄 Using integrated real data...")
            # Use the preprocessor to prepare data from integrated sources
            features_df, target_series = self.preprocessor.prepare_training_data(self.data_manager)
        
        self.training_data = {
            'features': features_df,
            'target': target_series
        }
        
        logger.info(f"✅ Training data prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, target_series
    
    def preprocess_data(self, features_df, target_series):
        """Preprocess data for ML training"""
        logger.info("🔧 Preprocessing data...")
        
        # Scale features
        scaled_features = self.preprocessor.scale_features(features_df, fit_scalers=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            scaled_features, target_series, test_size=0.2, random_state=42
        )
        
        # Save preprocessor state
        self.preprocessor.save_preprocessor('models/data_preprocessor.joblib')
        
        logger.info(f"✅ Data preprocessing completed:")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test, fast_mode=True):
        """Train ML models"""
        logger.info("🏋️ Training ML models...")
        
        # Train all models
        training_results = self.trainer.train_all_models(
            X_train, y_train, X_test, y_test, self.preprocessor, fast_mode
        )
        
        self.trained_models = training_results
        
        logger.info("✅ Model training completed!")
        
        return training_results
    
    def evaluate_models(self, training_results, X_test, y_test):
        """Evaluate trained models"""
        logger.info("📈 Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, model_data in training_results.items():
            if model_name == 'comparison':
                continue
                
            model = model_data['model']
            metrics = model_data['metrics']
            
            # Additional evaluation metrics
            predictions = self.trainer.predict_energy_consumption(
                model, X_test, model_type=model_name
            )
            
            # Calculate additional metrics
            additional_metrics = {
                'prediction_range': (predictions.min(), predictions.max()),
                'actual_range': (y_test.min(), y_test.max()),
                'prediction_mean': predictions.mean(),
                'actual_mean': y_test.mean(),
                'prediction_std': predictions.std(),
                'actual_std': y_test.std(),
            }
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'additional_metrics': additional_metrics,
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            }
        
        logger.info("✅ Model evaluation completed!")
        
        return evaluation_results
    
    def save_pipeline_results(self, training_results, evaluation_results):
        """Save all pipeline results"""
        logger.info("💾 Saving pipeline results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        pipeline_results = {
            'timestamp': timestamp,
            'config': {
                'ev_params': self.config.EV_PARAMS,
                'ml_params': self.config.ML_PARAMS,
                'weather_impact': self.config.WEATHER_IMPACT,
                'traffic_impact': self.config.TRAFFIC_IMPACT
            },
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'pipeline_summary': self.get_pipeline_summary()
        }
        
        # Save to JSON
        results_file = f"results/pipeline_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        logger.info(f"💾 Pipeline results saved: {results_file}")
        
        return results_file
    
    def generate_reports(self, training_results, evaluation_results):
        """Generate comprehensive reports"""
        logger.info("📋 Generating reports...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate model comparison report
        self._generate_model_comparison_report(training_results, timestamp)
        
        # Generate feature importance report
        self._generate_feature_importance_report(training_results, timestamp)
        
        # Generate performance summary
        self._generate_performance_summary(training_results, evaluation_results, timestamp)
        
        logger.info("✅ Reports generated successfully!")
    
    def _generate_model_comparison_report(self, training_results, timestamp):
        """Generate model comparison report"""
        comparison = training_results.get('comparison', {})
        
        report = f"""
# EV Energy Consumption Model Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Performance Summary

| Model | R² Score | MAE | RMSE | MAPE |
|-------|----------|-----|------|------|
"""
        
        for model in comparison.get('models', []):
            report += f"| {model['name'].replace('_', ' ').title()} | {model['test_r2']:.4f} | {model['test_mae']:.4f} | {model['test_rmse']:.4f} | {model['test_mape']:.2f}% |\n"
        
        report += f"""
## Best Model
**{comparison.get('best_model', 'N/A').replace('_', ' ').title()}** with R² score of **{comparison.get('best_score', 0):.4f}**

## Training Configuration
- Random Forest: {self.config.ML_PARAMS['random_forest']}
- Neural Network: {self.config.ML_PARAMS['neural_network']}
- EV Parameters: {self.config.EV_PARAMS}
"""
        
        # Save report
        report_file = f"results/model_comparison_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Model comparison report: {report_file}")
    
    def _generate_feature_importance_report(self, training_results, timestamp):
        """Generate feature importance report"""
        if 'random_forest' in training_results:
            rf_metrics = training_results['random_forest']['metrics']
            if 'feature_importance' in rf_metrics:
                feature_importance = rf_metrics['feature_importance']
                
                report = f"""
# Feature Importance Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Top 20 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
"""
                
                for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
                    report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"
                
                # Save report
                report_file = f"results/feature_importance_report_{timestamp}.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"📊 Feature importance report: {report_file}")
    
    def _generate_performance_summary(self, training_results, evaluation_results, timestamp):
        """Generate performance summary"""
        summary = self.get_pipeline_summary()
        
        report = f"""
# EV ML Pipeline Performance Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline Overview
- **Total Models Trained**: {summary['models_trained']}
- **Best Model**: {summary['best_model']}
- **Best R² Score**: {summary['best_r2_score']:.4f}
- **Training Duration**: {summary.get('training_duration', 'N/A')}

## Model Details
"""
        
        for model_name, details in summary['model_details'].items():
            report += f"""
### {model_name.replace('_', ' ').title()}
- **R² Score**: {details['test_r2']:.4f}
- **MAE**: {details['test_mae']:.4f}
- **RMSE**: {details['test_rmse']:.4f}
- **MAPE**: {details['test_mape']:.2f}%
- **Features**: {details['feature_count']}
"""
        
        # Save report
        report_file = f"results/performance_summary_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Performance summary: {report_file}")
    
    def get_pipeline_summary(self):
        """Get comprehensive pipeline summary"""
        if not self.trained_models:
            return {"error": "No models trained yet"}
        
        summary = self.trainer.get_model_summary(self.trained_models)
        
        # Add additional pipeline information
        if self.training_data:
            summary['data_info'] = {
                'total_samples': len(self.training_data['features']),
                'total_features': len(self.training_data['features'].columns),
                'target_mean': self.training_data['target'].mean(),
                'target_std': self.training_data['target'].std()
            }
        
        return summary
    
    def predict_energy_consumption(self, input_data, model_name=None):
        """
        Make energy consumption predictions using trained models
        
        Args:
            input_data: DataFrame with features
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Predictions
        """
        if not self.trained_models:
            raise ValueError("No models trained yet. Run the pipeline first.")
        
        # Use best model if none specified
        if model_name is None:
            model_name = self.trained_models['comparison']['best_model']
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]['model']
        
        # Preprocess input data
        if hasattr(self.preprocessor, 'scalers') and 'standard' in self.preprocessor.scalers:
            input_scaled = self.preprocessor.scale_features(input_data, fit_scalers=False)
        else:
            input_scaled = input_data
        
        # Make predictions
        predictions = self.trainer.predict_energy_consumption(model, input_scaled, model_name)
        
        return predictions
    
    def load_trained_pipeline(self, results_file):
        """Load a previously trained pipeline"""
        logger.info(f"📂 Loading trained pipeline from {results_file}")
        
        with open(results_file, 'r') as f:
            pipeline_data = json.load(f)
        
        # Load models
        timestamp = pipeline_data['timestamp']
        
        # Load Random Forest model
        rf_model_path = f"models/random_forest_{timestamp}.joblib"
        if os.path.exists(rf_model_path):
            self.trained_models = {
                'random_forest': {
                    'model': self.trainer.load_model(rf_model_path, 'random_forest'),
                    'metrics': pipeline_data['training_results']['random_forest']['metrics']
                }
            }
        
        # Load Neural Network model
        nn_model_path = f"models/neural_network_{timestamp}.h5"
        if os.path.exists(nn_model_path):
            self.trained_models['neural_network'] = {
                'model': self.trainer.load_model(nn_model_path, 'neural_network'),
                'metrics': pipeline_data['training_results']['neural_network']['metrics']
            }
        
        # Load preprocessor
        preprocessor_path = "models/data_preprocessor.joblib"
        if os.path.exists(preprocessor_path):
            self.preprocessor.load_preprocessor(preprocessor_path)
        
        logger.info("✅ Trained pipeline loaded successfully!")
        
        return pipeline_data

def main():
    """Main function to run the ML pipeline"""
    print("🚗 EV Energy Consumption ML Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = EVMLPipeline()
    
    # Run complete pipeline with synthetic data (for demonstration)
    print("\n🔄 Running ML pipeline with synthetic data...")
    results = pipeline.run_complete_pipeline(use_synthetic_data=True)
    
    # Display results
    print("\n📊 Pipeline Results Summary:")
    summary = results['pipeline_summary']
    print(f"   Models trained: {summary['models_trained']}")
    print(f"   Best model: {summary['best_model']}")
    print(f"   Best R² score: {summary['best_r2_score']:.4f}")
    
    print("\n✅ ML Pipeline completed successfully!")
    print("📁 Check the 'results/' directory for detailed reports and 'models/' directory for saved models.")

if __name__ == "__main__":
    main() 