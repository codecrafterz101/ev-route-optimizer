"""
Machine Learning Model Trainer for EV Route Optimization
Handles training, evaluation, and persistence of Random Forest and Deep Neural Network models
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import logging
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Deep Neural Network training will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelTrainer:
    """
    Comprehensive ML model trainer for EV energy consumption prediction
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_metrics = {}
        self.training_history = {}
        
        # Model configurations
        self.rf_config = config.ML_PARAMS['random_forest']
        self.nn_config = config.ML_PARAMS['neural_network']
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
    
    def train_all_models(self, X_train, y_train, X_test, y_test, preprocessor=None, fast_mode=False):
        """
        Train both Random Forest and Deep Neural Network models
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            preprocessor: MLDataPreprocessor instance for feature scaling
            fast_mode: Use fast training mode (skip expensive hyperparameter optimization)
            
        Returns:
            Dict containing trained models and metrics
        """
        logger.info("🚀 Starting comprehensive model training...")
        
        results = {}
        
        # Train Random Forest
        logger.info("🌲 Training Random Forest model...")
        rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test, fast_mode)
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics
        }
        
        # Train Deep Neural Network (if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            logger.info("🧠 Training Deep Neural Network model...")
            nn_model, nn_metrics = self.train_neural_network(X_train, y_train, X_test, y_test, preprocessor)
            results['neural_network'] = {
                'model': nn_model,
                'metrics': nn_metrics
            }
        else:
            logger.warning("⚠️ TensorFlow not available. Skipping Neural Network training.")
        
        # Compare models
        comparison = self.compare_models(results)
        results['comparison'] = comparison
        
        # Save models
        self.save_models(results)
        
        return results
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, fast_mode=False):
        """Train Random Forest model with hyperparameter optimization"""
        logger.info("🌲 Training Random Forest model...")
        
        if fast_mode:
            # Use default parameters for fast training
            logger.info("⚡ Fast mode: Using default Random Forest parameters")
            best_rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.rf_config['random_state'],
                n_jobs=-1
            )
            best_rf.fit(X_train, y_train)
        else:
            # Reduced hyperparameter grid for faster training
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            # Initialize base model
            base_rf = RandomForestRegressor(
                random_state=self.rf_config['random_state'],
                n_jobs=-1  # Use all CPU cores
            )
            
            # Grid search for hyperparameter optimization
            logger.info("🔍 Performing hyperparameter optimization...")
            grid_search = GridSearchCV(
                base_rf,
                param_grid,
                cv=3,  # Reduced from 5 to 3
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_rf = grid_search.best_estimator_
            logger.info(f"✅ Best Random Forest parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_pred_train = best_rf.predict(X_train)
        y_pred_test = best_rf.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance
        
        # Cross-validation score
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=3, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        
        metrics['cv_r2_std'] = cv_scores.std()
        
        logger.info(f"✅ Random Forest training completed. Test R²: {metrics['test_r2']:.4f}")
        
        return best_rf, metrics
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, preprocessor=None):
        """Train Deep Neural Network model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network training")
        
        logger.info("🧠 Training Deep Neural Network model...")
        
        # Scale features for neural network
        if preprocessor and 'standard' in preprocessor.scalers:
            X_train_scaled = preprocessor.scale_features(X_train, fit_scalers=False)
            X_test_scaled = preprocessor.scale_features(X_test, fit_scalers=False)
        else:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # Build neural network architecture
        model = self._build_neural_network(X_train_scaled.shape[1])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.nn_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks for training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info("🏋️ Training neural network...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=self.nn_config['epochs'],
            batch_size=self.nn_config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled).flatten()
        y_pred_test = model.predict(X_test_scaled).flatten()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        metrics['training_history'] = history.history
        
        logger.info(f"✅ Neural Network training completed. Test R²: {metrics['test_r2']:.4f}")
        
        return model, metrics
    
    def _build_neural_network(self, input_dim):
        """Build neural network architecture"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.nn_config['layers'][0],
            activation='relu',
            input_shape=(input_dim,)
        ))
        model.add(layers.Dropout(self.nn_config['dropout_rate']))
        
        # Hidden layers
        for units in self.nn_config['layers'][1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.nn_config['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        return model
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculate comprehensive model metrics"""
        metrics = {
            # Training metrics
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            
            # Test metrics
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            
            # Additional metrics
            'train_mean': y_train.mean(),
            'test_mean': y_test.mean(),
            'train_std': y_train.std(),
            'test_std': y_test.std(),
        }
        
        # Calculate percentage errors
        metrics['train_mape'] = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        metrics['test_mape'] = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        return metrics
    
    def compare_models(self, results):
        """Compare performance of different models"""
        logger.info("📊 Comparing model performance...")
        
        comparison = {
            'models': [],
            'best_model': None,
            'best_score': -np.inf
        }
        
        for model_name, model_data in results.items():
            if model_name == 'comparison':
                continue
                
            metrics = model_data['metrics']
            comparison['models'].append({
                'name': model_name,
                'test_r2': metrics['test_r2'],
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'test_mape': metrics['test_mape']
            })
            
            # Track best model
            if metrics['test_r2'] > comparison['best_score']:
                comparison['best_score'] = metrics['test_r2']
                comparison['best_model'] = model_name
        
        # Sort models by R² score
        comparison['models'].sort(key=lambda x: x['test_r2'], reverse=True)
        
        logger.info(f"🏆 Best model: {comparison['best_model']} (R²: {comparison['best_score']:.4f})")
        
        return comparison
    
    def save_models(self, results):
        """Save trained models and metadata"""
        logger.info("💾 Saving models and metadata...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model_data in results.items():
            if model_name == 'comparison':
                continue
                
            model = model_data['model']
            metrics = model_data['metrics']
            
            # Save model
            model_filename = f"models/{model_name}_{timestamp}.joblib"
            if model_name == 'neural_network':
                # Save Keras model
                model.save(f"models/{model_name}_{timestamp}.h5")
            else:
                # Save scikit-learn model
                joblib.dump(model, model_filename)
            
            # Save metrics
            metrics_filename = f"models/{model_name}_metrics_{timestamp}.json"
            with open(metrics_filename, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        json_metrics[key] = value.tolist()
                    elif isinstance(value, pd.DataFrame):
                        json_metrics[key] = value.to_dict()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_metrics[key] = float(value)
                    else:
                        json_metrics[key] = value
                
                json.dump(json_metrics, f, indent=2, default=str)
            
            logger.info(f"   💾 {model_name}: {model_filename}")
            logger.info(f"   📊 {model_name} metrics: {metrics_filename}")
        
        # Save comparison results
        comparison_filename = f"models/model_comparison_{timestamp}.json"
        with open(comparison_filename, 'w') as f:
            json.dump(results['comparison'], f, indent=2, default=str)
        
        logger.info(f"   📊 Model comparison: {comparison_filename}")
    
    def load_model(self, model_path, model_type='random_forest'):
        """Load a trained model"""
        logger.info(f"📂 Loading {model_type} model from {model_path}")
        
        if model_type == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required to load neural network models")
            model = keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)
        
        return model
    
    def predict_energy_consumption(self, model, X, model_type='random_forest'):
        """Make energy consumption predictions"""
        if model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            # For neural network, ensure input is properly scaled
            if isinstance(X, pd.DataFrame):
                X = X.values
            predictions = model.predict(X).flatten()
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def get_model_summary(self, results):
        """Generate comprehensive model summary"""
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'models_trained': len([k for k in results.keys() if k != 'comparison']),
            'best_model': results['comparison']['best_model'],
            'best_r2_score': results['comparison']['best_score'],
            'model_details': {}
        }
        
        for model_name, model_data in results.items():
            if model_name == 'comparison':
                continue
                
            metrics = model_data['metrics']
            summary['model_details'][model_name] = {
                'test_r2': metrics['test_r2'],
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'test_mape': metrics['test_mape'],
                'feature_count': len(metrics.get('feature_importance', []))
            }
        
        return summary
    
    def create_feature_importance_plot(self, results, save_path=None):
        """Create feature importance visualization"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get Random Forest feature importance
            if 'random_forest' in results:
                rf_metrics = results['random_forest']['metrics']
                if 'feature_importance' in rf_metrics:
                    feature_importance = rf_metrics['feature_importance']
                    
                    # Plot top 20 features
                    top_features = feature_importance.head(20)
                    
                    plt.figure(figsize=(12, 8))
                    sns.barplot(data=top_features, x='importance', y='feature')
                    plt.title('Random Forest Feature Importance (Top 20)')
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        logger.info(f"📊 Feature importance plot saved: {save_path}")
                    
                    plt.show()
                    
        except ImportError:
            logger.warning("⚠️ matplotlib/seaborn not available for plotting")
    
    def validate_model_performance(self, model, X_val, y_val, model_type='random_forest'):
        """Validate model performance on new data"""
        logger.info(f"🔍 Validating {model_type} model performance...")
        
        predictions = self.predict_energy_consumption(model, X_val, model_type)
        
        # Calculate validation metrics
        val_metrics = {
            'mae': mean_absolute_error(y_val, predictions),
            'mse': mean_squared_error(y_val, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
            'r2': r2_score(y_val, predictions),
            'mape': np.mean(np.abs((y_val - predictions) / y_val)) * 100
        }
        
        logger.info(f"   Validation R²: {val_metrics['r2']:.4f}")
        logger.info(f"   Validation MAE: {val_metrics['mae']:.4f}")
        logger.info(f"   Validation RMSE: {val_metrics['rmse']:.4f}")
        
        return val_metrics 