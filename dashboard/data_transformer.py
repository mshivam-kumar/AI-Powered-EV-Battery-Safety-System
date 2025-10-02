#!/usr/bin/env python3
"""
Data Transformation Pipeline
Converts real-world telemetry to standardized features for model inference
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TelemetryTransformer:
    """
    Transforms real-world telemetry data to standardized features
    that match the training data format
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.scaler = None
        self.label_encoder = None
        self.feature_stats = None
        self.load_preprocessing_artifacts()
    
    def load_preprocessing_artifacts(self):
        """Load the scaler and encoders used during training"""
        try:
            # Try to load saved preprocessing artifacts
            scaler_path = self.models_dir / "feature_scaler.pkl"
            encoder_path = self.models_dir / "label_encoder.pkl"
            stats_path = self.models_dir / "feature_stats.json"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # If artifacts don't exist, create default ones based on NASA data analysis
            if self.scaler is None:
                self.create_default_scaler()
                
        except Exception as e:
            print(f"Warning: Could not load preprocessing artifacts: {e}")
            self.create_default_scaler()
    
    def create_default_scaler(self):
        """Create default scaler based on NASA battery data statistics"""
        # These are approximate statistics from the NASA battery dataset
        # after feature engineering (before standardization)
        
        feature_means = np.array([
            3.7,    # voltage (V)
            0.0,    # current (A) - can be positive/negative
            25.0,   # temperature (°C)
            0.5,    # soc (0-1)
            20.0,   # ambient_temp (°C)
            0.5,    # humidity (0-1)
            1.0,    # charge_mode_encoded (0,1,2)
            9.25,   # power (W)
            0.6,    # c_rate
            5.0,    # temp_diff (°C)
            7.4,    # voltage_soc_ratio
            0.5,    # thermal_stress
            0.0,    # temp_gradient
            0.0,    # voltage_gradient
            0.0,    # soc_rate
            0.3     # env_stress
        ])
        
        feature_stds = np.array([
            0.4,    # voltage
            2.5,    # current
            8.0,    # temperature
            0.15,   # soc (smaller std for 0-1 range)
            6.0,    # ambient_temp
            0.2,    # humidity
            0.8,    # charge_mode_encoded
            4.0,    # power
            0.3,    # c_rate
            3.0,    # temp_diff
            3.0,    # voltage_soc_ratio
            0.3,    # thermal_stress
            1.0,    # temp_gradient
            0.1,    # voltage_gradient
            0.1,    # soc_rate
            0.2     # env_stress
        ])
        
        # Create a StandardScaler with these statistics
        self.scaler = StandardScaler()
        self.scaler.mean_ = feature_means
        self.scaler.scale_ = feature_stds
        self.scaler.n_features_in_ = len(feature_means)
        
        # Create label encoder for charge modes
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['fast', 'pause', 'slow'])
    
    def extract_features_from_telemetry(self, telemetry):
        """
        Extract 16 features from real-world telemetry data
        (same as training pipeline but without standardization)
        """
        # Basic features
        voltage = telemetry['voltage']
        current = telemetry['current']
        temperature = telemetry['temperature']
        soc = telemetry['soc']  # Should be 0-1, not 0-100
        ambient_temp = telemetry['ambient_temp']
        humidity = telemetry.get('humidity', 0.5)
        charge_mode = telemetry['charge_mode']
        
        # Encode charge mode
        try:
            charge_mode_encoded = self.label_encoder.transform([charge_mode])[0]
        except:
            # Default to 'pause' if unknown mode
            charge_mode_encoded = self.label_encoder.transform(['pause'])[0]
        
        # Derived features (same calculations as training)
        power = voltage * abs(current)
        c_rate = abs(current) / 2.5  # Assuming 2.5Ah nominal capacity
        temp_diff = temperature - ambient_temp
        voltage_soc_ratio = voltage / max(soc, 0.01)
        thermal_stress = max(0, (temperature - 25) / 25)  # Normalized temp stress
        
        # Gradients (simplified for real-time, would need history for accurate calculation)
        temp_gradient = 0.0  # Would need previous readings
        voltage_gradient = 0.0
        soc_rate = 0.0
        
        # Environmental stress
        env_stress = abs(ambient_temp - 25) / 25 + abs(humidity - 0.5) * 2
        
        # Combine all features
        features = np.array([
            voltage,
            current,
            temperature,
            soc,
            ambient_temp,
            humidity,
            charge_mode_encoded,
            power,
            c_rate,
            temp_diff,
            voltage_soc_ratio,
            thermal_stress,
            temp_gradient,
            voltage_gradient,
            soc_rate,
            env_stress
        ])
        
        return features
    
    def transform_telemetry(self, telemetry):
        """
        Transform real-world telemetry to standardized features for model inference
        
        Args:
            telemetry: Dict with keys: voltage, current, temperature, soc, 
                      ambient_temp, humidity, charge_mode
        
        Returns:
            Standardized features array ready for model prediction
        """
        # Extract raw features
        raw_features = self.extract_features_from_telemetry(telemetry)
        
        # Standardize features using the training scaler
        if self.scaler is not None:
            standardized_features = self.scaler.transform(raw_features.reshape(1, -1))
            return standardized_features
        else:
            # If no scaler available, return raw features (not recommended)
            return raw_features.reshape(1, -1)
    
    def inverse_transform_for_rl(self, standardized_values, feature_names):
        """
        Convert standardized values back to real-world values for RL agent
        This helps understand what the RL agent is seeing
        """
        if self.scaler is None:
            return standardized_values
        
        # Map feature names to indices
        feature_map = {
            'voltage': 0, 'current': 1, 'temperature': 2, 'soc': 3,
            'ambient_temp': 4, 'humidity': 5, 'charge_mode': 6,
            'power': 7, 'c_rate': 8, 'temp_diff': 9,
            'voltage_soc_ratio': 10, 'thermal_stress': 11,
            'temp_gradient': 12, 'voltage_gradient': 13,
            'soc_rate': 14, 'env_stress': 15
        }
        
        result = {}
        for feature_name in feature_names:
            if feature_name in feature_map:
                idx = feature_map[feature_name]
                # Inverse transform: value = (standardized * std) + mean
                real_value = (standardized_values[idx] * self.scaler.scale_[idx]) + self.scaler.mean_[idx]
                result[feature_name] = real_value
        
        return result
    
    def get_rl_thresholds_in_real_world(self):
        """
        Convert RL agent's standardized thresholds to real-world values
        This helps users understand what conditions trigger different actions
        """
        if self.scaler is None:
            return {}
        
        # RL agent thresholds (standardized)
        standardized_thresholds = {
            'high_temp': 2.0,      # High temperature threshold
            'very_high_temp': 1.5, # Very high temperature
            'low_soc': -10.0,      # Low SoC threshold
            'high_soc': 0.0,       # High SoC threshold
            'critical_low_soc': -15.0  # Critical low SoC
        }
        
        # Convert to real-world values
        real_world_thresholds = {}
        
        # Temperature thresholds
        temp_idx = 2  # temperature is index 2
        real_world_thresholds['high_temp_celsius'] = (
            standardized_thresholds['high_temp'] * self.scaler.scale_[temp_idx] + self.scaler.mean_[temp_idx]
        )
        real_world_thresholds['very_high_temp_celsius'] = (
            standardized_thresholds['very_high_temp'] * self.scaler.scale_[temp_idx] + self.scaler.mean_[temp_idx]
        )
        
        # SoC thresholds
        soc_idx = 3  # soc is index 3
        real_world_thresholds['low_soc_percent'] = (
            standardized_thresholds['low_soc'] * self.scaler.scale_[soc_idx] + self.scaler.mean_[soc_idx]
        ) * 100  # Convert to percentage
        
        real_world_thresholds['high_soc_percent'] = (
            standardized_thresholds['high_soc'] * self.scaler.scale_[soc_idx] + self.scaler.mean_[soc_idx]
        ) * 100
        
        real_world_thresholds['critical_low_soc_percent'] = (
            standardized_thresholds['critical_low_soc'] * self.scaler.scale_[soc_idx] + self.scaler.mean_[soc_idx]
        ) * 100
        
        return real_world_thresholds

# Example usage and testing
if __name__ == "__main__":
    # Test the transformer
    transformer = TelemetryTransformer()
    
    # Example real-world telemetry
    test_telemetry = {
        'voltage': 3.7,
        'current': 2.5,
        'temperature': 35.0,
        'soc': 0.6,  # 60%
        'ambient_temp': 25.0,
        'humidity': 0.5,
        'charge_mode': 'fast'
    }
    
    # Transform to standardized features
    standardized = transformer.transform_telemetry(test_telemetry)
    print("Standardized features:", standardized)
    
    # Get real-world thresholds
    thresholds = transformer.get_rl_thresholds_in_real_world()
    print("RL thresholds in real-world units:", thresholds)
