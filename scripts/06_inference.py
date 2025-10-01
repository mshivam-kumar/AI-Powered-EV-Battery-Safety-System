#!/usr/bin/env python3
"""
Step 6: Inference and Deployment Pipeline
Real-time inference and deployment system
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatteryTelemetry:
    """Battery telemetry data structure"""
    def __init__(self, voltage: float, current: float, temperature: float, 
                 soc: float, ambient_temp: float, humidity: float, charge_mode: str):
        self.voltage = voltage
        self.current = current
        self.temperature = temperature
        self.soc = soc
        self.ambient_temp = ambient_temp
        self.humidity = humidity
        self.charge_mode = charge_mode
        self.timestamp = datetime.now()

class InferenceEngine:
    """Real-time inference engine for battery anomaly detection"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_scaler = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        # Load Isolation Forest
        iforest_file = self.models_dir / "isolation_forest.pkl"
        if iforest_file.exists():
            with open(iforest_file, 'rb') as f:
                self.models['isolation_forest'] = pickle.load(f)
            logger.info("Loaded Isolation Forest")
        
        # Load supervised models
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            model_file = self.models_dir / f"{model_name}.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name}")
        
        # Load RL Agent
        q_table_file = self.models_dir / "q_table.pkl"
        rl_metadata_file = self.models_dir / "rl_metadata.json"
        if q_table_file.exists() and rl_metadata_file.exists():
            with open(q_table_file, 'rb') as f:
                self.models['q_table'] = pickle.load(f)
            with open(rl_metadata_file, 'r') as f:
                self.models['rl_metadata'] = json.load(f)
            logger.info("Loaded RL Agent")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def extract_features(self, telemetry: BatteryTelemetry) -> np.ndarray:
        """Extract 16 features from telemetry data"""
        # Basic features
        voltage = telemetry.voltage
        current = telemetry.current
        temperature = telemetry.temperature
        soc = telemetry.soc
        ambient_temp = telemetry.ambient_temp
        humidity = telemetry.humidity
        charge_mode = telemetry.charge_mode
        
        # Derived features
        power = voltage * abs(current)
        c_rate = abs(current)
        temp_diff = temperature - ambient_temp
        voltage_soc_ratio = voltage / (soc + 0.001)
        
        # Thermal stress
        ideal_temp = 25
        temp_range = 30
        thermal_stress = 1 - abs(temperature - ideal_temp) / temp_range
        thermal_stress = max(0, min(1, thermal_stress))
        
        # Environmental stress
        env_stress = (ambient_temp / 50.0) * humidity
        
        # Create feature vector
        features = np.array([
            voltage, current, temperature, soc, ambient_temp, humidity,
            power, c_rate, temp_diff, voltage_soc_ratio, thermal_stress,
            0, 0, 0, env_stress  # Gradients set to 0 for single sample
        ])
        
        return features
    
    def predict_anomaly(self, telemetry: BatteryTelemetry) -> Dict[str, Any]:
        """Predict anomaly using ensemble of models"""
        # Extract features
        features = self.extract_features(telemetry)
        features = features.reshape(1, -1)
        
        predictions = {}
        weights = {}
        
        # Isolation Forest prediction
        if 'isolation_forest' in self.models:
            iforest = self.models['isolation_forest']
            anomaly_score = iforest.decision_function(features)[0]
            is_anomaly = anomaly_score < 0
            predictions['isolation_forest'] = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'confidence': float(abs(anomaly_score))
            }
            weights['isolation_forest'] = 0.4
        
        # Supervised model predictions
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            if model_name in self.models:
                model = self.models[model_name]
                pred = model.predict(features)[0]
                pred_proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = {
                    'is_anomaly': bool(pred == 0),
                    'confidence': float(pred_proba[0] if pred_proba is not None else 0.5)
                }
                weights[model_name] = 0.2
        
        # RL Agent prediction
        if 'q_table' in self.models and 'rl_metadata' in self.models:
            q_table = self.models['q_table']
            metadata = self.models['rl_metadata']
            
            # Discretize state
            soc_bins = metadata['soc_bins']
            temp_bins = metadata['temp_bins']
            ambient_bins = metadata['ambient_bins']
            actions = metadata['actions']
            
            soc_bin = min(int(telemetry.soc * (len(soc_bins) - 1)), len(soc_bins) - 2)
            temp_bin = min(int(telemetry.temperature * (len(temp_bins) - 1)), len(temp_bins) - 2)
            ambient_bin = min(int(telemetry.ambient_temp * (len(ambient_bins) - 1)), len(ambient_bins) - 2)
            
            # Get Q-values
            q_values = q_table[soc_bin, temp_bin, ambient_bin]
            best_action_idx = np.argmax(q_values)
            best_action = actions[best_action_idx]
            
            # Convert action to anomaly prediction
            is_anomaly = (best_action == 'pause')
            confidence = float(q_values[best_action_idx] / np.max(q_values)) if np.max(q_values) > 0 else 0.5
            
            predictions['rl_agent'] = {
                'is_anomaly': is_anomaly,
                'recommended_action': best_action,
                'confidence': confidence,
                'q_values': q_values.tolist()
            }
            weights['rl_agent'] = 0.2
        
        # Ensemble prediction
        ensemble_prediction = self.create_ensemble_prediction(predictions, weights)
        
        return {
            'timestamp': telemetry.timestamp.isoformat(),
            'telemetry': {
                'voltage': telemetry.voltage,
                'current': telemetry.current,
                'temperature': telemetry.temperature,
                'soc': telemetry.soc,
                'ambient_temp': telemetry.ambient_temp,
                'humidity': telemetry.humidity,
                'charge_mode': telemetry.charge_mode
            },
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction
        }
    
    def create_ensemble_prediction(self, predictions: Dict[str, Dict], weights: Dict[str, float]) -> Dict[str, Any]:
        """Create ensemble prediction from individual model predictions"""
        if not predictions:
            return {'is_anomaly': False, 'confidence': 0.5, 'reason': 'No models available'}
        
        # Weighted voting
        total_weight = 0
        anomaly_votes = 0
        confidence_sum = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                total_weight += weight
                
                if pred['is_anomaly']:
                    anomaly_votes += weight
                
                confidence_sum += pred['confidence'] * weight
        
        # Normalize weights
        if total_weight > 0:
            anomaly_ratio = anomaly_votes / total_weight
            avg_confidence = confidence_sum / total_weight
        else:
            anomaly_ratio = 0.5
            avg_confidence = 0.5
        
        # Determine final prediction
        is_anomaly = anomaly_ratio > 0.5
        confidence = avg_confidence
        
        # Generate reason
        if is_anomaly:
            reason = "Multiple models detected anomalous behavior"
        else:
            reason = "Battery operating within normal parameters"
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_ratio': anomaly_ratio,
            'reason': reason,
            'model_count': len(predictions)
        }
    
    def calculate_bhi(self, telemetry: BatteryTelemetry) -> Dict[str, Any]:
        """Calculate Battery Health Index"""
        # Temperature stress
        ideal_temp = 25
        temp_stress = 1 - abs(telemetry.temperature - ideal_temp) / 30
        temp_stress = max(0, min(1, temp_stress))
        
        # Current stress
        current_stress = 1 - (abs(telemetry.current) - 0.5) / 2
        current_stress = max(0, min(1, current_stress))
        
        # SoC stress
        if 0.3 <= telemetry.soc <= 0.8:
            soc_stress = 1.0
        else:
            soc_stress = 0.5
        
        # Ambient stress
        ambient_stress = 1 - (telemetry.ambient_temp - 35) / 20 - 0.5 * telemetry.humidity
        ambient_stress = max(0, min(1, ambient_stress))
        
        # BHI (geometric mean)
        bhi = (temp_stress * current_stress * soc_stress * ambient_stress) ** 0.25
        
        # Severity
        if bhi >= 0.75:
            severity = "low"
        elif bhi >= 0.5:
            severity = "medium"
        else:
            severity = "high"
        
        return {
            'bhi': float(bhi),
            'severity': severity,
            'components': {
                'temp_stress': float(temp_stress),
                'current_stress': float(current_stress),
                'soc_stress': float(soc_stress),
                'ambient_stress': float(ambient_stress)
            }
        }

class DeploymentSystem:
    """Deployment system for real-time battery monitoring"""
    
    def __init__(self, models_dir: str = "models"):
        self.inference_engine = InferenceEngine(models_dir)
        self.alert_threshold = 0.6
        self.alerts = []
    
    def process_telemetry(self, telemetry: BatteryTelemetry) -> Dict[str, Any]:
        """Process telemetry data and return comprehensive analysis"""
        logger.info(f"Processing telemetry: V={telemetry.voltage:.2f}V, T={telemetry.temperature:.1f}°C, SoC={telemetry.soc:.2f}")
        
        # Get anomaly prediction
        anomaly_result = self.inference_engine.predict_anomaly(telemetry)
        
        # Calculate BHI
        bhi_result = self.inference_engine.calculate_bhi(telemetry)
        
        # Check for alerts
        alerts = self.check_alerts(telemetry, anomaly_result, bhi_result)
        
        # Create comprehensive result
        result = {
            'timestamp': telemetry.timestamp.isoformat(),
            'telemetry': {
                'voltage': telemetry.voltage,
                'current': telemetry.current,
                'temperature': telemetry.temperature,
                'soc': telemetry.soc,
                'ambient_temp': telemetry.ambient_temp,
                'humidity': telemetry.humidity,
                'charge_mode': telemetry.charge_mode
            },
            'anomaly_detection': anomaly_result,
            'battery_health': bhi_result,
            'alerts': alerts,
            'recommendations': self.generate_recommendations(telemetry, anomaly_result, bhi_result)
        }
        
        return result
    
    def check_alerts(self, telemetry: BatteryTelemetry, anomaly_result: Dict, bhi_result: Dict) -> List[Dict[str, Any]]:
        """Check for safety alerts"""
        alerts = []
        
        # Temperature alerts
        if telemetry.temperature > 45:
            alerts.append({
                'type': 'critical',
                'message': f'High battery temperature: {telemetry.temperature:.1f}°C',
                'action': 'Stop charging immediately'
            })
        elif telemetry.temperature > 40:
            alerts.append({
                'type': 'warning',
                'message': f'Elevated battery temperature: {telemetry.temperature:.1f}°C',
                'action': 'Reduce charging rate'
            })
        
        # SoC alerts
        if telemetry.soc > 0.95:
            alerts.append({
                'type': 'warning',
                'message': f'High SoC: {telemetry.soc:.1%}',
                'action': 'Stop charging to prevent overcharge'
            })
        elif telemetry.soc < 0.1:
            alerts.append({
                'type': 'warning',
                'message': f'Low SoC: {telemetry.soc:.1%}',
                'action': 'Charge battery immediately'
            })
        
        # Anomaly alerts
        if anomaly_result['ensemble_prediction']['is_anomaly']:
            alerts.append({
                'type': 'critical',
                'message': 'Anomalous behavior detected',
                'action': 'Check battery system'
            })
        
        # BHI alerts
        if bhi_result['severity'] == 'high':
            alerts.append({
                'type': 'critical',
                'message': f'High battery stress: BHI={bhi_result["bhi"]:.2f}',
                'action': 'Reduce charging rate and check conditions'
            })
        elif bhi_result['severity'] == 'medium':
            alerts.append({
                'type': 'warning',
                'message': f'Moderate battery stress: BHI={bhi_result["bhi"]:.2f}',
                'action': 'Monitor battery closely'
            })
        
        return alerts
    
    def generate_recommendations(self, telemetry: BatteryTelemetry, anomaly_result: Dict, bhi_result: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Temperature-based recommendations
        if telemetry.temperature > 40:
            recommendations.append("Reduce charging rate due to high temperature")
        elif telemetry.temperature < 10:
            recommendations.append("Warm up battery before charging")
        
        # SoC-based recommendations
        if telemetry.soc > 0.8:
            recommendations.append("Consider reducing charging rate for battery longevity")
        elif telemetry.soc < 0.2:
            recommendations.append("Charge battery to maintain health")
        
        # Anomaly-based recommendations
        if anomaly_result['ensemble_prediction']['is_anomaly']:
            recommendations.append("Investigate potential battery issues")
        
        # BHI-based recommendations
        if bhi_result['severity'] == 'high':
            recommendations.append("Reduce environmental stress on battery")
        elif bhi_result['severity'] == 'medium':
            recommendations.append("Monitor battery health closely")
        
        # RL Agent recommendations
        if 'rl_agent' in anomaly_result['individual_predictions']:
            rl_action = anomaly_result['individual_predictions']['rl_agent']['recommended_action']
            if rl_action == 'pause':
                recommendations.append("RL Agent recommends pausing charging")
            elif rl_action == 'slow':
                recommendations.append("RL Agent recommends slow charging")
            elif rl_action == 'fast':
                recommendations.append("RL Agent recommends fast charging")
        
        return recommendations
    
    def run_demo(self):
        """Run demonstration with sample telemetry data"""
        logger.info("Running deployment demo...")
        
        # Sample telemetry data
        sample_telemetry = [
            BatteryTelemetry(3.8, 2.5, 25, 0.5, 22, 0.6, 'fast'),
            BatteryTelemetry(4.0, 1.0, 35, 0.7, 30, 0.8, 'slow'),
            BatteryTelemetry(3.9, 0.0, 45, 0.9, 35, 0.7, 'pause'),
            BatteryTelemetry(3.7, 3.0, 50, 0.3, 40, 0.9, 'fast'),
        ]
        
        results = []
        for i, telemetry in enumerate(sample_telemetry):
            logger.info(f"\n--- Sample {i+1} ---")
            result = self.process_telemetry(telemetry)
            results.append(result)
            
            # Print summary
            print(f"\nSample {i+1} Results:")
            print(f"  Anomaly: {result['anomaly_detection']['ensemble_prediction']['is_anomaly']}")
            print(f"  BHI: {result['battery_health']['bhi']:.3f} ({result['battery_health']['severity']})")
            print(f"  Alerts: {len(result['alerts'])}")
            print(f"  Recommendations: {len(result['recommendations'])}")
        
        # Save demo results
        demo_file = Path("results") / "demo_results.json"
        demo_file.parent.mkdir(exist_ok=True)
        with open(demo_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Demo results saved to {demo_file}")
        
        return results

def main():
    """Main function to run inference demo"""
    deployment = DeploymentSystem()
    results = deployment.run_demo()
    
    if results is not None:
        print(f"\n✅ Inference demo completed!")
        print(f"📊 Processed {len(results)} samples")
        print(f"📁 Demo results saved to: results/demo_results.json")
    else:
        print("❌ Inference demo failed!")

if __name__ == "__main__":
    main()
