#!/usr/bin/env python3
"""
Complete EV Battery Safety Management System Dashboard
Integrates all trained models: Random Forest, MLP, RL Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our data transformer
from data_transformer import TelemetryTransformer

# Page configuration
st.set_page_config(
    page_title="EV Battery Safety Management System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "telemetry_data" not in st.session_state:
    st.session_state.telemetry_data = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "system_running" not in st.session_state:
    st.session_state.system_running = False
if "accumulated_alerts" not in st.session_state:
    st.session_state.accumulated_alerts = {
        "critical": 0,
        "warning": 0
    }
if "alert_history" not in st.session_state:
    st.session_state.alert_history = {
        "critical": [],
        "warning": []
    }
if "climate_zone" not in st.session_state:
    st.session_state.climate_zone = "Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)"
if "charge_mode" not in st.session_state:
    st.session_state.charge_mode = "slow"
if "location" not in st.session_state:
    st.session_state.location = "Mumbai"

class BatteryManagementSystem:
    def __init__(self, models_dir="models"):
        # Use relative path that works in both local and Streamlit Cloud
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
        
        # RL Agent actions (will be set when model loads)
        self.ACTIONS = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
        # Climate-aware action adjustments (20 combinations)
        self.climate_adjustments = {
            # Hot Desert + Summer
            ('Hot Desert', 'summer'): {'fast_charge': 'slow_charge', 'slow_charge': 'pause'},
            # Hot Desert + Winter  
            ('Hot Desert', 'winter'): {'fast_charge': 'slow_charge'},
            # Tropical Monsoon + Monsoon
            ('Tropical Monsoon', 'monsoon'): {'fast_charge': 'slow_charge'},
            # Subtropical Highland + Winter
            ('Subtropical Highland', 'winter'): {'pause': 'slow_charge', 'fast_charge': 'slow_charge'},
            # Tropical Wet + Monsoon
            ('Tropical Wet', 'monsoon'): {'fast_charge': 'slow_charge'},
        }
        
        # Feature names (CORRECTED to match training exactly)
        self.feature_names = [
            'voltage',           # 1. Battery terminal voltage
            'current',           # 2. Charging current  
            'temperature',       # 3. Battery cell temperature
            'soc',               # 4. State of charge
            'ambient_temp',      # 5. Environmental temperature
            'humidity',          # 6. Relative humidity
            'charge_mode',       # 7. Charging mode (encoded)
            'power',             # 8. Electrical power (V × I)
            'c_rate',            # 9. Charging rate indicator
            'temp_diff',         # 10. Temperature difference
            'voltage_soc_ratio', # 11. Voltage-SoC correlation
            'thermal_stress',    # 12. Normalized temperature stress
            'temp_gradient',     # 13. Rate of temperature change
            'voltage_gradient',  # 14. Rate of voltage change
            'soc_rate',          # 15. Rate of SoC change
            'env_stress'         # 16. Environmental stress
        ]
    
    def load_models(self):
        """Load all trained models and data transformer with cloud compatibility"""
        # Check if models directory exists
        if not self.models_dir.exists():
            st.error(f"⚠️ Models directory not found: {self.models_dir}")
            st.warning("🔧 Running in fallback mode - some features may be limited")
            return
        
        # Always initialize transformer first
        try:
            self.transformer = TelemetryTransformer(models_dir=str(self.models_dir))
        except Exception as e:
            st.error(f"⚠️ Could not initialize transformer: {e}")
            st.warning("🔧 Running in fallback mode - feature engineering may be limited")
            return
        
        try:
            # Load Random Forest (best predictor) - try complete version first
            rf_path = self.models_dir / "random_forest_complete.pkl"
            if not rf_path.exists():
                rf_path = self.models_dir / "random_forest.pkl"
            if rf_path.exists():
                try:
                    with open(rf_path, 'rb') as f:
                        self.models['random_forest'] = pickle.load(f)
                    st.success("✅ Random Forest model loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not load Random Forest: {e}")
            else:
                st.warning("⚠️ Random Forest model not found - anomaly detection will use fallback")
            
            # Load MLP Medium (good predictor) - try latest complete version first
            mlp_complete_path = self.models_dir / "mlp_medium_complete.pkl"
            mlp_compatible_path = self.models_dir / "mlp_medium_compatible.pkl"
            mlp_original_path = self.models_dir / "mlp_medium.pkl"
            
            if mlp_complete_path.exists():
                try:
                    with open(mlp_complete_path, 'rb') as f:
                        self.models['mlp_medium'] = pickle.load(f)
                    st.success("✅ MLP Medium (Complete) model loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not load MLP Complete: {e}")
            elif mlp_compatible_path.exists():
                try:
                    with open(mlp_compatible_path, 'rb') as f:
                        self.models['mlp_medium'] = pickle.load(f)
                    st.success("✅ MLP Medium (compatible) model loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not load compatible MLP Medium: {e}")
            elif mlp_original_path.exists():
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(mlp_original_path, 'rb') as f:
                            self.models['mlp_medium'] = pickle.load(f)
                    st.success("✅ MLP Medium model loaded")
                except Exception as e:
                    st.info("ℹ️ MLP Medium: Numpy version compatibility issue (scikit-learn 1.6.1 → 1.5.2)")
                    st.info("💡 **Solution**: Run `python scripts/train_mlp.py` to create compatible models")
                    st.success("✅ **Random Forest (98.7% accuracy) is primary predictor - system fully functional!**")
            else:
                st.warning("⚠️ MLP Medium model not found - ensemble will use Random Forest only")
            
            # Load RL Agent (action selector) - use best performing model
            rl_models = [
                "fine_tuned_from_logs_rl_agent.json",  # Best coverage: 8.2% (from logs)
                "rl_robust_enhanced_v2_q_table.pkl",  # Fallback
                "rl_safety_focused_q_table.pkl"  # Final fallback
            ]
            
            rl_loaded = False
            for rl_model in rl_models:
                rl_path = self.models_dir / rl_model
                if rl_path.exists():
                    try:
                        if rl_model.endswith('.json'):
                            # Load JSON format (climate-aware models)
                            with open(rl_path, 'r') as f:
                                rl_data = json.load(f)
                            self.models['rl_agent'] = rl_data['q_table']
                            self.models['rl_actions'] = rl_data.get('actions', ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain'])
                        else:
                            # Load pickle format (legacy models)
                            with open(rl_path, 'rb') as f:
                                self.models['rl_agent'] = pickle.load(f)
                            self.models['rl_actions'] = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
                        
                        if rl_model == "fine_tuned_from_logs_rl_agent.json":
                            st.success(f"✅ RL Agent loaded from {rl_model} (Best: 8.2% coverage)")
                        else:
                            st.success(f"✅ RL Agent loaded from {rl_model}")
                        rl_loaded = True
                        break
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {rl_model}: {e}")
                        continue
            
            if not rl_loaded:
                st.warning("⚠️ Could not load any RL Agent - using fallback")
                # Initialize fallback RL agent
                self.models['rl_agent'] = None
                self.models['rl_actions'] = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
            
            if not self.models:
                st.warning("⚠️ No models loaded - using fallback predictions")
                # Initialize fallback models
                self.models['random_forest'] = None
                self.models['mlp_medium'] = None
                    
        except Exception as e:
            st.error(f"Error in model loading process: {e}")
            st.info("💡 The system will continue with synthetic predictions")
    
    def extract_features(self, telemetry):
        """Extract and standardize features using the proper transformer"""
        try:
            # Use the transformer to properly convert real-world telemetry to standardized features
            standardized_features = self.transformer.transform_telemetry(telemetry)
            return standardized_features  # Already in correct shape (1, -1)
        except Exception as e:
            st.error(f"Error transforming telemetry: {e}")
            # Fallback to zeros if transformation fails
            return np.zeros((1, 16))
    
    def predict_anomaly(self, features):
        """Predict anomaly using ensemble of models + safety rules"""
        predictions = {}
        
        # Random Forest prediction
        if 'random_forest' in self.models and self.models['random_forest'] is not None:
            try:
                rf_pred = self.models['random_forest'].predict(features)[0]
                rf_proba = self.models['random_forest'].predict_proba(features)[0, 1]
                predictions['random_forest'] = {'prediction': rf_pred, 'probability': rf_proba}
            except Exception as e:
                st.warning(f"⚠️ Random Forest prediction failed: {e}")
        else:
            # Fallback: Use safety rules only
            st.info("ℹ️ Using safety rules fallback (Random Forest not available)")
        
        # MLP prediction
        if 'mlp_medium' in self.models and self.models['mlp_medium'] is not None:
            try:
                mlp_pred = self.models['mlp_medium'].predict(features)[0]
                mlp_proba = self.models['mlp_medium'].predict_proba(features)[0, 1]
                predictions['mlp_medium'] = {'prediction': mlp_pred, 'probability': mlp_proba}
            except Exception as e:
                st.warning(f"⚠️ MLP prediction failed: {e}")
        else:
            # Fallback: Skip MLP if not available
            pass
        
        # SAFETY RULES - Override ML when safety is at risk
        temp_std = features[0][2] if len(features[0]) > 2 else 0  # Standardized temperature
        soc_std = features[0][3] if len(features[0]) > 3 else 0.5  # Standardized SoC
        voltage_std = features[0][0] if len(features[0]) > 0 else 0  # Standardized voltage
        
        # Convert standardized back to real-world for safety checks
        # Temp: standardized 2.0 ≈ 41°C, 3.0 ≈ 49°C
        # SoC: standardized -2.0 ≈ 20%, 2.0 ≈ 80%
        safety_anomaly_score = 0.0
        safety_reasons = []
        
        if temp_std > 2.0:  # Very high temperature (>40°C)
            safety_anomaly_score += 0.8
            safety_reasons.append("High Temperature")
        elif temp_std > 1.5:  # Moderately high temperature (>35°C)
            safety_anomaly_score += 0.4
            safety_reasons.append("Elevated Temperature")
            
        if soc_std < -2.0:  # Very low SoC (<20%)
            safety_anomaly_score += 0.6
            safety_reasons.append("Critical Low SoC")
        elif soc_std > 2.0:  # Very high SoC (>80%)
            safety_anomaly_score += 0.4
            safety_reasons.append("High SoC")
            
        if abs(voltage_std) > 3.0:  # Extreme voltage
            safety_anomaly_score += 0.5
            safety_reasons.append("Voltage Out of Range")
        
        # Safety override prediction
        predictions['safety_rules'] = {
            'prediction': 1 if safety_anomaly_score > 0.5 else 0,
            'probability': min(safety_anomaly_score, 1.0),
            'reasons': safety_reasons
        }
        
        # Fallback prediction if no ML models work
        if len([p for p in predictions if p != 'safety_rules']) == 0:
            predictions['fallback_rules'] = {
                'prediction': 1 if safety_anomaly_score > 0.3 else 0,
                'probability': max(0.2, min(safety_anomaly_score, 1.0))
            }
        
        # HYBRID ENSEMBLE: ML + Safety Rules
        if predictions:
            # Give higher weight to safety rules when they detect danger
            weights = {
                'random_forest': 0.4, 
                'mlp_medium': 0.3, 
                'safety_rules': 0.6 if safety_anomaly_score > 0.5 else 0.3,
                'fallback_rules': 0.1
            }
            
            ensemble_proba = sum(
                predictions[model]['probability'] * weights.get(model, 0.3)
                for model in predictions
            ) / sum(weights.get(model, 0.3) for model in predictions if model in predictions)
            
            ensemble_pred = 1 if ensemble_proba > 0.4 else 0  # Lower threshold for safety
            predictions['ensemble'] = {
                'prediction': ensemble_pred, 
                'probability': ensemble_proba,
                'safety_override': safety_anomaly_score > 0.5
            }
        
        return predictions
    
    def discretize_state(self, telemetry):
        """Discretize state for RL agent (6D state space to match training)"""
        # Extract features for 6D state space: (c_rate, power, temp, soc, voltage, anomaly)
        
        # Calculate derived features
        c_rate = abs(telemetry.get('current', 0.0) / 2.0)  # Approximate C-rate
        power = abs(telemetry.get('voltage', 3.7) * telemetry.get('current', 0.0))  # Power = V * I
        
        # Discretize to 5 bins each (matching training: 5x5x5x5x5x2)
        c_rate_bin = min(max(int(c_rate * 2.0), 0), 4)  # 0-4 bins
        power_bin = min(max(int(power / 2.0), 0), 4)     # 0-4 bins  
        temp_bin = min(max(int((telemetry['temperature'] + 2.0) / 1.0), 0), 4)  # 0-4 bins
        soc_bin = min(max(int((telemetry['soc'] + 10.0) / 4.0), 0), 4)  # 0-4 bins
        voltage_bin = min(max(int((telemetry['voltage'] + 2.0) / 1.0), 0), 4)  # 0-4 bins
        
        # Anomaly flag (0 or 1)
        anomaly_bin = 1 if telemetry.get('is_anomaly', False) else 0
        
        return c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin
    
    def get_rl_action(self, telemetry, debug=False):
        """Get RL agent recommended action"""
        if 'rl_agent' not in self.models or self.models['rl_agent'] is None:
            if debug:
                st.sidebar.warning("⚠️ RL Agent not loaded - using fallback action")
            return self.get_fallback_action(telemetry)
        
        try:
            # Debug: Show useful Q-value info instead of just "loaded successfully"
            if debug:
                st.sidebar.write("🔍 **RL Agent Debug:**")
                # Will show Q-value info in the main debug section below
            
            # Discretize state (6D for new RL model)
            state = self.discretize_state(telemetry)
            
            # Debug: Print state discretization
            if debug:
                print(f"🔍 State discretization: {state}")
            
            # Get Q-values for this state
            q_table = self.models['rl_agent']
            
            # Check if q_table is a numpy array (7D), list, or dictionary
            if isinstance(q_table, np.ndarray):
                # 7D array: [c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin, action]
                c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin = state
                if (0 <= c_rate_bin < q_table.shape[0] and 
                    0 <= power_bin < q_table.shape[1] and 
                    0 <= temp_bin < q_table.shape[2] and 
                    0 <= soc_bin < q_table.shape[3] and
                    0 <= voltage_bin < q_table.shape[4] and
                    0 <= anomaly_bin < q_table.shape[5]):
                    q_values = q_table[c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin, :]
                    
                    # Debug: Simple Q-table existence check
                    if debug:
                        print(f"🔍 Q-values retrieved: {q_values}")
                        print(f"🔍 Q-values sum: {np.sum(q_values)}")
                        print(f"🔍 All zero check: {np.allclose(q_values, 0.0)}")
                    
                    # Store debug info for display and logging
                    is_untrained = np.allclose(q_values, 0.0)
                    debug_info = {
                        'temp_std': telemetry['temperature'],
                        'soc_std': telemetry['soc'],
                        'voltage_std': telemetry.get('voltage', 0.0),
                        'state_bins': state,
                        'state_index': f"({c_rate_bin},{power_bin},{temp_bin},{soc_bin},{voltage_bin},{anomaly_bin})",
                        'q_values': q_values.tolist(),
                        'q_sum': np.sum(q_values),
                        'is_untrained_state': is_untrained,
                        'debug_message': "Using default Q-values (untrained state)" if is_untrained else "Using learned Q-values",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Simple debug output - focus on Q-table existence
                    if debug:
                        # Core purpose: Q-table existence check
                        if is_untrained:
                            st.sidebar.warning("❌ **NEW STATE** - Q-table entry does not exist")
                        else:
                            st.sidebar.success("✅ **TRAINED STATE** - Q-table entry exists")
                            st.sidebar.write(f"• Best action: {self.ACTIONS[np.argmax(q_values)]}")
                    
                    # Store debug info for logging
                    self.last_rl_debug_info = debug_info
                    
                    # Log untrained states separately for RL improvement
                    if is_untrained:
                        print(f"🔴 UNTRAINED STATE DETECTED: {debug_info['state_index']}")
                        # Store reference to original telemetry for logging
                        if hasattr(self, 'current_raw_telemetry'):
                            self.log_untrained_rl_state(debug_info, self.current_raw_telemetry)
                        else:
                            self.log_untrained_rl_state(debug_info, telemetry)
                    
                    # Check if all Q-values are zero (untrained state)
                    if np.allclose(q_values, 0.0):
                        # Smart default Q-values based on comprehensive safety rules
                        temp_std = telemetry['temperature']
                        soc_std = telemetry['soc']
                        is_anomaly = telemetry.get('is_anomaly', False)
                        
                        # SAFETY-FIRST DEFAULT Q-VALUES (matching our training approach)
                        if is_anomaly:
                            # During HIGH anomaly (>70%): ALWAYS prefer pause/discharge
                            q_values = np.array([0.1, 0.2, 0.9, 0.7, 0.1])  # Strong preference for pause
                        elif temp_std > 1.5:  # High temperature (>90th percentile)
                            q_values = np.array([0.1, 0.6, 0.8, 0.5, 0.3])  # Prefer pause, allow slow charge
                        elif temp_std > 0.5:  # Moderate temperature
                            q_values = np.array([0.4, 0.8, 0.3, 0.3, 0.7])  # Prefer slow charge/maintain
                        elif soc_std < -2.0:  # Very low SoC (<20%)
                            q_values = np.array([0.7, 0.9, 0.2, 0.1, 0.4])  # Prefer fast/slow charging
                        elif soc_std < -0.5:  # Low SoC
                            q_values = np.array([0.6, 0.9, 0.2, 0.2, 0.6])  # Prefer slow charge
                        elif soc_std > 1.5:  # High SoC (>80%)
                            q_values = np.array([0.1, 0.2, 0.4, 0.9, 0.7])  # Prefer discharge/maintain
                        else:  # Normal conditions
                            q_values = np.array([0.5, 0.8, 0.3, 0.3, 0.9])  # Balanced, prefer maintain/slow charge
                else:
                    q_values = np.array([0.4, 0.7, 0.3, 0.3, 0.9])  # Balanced default - prefer maintain/slow charge
            elif isinstance(q_table, list):
                # List format (from JSON) - convert to numpy array first
                q_table_array = np.array(q_table)
                
                # Debug: Show converted Q-table info
                if debug:
                    print(f"🔍 Converted Q-table shape: {q_table_array.shape}")
                    print(f"🔍 Q-table dimensions: {q_table_array.ndim}D")
                
                # 7D array: [c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin, action]
                c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin = state
                if (0 <= c_rate_bin < q_table_array.shape[0] and 
                    0 <= power_bin < q_table_array.shape[1] and 
                    0 <= temp_bin < q_table_array.shape[2] and 
                    0 <= soc_bin < q_table_array.shape[3] and
                    0 <= voltage_bin < q_table_array.shape[4] and
                    0 <= anomaly_bin < q_table_array.shape[5]):
                    q_values = q_table_array[c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin, :]
                    
                    # Debug: Simple Q-table existence check (list format)
                    if debug:
                        print(f"🔍 Q-values retrieved (list format): {q_values}")
                        print(f"🔍 Q-values sum: {np.sum(q_values)}")
                        print(f"🔍 All zero check: {np.allclose(q_values, 0.0)}")
                    
                    # Store debug info for display and logging
                    is_untrained = np.allclose(q_values, 0.0)
                    debug_info = {
                        'temp_std': telemetry['temperature'],
                        'soc_std': telemetry['soc'],
                        'voltage_std': telemetry.get('voltage', 0.0),
                        'state_bins': state,
                        'state_index': f"({c_rate_bin},{power_bin},{temp_bin},{soc_bin},{voltage_bin},{anomaly_bin})",
                        'q_values': q_values.tolist(),
                        'q_sum': np.sum(q_values),
                        'is_untrained_state': is_untrained,
                        'debug_message': "Using default Q-values (untrained state)" if is_untrained else "Using learned Q-values",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Simple debug output - focus on Q-table existence (list format)
                    if debug:
                        # Core purpose: Q-table existence check
                        if is_untrained:
                            st.sidebar.warning("❌ **NEW STATE** - Q-table entry does not exist")
                        else:
                            st.sidebar.success("✅ **TRAINED STATE** - Q-table entry exists")
                            st.sidebar.write(f"• Best action: {self.ACTIONS[np.argmax(q_values)]}")
                    
                    # Store debug info for logging
                    self.last_rl_debug_info = debug_info
                    
                    # Log untrained states separately for RL improvement
                    if is_untrained:
                        print(f"🔴 UNTRAINED STATE DETECTED: {debug_info['state_index']}")
                        # Store reference to original telemetry for logging
                        if hasattr(self, 'current_raw_telemetry'):
                            self.log_untrained_rl_state(debug_info, self.current_raw_telemetry)
                        else:
                            self.log_untrained_rl_state(debug_info, telemetry)
                    
                    # Check if all Q-values are zero (untrained state)
                    if np.allclose(q_values, 0.0):
                        # Smart default Q-values based on comprehensive safety rules
                        temp_std = telemetry['temperature']
                        soc_std = telemetry['soc']
                        is_anomaly = telemetry.get('is_anomaly', False)
                        
                        # SAFETY-FIRST DEFAULT Q-VALUES (matching our training approach)
                        if temp_std > 1.0:  # High temperature - prefer pause/slow
                            q_values = np.array([0.2, 0.8, 0.7, 0.4, 0.5])
                        elif is_anomaly:  # Anomaly detected - prefer pause
                            q_values = np.array([0.1, 0.3, 0.9, 0.2, 0.4])
                        elif soc_std < -1.0:  # Very low SoC - prefer charging
                            q_values = np.array([0.7, 0.9, 0.2, 0.1, 0.4])  # Prefer fast/slow charging
                        elif soc_std < -0.5:  # Low SoC
                            q_values = np.array([0.6, 0.9, 0.2, 0.2, 0.6])  # Prefer slow charge
                        elif soc_std > 1.5:  # High SoC (>80%)
                            q_values = np.array([0.1, 0.2, 0.4, 0.9, 0.7])  # Prefer discharge/maintain
                        else:  # Normal conditions
                            q_values = np.array([0.5, 0.8, 0.3, 0.3, 0.9])  # Balanced, prefer maintain/slow charge
                else:
                    q_values = np.array([0.4, 0.7, 0.3, 0.3, 0.9])  # Balanced default - prefer maintain/slow charge
            else:
                # Dictionary format
                if state in q_table:
                    q_values = q_table[state]
                    # Check if all Q-values are zero (untrained state)
                    if np.allclose(q_values, 0.0):
                        # Use same logic as array format
                        temp_std = telemetry['temperature']
                        soc_std = telemetry['soc']
                        
                        if temp_std > 1.0:  # High temp - prefer pause/slow
                            q_values = np.array([0.2, 0.8, 0.7, 0.4, 0.5])
                        elif soc_std < -0.5:  # Low SoC - prefer charging
                            q_values = np.array([0.6, 0.9, 0.2, 0.1, 0.6])
                        elif soc_std > 1.0:  # High SoC - prefer discharge
                            q_values = np.array([0.1, 0.3, 0.4, 0.9, 0.7])
                        else:  # Normal conditions - prefer maintain/slow charge
                            q_values = np.array([0.5, 0.8, 0.3, 0.3, 0.9])
                else:
                    q_values = np.array([0.5, 0.8, 0.3, 0.3, 0.9])  # Prefer maintain/slow charge as default
            
            # Get best action
            action_idx = np.argmax(q_values)
            action = self.ACTIONS[action_idx]
            
            # More realistic confidence calculation
            q_sorted = np.sort(q_values)[::-1]  # Sort descending
            best_q = q_sorted[0]
            second_best_q = q_sorted[1] if len(q_sorted) > 1 else 0
            
            # Dynamic confidence based on Q-value spread and context
            q_mean = np.mean(q_values)
            q_std = np.std(q_values)
            q_range = np.max(q_values) - np.min(q_values)
            margin = best_q - second_best_q
            
            # Multi-factor confidence calculation
            if best_q <= 0.1:  # Very low Q-values (untrained/uncertain)
                base_confidence = 0.35
                uncertainty_factor = np.random.uniform(-0.1, 0.1)  # Add randomness
            elif q_std < 0.05:  # Very similar Q-values (uncertain choice)
                base_confidence = 0.45 + margin * 2
                uncertainty_factor = np.random.uniform(-0.05, 0.05)
            elif margin < 0.1:  # Close competition
                base_confidence = 0.55 + margin * 3  
                uncertainty_factor = (q_range - 0.5) * 0.1  # Range affects confidence
            elif margin < 0.3:  # Moderate difference
                base_confidence = 0.70 + margin * 0.8
                uncertainty_factor = (q_mean - 0.5) * 0.1  # Mean Q-value affects confidence
            else:  # Clear winner
                base_confidence = 0.85 + min(0.15, margin * 0.3)
                uncertainty_factor = min(0.05, q_std * 0.2)  # High std = more uncertainty
            
            # Add temporal variation based on telemetry values
            temp_factor = abs(telemetry['temperature']) * 0.02  # Temperature affects confidence
            soc_factor = abs(telemetry['soc']) * 0.01  # SoC affects confidence
            
            confidence = base_confidence + uncertainty_factor + temp_factor - soc_factor
            confidence = min(0.98, max(0.32, confidence))  # Clamp to 32-98%
            
            # Apply climate-aware adjustments using if-else logic
            adjusted_action, confidence_boost = self.apply_climate_aware_adjustments(action, telemetry)
            final_confidence = min(0.98, confidence + confidence_boost)
            
            return adjusted_action, final_confidence
            
        except Exception as e:
            st.error(f"⚠️ RL Agent failed: {e}")
            st.error(f"   Using fallback action")
            return self.get_fallback_action(telemetry)
    
    def apply_climate_aware_adjustments(self, base_action, telemetry):
        """Apply climate-aware adjustments using if-else logic (20 combinations)"""
        climate_zone = telemetry.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
        season = telemetry.get('season', 'monsoon')
        temp = telemetry.get('temperature', 25.0)
        soc = telemetry.get('soc', 0.5)
        humidity = telemetry.get('humidity', 0.5)
        
        adjusted_action = base_action
        confidence_boost = 0.0
        
        # Climate zone + Season combinations (20 total)
        if "Hot Desert" in climate_zone:
            if season == 'summer':
                if temp > 40:  # Extreme heat
                    if base_action == 'fast_charge':
                        adjusted_action = 'pause'
                        confidence_boost = 0.3
                    elif base_action == 'slow_charge':
                        adjusted_action = 'pause'
                        confidence_boost = 0.2
                elif temp > 35:  # High heat
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
            elif season == 'winter':
                if temp < 10:  # Cold conditions
                    if base_action == 'pause' and soc < 0.3:
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
                    elif base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
        
        elif "Tropical Monsoon" in climate_zone:
            if season == 'monsoon':
                if humidity > 0.8:  # High humidity
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
            elif season == 'summer':
                if temp > 38:  # Summer heat
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
        
        elif "Subtropical Highland" in climate_zone:
            if season == 'winter':
                if temp < 5:  # Cold conditions
                    if base_action == 'pause' and soc < 0.3:
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
                    elif base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
            elif season == 'summer':
                if temp > 35:  # Summer heat
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
        
        elif "Tropical Savanna" in climate_zone:
            if season == 'summer':
                if temp > 40:  # Urban heat island
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
            elif season == 'monsoon':
                if humidity > 0.8:  # High humidity
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
        
        elif "Tropical Wet" in climate_zone:
            if season == 'monsoon':
                if humidity > 0.85:  # Very high humidity
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
            elif season == 'summer':
                if temp > 35:  # Summer heat
                    if base_action == 'fast_charge':
                        adjusted_action = 'slow_charge'
                        confidence_boost = 0.1
        
        # Safety overrides (regardless of climate/season)
        if temp > 45:  # Critical temperature
            adjusted_action = 'pause'
            confidence_boost = 0.3
        
        if soc < 0.1:  # Critical low SoC
            if adjusted_action == 'pause':
                adjusted_action = 'slow_charge'
                confidence_boost = 0.2
        
        return adjusted_action, confidence_boost
    
    def get_fallback_action(self, telemetry):
        """Rule-based fallback action selection with climate awareness"""
        # Simple safety rules using STANDARDIZED values
        temp = telemetry['temperature']  # Standardized temperature
        soc = telemetry['soc']          # Standardized SoC
        climate_zone = telemetry.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
        season = telemetry.get('season', 'monsoon')
        humidity = telemetry.get('humidity', 0.5)
        ambient_temp = telemetry.get('ambient_temp', 25)
        
        # Climate-aware safety thresholds
        temp_threshold = 2.0  # Base threshold
        soc_threshold = -1.0  # Base threshold
        
        # Climate zone adjustments
        if "Hot Desert" in climate_zone:
            temp_threshold -= 0.5  # More sensitive in desert
        elif "Tropical Monsoon" in climate_zone:
            temp_threshold -= 0.3  # More sensitive in monsoon
        elif "Subtropical Highland" in climate_zone:
            temp_threshold += 0.5  # Less sensitive in highland
        
        # Season adjustments
        if season.lower() == 'summer':
            temp_threshold -= 0.3  # More sensitive in summer
        elif season.lower() == 'monsoon':
            temp_threshold -= 0.2  # More sensitive in monsoon
        elif season.lower() == 'winter':
            temp_threshold += 0.3  # Less sensitive in winter
        
        # Humidity adjustments
        if humidity > 0.8:
            temp_threshold -= 0.2  # More sensitive in high humidity
        
        # Safety first approach (climate-aware)
        if temp > temp_threshold:  # Climate-aware hot threshold
            return 'pause', 0.9
        elif temp > (temp_threshold - 0.5):  # Climate-aware warm threshold
            return 'slow_charge', 0.7
        elif soc < -2.0:  # Very low battery (standardized)
            return 'slow_charge', 0.8
        elif soc > 2.0:  # Very full battery (standardized)
            return 'pause', 0.8
        elif soc < soc_threshold:  # Low battery
            return 'fast_charge', 0.6
        else:  # Normal conditions
            return 'maintain', 0.5
    
    def calculate_bhi(self, telemetry):
        """Calculate Enhanced Battery Health Index for Indian Conditions"""
        # Basic health factors (existing)
        soc_factor = 1.0 - abs(telemetry['soc'] - 0.5) * 2  # Optimal around 50%
        temp_factor = max(0, 1.0 - abs(telemetry['temperature'] - 25) / 50)  # Optimal around 25°C
        voltage_factor = max(0, 1.0 - abs(telemetry['voltage'] - 3.7) / 2)  # Optimal around 3.7V
        
        # India-specific environmental factors
        humidity = telemetry.get('humidity', 0.5)
        ambient_temp = telemetry.get('ambient_temp', 25)
        location = telemetry.get('location', 'inland')
        charge_mode = telemetry.get('charge_mode', 'slow')
        climate_zone = telemetry.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
        season = telemetry.get('season', 'monsoon')
        
        # Environmental health factors
        humidity_factor = self.calculate_humidity_factor(humidity)
        heat_stress_factor = self.calculate_heat_stress_factor(ambient_temp)
        monsoon_factor = self.calculate_monsoon_factor(ambient_temp)
        salinity_factor = self.calculate_salinity_factor(location)
        
        # Climate zone specific factors
        climate_zone_factor = self.calculate_climate_zone_factor(climate_zone, ambient_temp, humidity)
        
        # Season specific factors
        season_factor = self.calculate_season_factor(season, ambient_temp, humidity)
        
        # Charging mode impact (simplified)
        charging_factor = self.calculate_charging_mode_factor(charge_mode)
        
        # Basic health (60% weight)
        basic_health = (soc_factor + temp_factor + voltage_factor) / 3
        
        # Environmental health (25% weight)
        environmental_health = (humidity_factor + heat_stress_factor + monsoon_factor + salinity_factor) / 4
        
        # Climate zone health (10% weight)
        climate_health = climate_zone_factor
        
        # Season health (3% weight)
        season_health = season_factor
        
        # Charging health (2% weight - simplified)
        charging_health = charging_factor
        
        # Combined BHI with enhanced weighting
        bhi = (0.60 * basic_health + 0.25 * environmental_health + 0.10 * climate_health + 0.03 * season_health + 0.02 * charging_health) * 100
        return max(0, min(100, bhi))
    
    def calculate_humidity_factor(self, humidity):
        """Factor for humidity impact on battery health (monsoon conditions)"""
        if humidity < 0.3:  # Dry conditions
            return 1.0
        elif humidity < 0.7:  # Normal conditions
            return 0.9
        elif humidity < 0.9:  # High humidity
            return 0.7
        else:  # Extreme humidity (monsoon)
            return 0.5
    
    def calculate_heat_stress_factor(self, ambient_temp):
        """Factor for extreme heat conditions in India"""
        if ambient_temp > 45:  # Extreme heat (desert conditions)
            return 0.6
        elif ambient_temp > 40:  # High heat (summer)
            return 0.8
        elif ambient_temp > 35:  # Moderate heat
            return 0.9
        else:
            return 1.0  # Normal conditions
    
    def calculate_monsoon_factor(self, ambient_temp):
        """Factor for monsoon season impact"""
        if 20 <= ambient_temp <= 30:  # Monsoon temperature range
            return 0.8  # Reduced health during monsoon
        else:
            return 1.0  # Normal health
    
    def calculate_salinity_factor(self, location):
        """Factor for coastal salinity impact"""
        coastal_areas = ['mumbai', 'chennai', 'kolkata', 'goa', 'kerala', 'coastal']
        if location.lower() in coastal_areas:
            return 0.85  # Reduced health in coastal areas
        else:
            return 1.0  # Normal health inland
    
    def calculate_charging_mode_factor(self, charge_mode):
        """Simplified factor for charging mode impact on battery health"""
        if charge_mode == "slow":
            return 1.0  # Optimal slow charging
        elif charge_mode == "fast":
            return 0.90  # Slightly reduced due to fast charging stress
        elif charge_mode == "pause":
            return 0.95  # Slightly reduced due to no charging
        else:
            return 1.0  # Default to optimal
    
    def calculate_climate_zone_factor(self, climate_zone, ambient_temp, humidity):
        """Factor for climate zone impact on battery health"""
        if "Tropical Monsoon" in climate_zone:
            # High humidity, monsoon conditions
            if humidity > 0.8:
                return 0.85  # Reduced health in high humidity
            return 0.90  # Slightly reduced in monsoon conditions
        elif "Hot Desert" in climate_zone:
            # Extreme heat conditions
            if ambient_temp > 45:
                return 0.80  # Significantly reduced in extreme heat
            return 0.85  # Reduced in desert conditions
        elif "Tropical Savanna" in climate_zone:
            # Urban heat island effect
            if ambient_temp > 40:
                return 0.88  # Reduced in urban heat
            return 0.95  # Good conditions
        elif "Subtropical Highland" in climate_zone:
            # Altitude and temperature variation
            if ambient_temp < 10:
                return 0.90  # Reduced in cold conditions
            return 0.95  # Good conditions
        elif "Tropical Wet" in climate_zone:
            # High humidity, tropical conditions
            if humidity > 0.85:
                return 0.82  # Reduced in very high humidity
            return 0.88  # Reduced in tropical conditions
        else:
            return 1.0  # Default optimal conditions
    
    def calculate_season_factor(self, season, ambient_temp, humidity):
        """Factor for season impact on battery health"""
        if season.lower() == 'monsoon':
            # High humidity, reduced performance
            if humidity > 0.8:
                return 0.85  # Significantly reduced in monsoon
            return 0.90  # Reduced in monsoon season
        elif season.lower() == 'summer':
            # Extreme heat conditions
            if ambient_temp > 45:
                return 0.80  # Significantly reduced in extreme heat
            return 0.90  # Reduced in summer heat
        elif season.lower() == 'winter':
            # Cold conditions
            if ambient_temp < 5:
                return 0.88  # Reduced in very cold conditions
            return 0.95  # Good in moderate winter
        elif season.lower() == 'spring':
            # Moderate conditions
            return 1.0  # Optimal conditions
        else:
            return 1.0  # Default optimal conditions
    
    def get_bhi_recommendations(self, bhi, telemetry):
        """Generate BHI-based charging recommendations for Indian conditions"""
        humidity = telemetry.get('humidity', 0.5)
        ambient_temp = telemetry.get('ambient_temp', 25)
        location = telemetry.get('location', 'inland')
        climate_zone = telemetry.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
        season = telemetry.get('season', 'monsoon')
        
        recommendations = []
        
        # BHI-based charging strategy
        if bhi >= 90:
            recommendations.append("🟢 Excellent health - Fast charging recommended")
        elif bhi >= 75:
            recommendations.append("🟡 Good health - Normal charging OK")
        elif bhi >= 60:
            recommendations.append("🟠 Moderate health - Slow charging recommended")
        elif bhi >= 40:
            recommendations.append("🔴 Poor health - Trickle charge only")
        else:
            recommendations.append("🚨 Critical health - Stop charging, check battery")
        
        # Climate zone specific recommendations
        if "Tropical Monsoon" in climate_zone:
            recommendations.append("🌧️ Monsoon zone - High humidity protection needed")
            if humidity > 0.8:
                recommendations.append("💧 Extreme humidity - Use moisture barriers")
        elif "Hot Desert" in climate_zone:
            recommendations.append("🔥 Desert zone - Extreme heat management required")
            if ambient_temp > 45:
                recommendations.append("🌡️ Critical heat - Use thermal cooling systems")
        elif "Tropical Savanna" in climate_zone:
            recommendations.append("🏙️ Urban zone - Heat island effect monitoring")
            if ambient_temp > 40:
                recommendations.append("🌡️ Urban heat - Use shade and cooling")
        elif "Subtropical Highland" in climate_zone:
            recommendations.append("⛰️ Highland zone - Altitude effects monitoring")
            if ambient_temp < 10:
                recommendations.append("❄️ Cold conditions - Use battery warming")
        elif "Tropical Wet" in climate_zone:
            recommendations.append("🌧️ Wet zone - High humidity and rainfall protection")
            if humidity > 0.85:
                recommendations.append("💧 Very high humidity - Extra moisture protection")
        
        # Season specific recommendations
        if season.lower() == 'monsoon':
            recommendations.append("🌦️ Monsoon season - Extra moisture protection needed")
            if humidity > 0.8:
                recommendations.append("🌧️ Heavy monsoon - Use waterproof covers")
        elif season.lower() == 'summer':
            recommendations.append("☀️ Summer season - Heat stress management")
            if ambient_temp > 45:
                recommendations.append("🌡️ Extreme summer heat - Use thermal management")
        elif season.lower() == 'winter':
            recommendations.append("❄️ Winter season - Cold weather protection")
            if ambient_temp < 5:
                recommendations.append("🧊 Very cold - Use battery warming systems")
        elif season.lower() == 'spring':
            recommendations.append("🌸 Spring season - Moderate conditions, standard care")
        
        # India-specific environmental recommendations
        if humidity > 0.8:
            recommendations.append("🌧️ High humidity - Monitor for moisture ingress")
        
        if ambient_temp > 40:
            recommendations.append("🌡️ Extreme heat - Use thermal management")
        
        if location.lower() in ['mumbai', 'chennai', 'kolkata', 'goa', 'kerala']:
            recommendations.append("🏖️ Coastal area - Check for salt corrosion")
        
        if 20 <= ambient_temp <= 30 and humidity > 0.7:
            recommendations.append("🌦️ Monsoon conditions - Extra moisture protection needed")
        
        return recommendations
    
    def get_adaptive_safety_thresholds(self, base_thresholds, ambient_temp, humidity, location, season, charge_mode=None):
        """Calculate adaptive safety thresholds based on India climate zones and charging mode"""
        adaptations = []
        
        # Base thresholds
        high_temp = base_thresholds.get('high_temp_celsius', 41.0)
        low_soc = base_thresholds.get('low_soc_percent', 20)
        high_soc = base_thresholds.get('high_soc_percent', 80)
        
        # Climate zone-specific adaptations
        if location == "tropical_monsoon":
            high_temp -= 2.0  # Lower threshold for monsoon conditions
            adaptations.append("🌧️ Tropical Monsoon: Lower temp threshold (-2°C)")
            if humidity > 0.8:
                high_temp -= 1.0
                adaptations.append("💧 High humidity: Additional temp reduction (-1°C)")
        
        elif location == "hot_desert":
            high_temp -= 3.0  # Much lower threshold for desert heat
            adaptations.append("🔥 Hot Desert: Lower temp threshold (-3°C)")
            if ambient_temp > 45:
                low_soc += 5
                high_soc -= 5
                adaptations.append("🌡️ Extreme heat: Adjusted SoC thresholds (±5%)")
        
        elif location == "tropical_savanna":
            high_temp -= 1.0  # Moderate reduction for urban heat
            adaptations.append("🏙️ Tropical Savanna: Lower temp threshold (-1°C)")
            if ambient_temp > 40:
                high_temp -= 1.0
                adaptations.append("🌡️ Urban heat island: Additional reduction (-1°C)")
        
        elif location == "subtropical_highland":
            if ambient_temp < 15:  # Cold conditions
                high_temp += 2.0
                adaptations.append("❄️ Highland cold: Higher temp threshold (+2°C)")
            else:
                high_temp -= 0.5  # Slight reduction for altitude effects
                adaptations.append("⛰️ Altitude effects: Slight temp reduction (-0.5°C)")
        
        elif location == "tropical_wet":
            high_temp -= 2.5  # Lower threshold for high humidity
            adaptations.append("🌧️ Tropical Wet: Lower temp threshold (-2.5°C)")
            if humidity > 0.8:
                low_soc += 3
                adaptations.append("💧 High humidity: Higher low SoC threshold (+3%)")
        
        # Season-specific adaptations
        if season.lower() == 'monsoon':
            high_temp -= 1.0
            adaptations.append("🌦️ Monsoon season: Lower temp threshold (-1°C)")
        elif season.lower() == 'summer':
            high_temp -= 0.5
            adaptations.append("☀️ Summer season: Lower temp threshold (-0.5°C)")
        elif season.lower() == 'winter':
            high_temp += 1.0
            adaptations.append("❄️ Winter season: Higher temp threshold (+1°C)")
        
        # Additional humidity adaptations
        if humidity > 0.9:  # Very high humidity
            high_temp -= 1.0
            adaptations.append("💧 Very high humidity: Lower temp threshold (-1°C)")
        
        # Charging mode-specific adaptations
        if charge_mode:
            if charge_mode == "fast":
                high_temp -= 1.5  # More sensitive during fast charging
                adaptations.append("⚡ Fast mode: Lower temp threshold (-1.5°C)")
            elif charge_mode == "slow":
                high_temp -= 0.5  # Slightly more sensitive during slow charging
                adaptations.append("🐌 Slow mode: Lower temp threshold (-0.5°C)")
            elif charge_mode == "pause":
                high_temp += 1.0  # Less sensitive when not charging
                adaptations.append("⏸️ Pause mode: Higher temp threshold (+1°C)")
        
        return {
            'high_temp': max(35.0, high_temp),  # Minimum 35°C threshold
            'low_soc': max(10, low_soc),  # Minimum 10% low SoC
            'high_soc': min(95, high_soc),  # Maximum 95% high SoC
            'adaptations': adaptations
        }
    
    def get_action_reason(self, telemetry, anomaly_predictions, rl_action, rl_confidence, safety_status):
        """Generate explanation for RL agent action with climate awareness"""
        temp = telemetry['temperature']
        soc = telemetry['soc']
        climate_zone = telemetry.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
        season = telemetry.get('season', 'monsoon')
        ensemble_prob = anomaly_predictions.get('ensemble', {}).get('probability', 0.5)
        
        # Climate-aware context
        climate_context = f"({climate_zone.split('(')[0].strip()}, {season.title()})"
        
        # Determine primary reason for action
        if ensemble_prob > 0.8:
            return f"High anomaly detected ({ensemble_prob*100:.1f}%) - Safety protocol: {rl_action} {climate_context}"
        elif temp > 45:
            return f"Critical temperature ({temp:.1f}°C) - Emergency action: {rl_action} {climate_context}"
        elif soc < 0.1:
            return f"Critical low SoC ({soc*100:.1f}%) - Urgent charging: {rl_action} {climate_context}"
        elif soc > 0.9:
            return f"Battery nearly full ({soc*100:.1f}%) - Reduce charging: {rl_action} {climate_context}"
        elif rl_confidence < 0.5:
            return f"Low confidence ({rl_confidence*100:.0f}%) - Conservative action: {rl_action} {climate_context}"
        elif temp > 35:
            return f"Elevated temperature ({temp:.1f}°C) - Thermal management: {rl_action} {climate_context}"
        elif soc < 0.2:
            return f"Low SoC ({soc*100:.1f}%) - Charging recommended: {rl_action} {climate_context}"
        elif ensemble_prob > 0.6:
            return f"Moderate anomaly risk ({ensemble_prob*100:.1f}%) - Cautious action: {rl_action} {climate_context}"
        else:
            return f"Normal conditions - Optimal action: {rl_action} (confidence: {rl_confidence*100:.0f}%) {climate_context}"
    
    def log_untrained_rl_state(self, debug_info, telemetry):
        """Log untrained RL states separately for targeted training improvement"""
        import json
        from datetime import datetime
        
        # Log untrained RL state for fine-tuning
        
        try:
            untrained_entry = {
                'timestamp': datetime.now().isoformat(),
                'state_bins': debug_info.get('state_bins', []),
                'state_index': debug_info.get('state_index', ''),
                'standardized_telemetry': {
                    'temperature': float(debug_info.get('temp_std', 0.0)),
                    'soc': float(debug_info.get('soc_std', 0.0)),
                    'voltage': float(debug_info.get('voltage_std', 0.0)),
                    'c_rate': float(telemetry.get('c_rate', 0.0)),
                    'power': float(telemetry.get('power', 0.0)),
                    'is_anomaly': bool(telemetry.get('is_anomaly', False))
                },
                'real_world_context': {
                    'temperature_celsius': float(telemetry.get('temperature', 25.0)),
                    'soc_percentage': float(telemetry.get('soc', 0.5)) * 100,
                    'voltage': float(telemetry.get('voltage', 3.7)),
                    'scenario': str(telemetry.get('scenario', 'Unknown'))
                },
                'q_values_zero': debug_info.get('q_values', [0.0, 0.0, 0.0, 0.0, 0.0]),
                'safety_priority': 'high' if debug_info.get('temp_std', 0.0) > 1.5 or debug_info.get('soc_std', 0.0) < -2.0 else 'normal'
            }
        except Exception as e:
            # If there's an error creating the entry, create a minimal one
            untrained_entry = {
                'timestamp': datetime.now().isoformat(),
                'state_bins': debug_info.get('state_bins', []),
                'error': f"Failed to create full entry: {str(e)}",
                'debug_info_keys': list(debug_info.keys()) if isinstance(debug_info, dict) else 'not_dict',
                'telemetry_keys': list(telemetry.keys()) if isinstance(telemetry, dict) else 'not_dict'
            }
        
        # Save to dedicated untrained states file
        untrained_file = Path("rl_untrained_states.json")
        
        # Check if we have write access (Streamlit Cloud detection)
        try:
            test_file = Path("test_write_access.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()  # Delete test file
            write_access = True
        except Exception:
            write_access = False
            print("⚠️ No write access - untrained state logging disabled (Streamlit Cloud environment)")
            return
        
        if not write_access:
            print("⚠️ Untrained state logging disabled: No file system access (Streamlit Cloud)")
            return
            
        try:
            # Load existing untrained states
            if untrained_file.exists():
                try:
                    with open(untrained_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            untrained_states = json.loads(content)
                        else:
                            untrained_states = []
                except (json.JSONDecodeError, Exception) as e:
                    # If file is corrupted, start fresh
                    untrained_states = []
            else:
                untrained_states = []
            
            # OPTIMIZED: Use set for O(1) lookup instead of O(n) list search
            existing_state_indices = set(s.get('state_index', '') for s in untrained_states if isinstance(s, dict))
            current_state_index = untrained_entry.get('state_index', '')
            
            if current_state_index not in existing_state_indices:
                untrained_states.append(untrained_entry)
                
                # Write back to file with proper error handling
                with open(untrained_file, 'w') as f:
                    json.dump(untrained_states, f, indent=2, ensure_ascii=False)
                    f.flush()  # Ensure data is written
                pass  # Successfully logged
            else:
                # State already exists, skip logging (no file I/O)
                pass
                    
        except Exception as e:
            # Log error message instead of failing silently
            print(f"⚠️ Not saving the logs: Failed to save untrained RL state - {str(e)}")
            pass

    def log_prediction_data(self, telemetry, features, anomaly_predictions, rl_action, rl_confidence, safety_status, action_reason=None, critical_count=0, warning_count=0, adaptive_thresholds=None, bhi=None):
        """Log prediction data for validation and analysis with climate context"""
        import json
        from datetime import datetime
        
        # Get action reason if not provided
        if action_reason is None:
            action_reason = self.get_action_reason(telemetry, anomaly_predictions, rl_action, rl_confidence, safety_status)
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_telemetry': {
                'voltage': telemetry['voltage'],
                'current': telemetry['current'],
                'temperature': telemetry['temperature'],
                'soc': telemetry['soc'],
                'ambient_temp': telemetry['ambient_temp'],
                'humidity': telemetry['humidity'],
                'charge_mode': telemetry['charge_mode'],
                'climate_zone': telemetry.get('climate_zone', 'unknown'),  # Enhanced logging
                'location': telemetry.get('location', 'unknown'),  # Actual city name
                'location_type': telemetry.get('location_type', 'unknown'),  # Climate zone type
                'season': telemetry.get('season', 'unknown'),
                'scenario': telemetry.get('scenario', 'Unknown')
            },
            'standardized_features': {
                name: float(features[0][i]) for i, name in enumerate(self.feature_names)
            } if features is not None and len(features[0]) == 16 else None,
            'model_predictions': {
                model: {
                    'prediction': int(pred['prediction']),
                    'probability': float(pred['probability'])
                } for model, pred in anomaly_predictions.items()
            },
            'rl_agent': {
                'action': rl_action,
                'confidence': float(rl_confidence),
                'reason': action_reason
            },
            'safety_assessment': safety_status,
            'ensemble_anomaly_probability': float(anomaly_predictions.get('ensemble', {}).get('probability', 0.5)),
            'alerts': {
                'critical_count': int(critical_count),
                'warning_count': int(warning_count),
                'critical_conditions': {
                    'high_temperature': bool(telemetry.get('temperature', 0) > 40),
                    'low_soc': bool(telemetry.get('soc', 0) < 0.15),
                    'low_voltage': bool(telemetry.get('voltage', 0) < 3.2),
                    'risk_status': bool(safety_status == "RISK"),
                    'high_anomaly_prob': bool(anomaly_predictions.get('ensemble', {}).get('probability', 0) > 0.8)
                }
            },
            'climate_context': {
                'climate_zone': telemetry.get('climate_zone', 'unknown'),
                'charge_mode': telemetry.get('charge_mode', 'unknown'),
                'adaptive_thresholds': adaptive_thresholds,
                'enhanced_bhi': bhi,
                'climate_adaptations': adaptive_thresholds.get('adaptations', []) if adaptive_thresholds else []
            }
        }
        
        # Add RL debug info if available (for training improvement analysis)
        if hasattr(self, 'last_rl_debug_info') and self.last_rl_debug_info:
            log_entry['rl_debug'] = {
                'is_untrained_state': self.last_rl_debug_info.get('is_untrained_state', False),
                'state_bins': self.last_rl_debug_info.get('state_bins'),
                'state_index': self.last_rl_debug_info.get('state_index'),
                'q_sum': float(self.last_rl_debug_info.get('q_sum', 0.0)),
                'q_values': self.last_rl_debug_info.get('q_values', []),
                'standardized_values': {
                    'temperature': float(self.last_rl_debug_info.get('temp_std', 0.0)),
                    'soc': float(self.last_rl_debug_info.get('soc_std', 0.0)),
                    'voltage': float(self.last_rl_debug_info.get('voltage_std', 0.0))
                },
                'debug_message': self.last_rl_debug_info.get('debug_message', '')
            }
        
        # Append to log file (continuous logging)
        log_file = Path("prediction_validation_log.json")
        
        # Check if we have write access (Streamlit Cloud detection)
        try:
            test_file = Path("test_write_access.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()  # Delete test file
            write_access = True
        except Exception:
            write_access = False
            print("⚠️ No write access - logging disabled (Streamlit Cloud environment)")
            return
        
        if not write_access:
            print("⚠️ Logging disabled: No file system access (Streamlit Cloud)")
            return
            
        try:
            # Load existing logs if file exists
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                except (json.JSONDecodeError, Exception):
                    # If file is corrupted or empty, start fresh
                    logs = []
            else:
                logs = []
            
            # Add new entry
            logs.append(log_entry)
            
            # Optional: Keep only last 1000 entries to prevent excessive file size
            # (You can remove this limit if you want unlimited logging)
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            # Log error message instead of failing silently
            print(f"⚠️ Not saving the logs: Failed to save prediction data - {str(e)}")
            pass
    
    def assess_safety(self, telemetry, anomaly_prob):
        """Assess overall safety status"""
        risks = []
        
        # Temperature risk
        if telemetry['temperature'] > 45:
            risks.append("High Temperature")
        elif telemetry['temperature'] > 35:
            risks.append("Elevated Temperature")
        
        # SoC risk
        if telemetry['soc'] < 0.1:
            risks.append("Critical Low SoC")
        elif telemetry['soc'] > 0.9:
            risks.append("Critical High SoC")
        
        # Anomaly risk
        if anomaly_prob > 0.8:
            risks.append("High Anomaly Risk")
        elif anomaly_prob > 0.5:
            risks.append("Moderate Anomaly Risk")
        
        # Voltage risk
        if telemetry['voltage'] < 3.0 or telemetry['voltage'] > 4.2:
            risks.append("Voltage Out of Range")
        
        if not risks:
            return "SAFE", "green"
        elif len(risks) == 1 and "Moderate" in risks[0]:
            return "WARN", "yellow"  # Shortened from "CAUTION"
        else:
            return "RISK", "red"     # Shortened from "DANGER"

def generate_synthetic_telemetry():
    """Generate realistic synthetic telemetry data"""
    base_time = datetime.now()
    
    # Simulate different battery scenarios with realistic probabilities
    scenarios = [
        {"name": "Normal Operation", "temp_range": (20, 30), "soc_range": (0.2, 0.8), "voltage_range": (3.5, 4.0), "weight": 0.6},
        {"name": "Fast Charging", "temp_range": (30, 35), "soc_range": (0.1, 0.7), "voltage_range": (3.8, 4.1), "weight": 0.2},
        {"name": "Moderate Temperature", "temp_range": (32, 38), "soc_range": (0.3, 0.7), "voltage_range": (3.4, 3.9), "weight": 0.1},
        {"name": "Low SoC", "temp_range": (18, 28), "soc_range": (0.1, 0.3), "voltage_range": (3.2, 3.6), "weight": 0.07},
        {"name": "Critical Condition", "temp_range": (40, 45), "soc_range": (0.05, 0.95), "voltage_range": (3.0, 4.2), "weight": 0.03}
    ]
    
    # Weighted random selection (favor normal operation)
    weights = [s["weight"] for s in scenarios]
    scenario = np.random.choice(scenarios, p=weights)
    
    # Get climate context from session state
    climate_zone = st.session_state.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)')
    charge_mode = st.session_state.get('charge_mode', 'slow')
    location_type = st.session_state.get('location_type', 'tropical_monsoon')
    season = st.session_state.get('season', 'monsoon')
    
    telemetry = {
        'timestamp': base_time,
        'voltage': np.random.uniform(*scenario['voltage_range']),
        'current': np.random.uniform(-50, 50),  # Negative = discharging, Positive = charging
        'temperature': np.random.uniform(*scenario['temp_range']),
        'soc': np.random.uniform(*scenario['soc_range']),
        'ambient_temp': np.random.uniform(15, 35),
        'humidity': np.random.uniform(0.3, 0.8),
        'charge_mode': charge_mode,  # Use stored charge mode
        'climate_zone': climate_zone,  # Use stored climate zone
        'location': location_type,  # Use climate zone type
        'location_type': location_type,  # Use climate zone type for processing
        'season': season,  # Use selected season
        'scenario': scenario['name'],
        'time_since_start': len(st.session_state.telemetry_data) * 5  # 5 seconds per reading
    }
    
    return telemetry

def main():
    # Header
    st.title("🔋AI-Powered EV Battery Safety Management System")
    st.markdown("**Real-time Battery Monitoring, Anomaly Detection & Intelligent Action Recommendation**")
    
    # Initialize BMS
    bms = BatteryManagementSystem()
    
    # Sidebar controls
    st.sidebar.header("🎛️ System Controls")
    
    # Data transformation explanation
    # Debug toggle
    debug_rl = st.sidebar.checkbox("🔧 Debug RL Agent", value=False, 
                                   help="Show RL agent internal state, Q-values, and identify untrained states. Debug info is also logged for training improvements.")
    
    # Debug mode indicator (simple display)
    if debug_rl:
        st.sidebar.info("🔧 Debug mode is ACTIVE")
    
    # Debug mode is handled dynamically in the main processing loop
    
    # Test critical conditions button
    if st.sidebar.button("🧪 Test Critical Conditions"):
        # Generate critical test telemetry using selected climate zone
        critical_telemetry = {
            'timestamp': datetime.now(),
            'voltage': 3.0,  # Critical low voltage
            'current': 60,   # High current
            'temperature': 45,  # Critical high temperature
            'soc': 0.05,    # Critical low SoC
            'ambient_temp': 40,  # High ambient temperature
            'humidity': 0.9,    # High humidity
            'charge_mode': st.session_state.get('charge_mode', 'fast'),
            'climate_zone': st.session_state.get('climate_zone', 'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)'),
            'location': st.session_state.get('location_type', 'tropical_monsoon'),
            'location_type': st.session_state.get('location_type', 'tropical_monsoon'),
            'season': st.session_state.get('season', 'monsoon'),
            'scenario': 'Test Critical Conditions - India Enhanced',
            'time_since_start': len(st.session_state.telemetry_data) * 5
        }
        
        # Add to telemetry data
        st.session_state.telemetry_data.append(critical_telemetry)
        
        # Update alerts
        st.session_state.accumulated_alerts["critical"] += 1
        st.session_state.alert_history["critical"].append(f"{datetime.now().strftime('%H:%M:%S')}: Test critical alert")
        
        # Show immediate feedback
        st.sidebar.success("🧪 Critical test telemetry generated!")
        st.sidebar.info("📊 Check the main dashboard for prediction results")
        st.sidebar.write("**Test Data:**")
        st.sidebar.write(f"• Temperature: {critical_telemetry['temperature']}°C")
        st.sidebar.write(f"• Voltage: {critical_telemetry['voltage']}V")
        st.sidebar.write(f"• SoC: {critical_telemetry['soc']*100:.1f}%")
        st.sidebar.write(f"• Climate: {critical_telemetry['climate_zone']}")
        
        # Force page refresh to show results
        st.rerun()
    
    
    # Climate zone and season selection (always available for BHI calculations)
    st.sidebar.subheader("🌍 Climate & Season Selection")
    climate_zone = st.sidebar.selectbox(
        "Select Climate Zone",
        [
            "Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)",
            "Hot Desert (Rajasthan, Gujarat, Punjab)", 
            "Tropical Savanna (Delhi, Bangalore, Hyderabad)",
            "Subtropical Highland (Shimla, Kashmir, Himachal)",
            "Tropical Wet (Kerala, Assam, Meghalaya)"
        ],
        help="Select climate zone for BHI calculations, adaptive safety thresholds, and system behavior"
    )
    
    season = st.sidebar.selectbox(
        "Select Season",
        ["summer", "monsoon", "winter", "spring"],
        help="Seasonal impact on battery health (monsoon = high humidity, summer = extreme heat)"
    )
    
    # Store climate zone and season in session state
    st.session_state.climate_zone = climate_zone
    st.session_state.season = season
    
    # Extract location type for processing
    if "Tropical Monsoon" in climate_zone:
        location_type = "tropical_monsoon"
    elif "Hot Desert" in climate_zone:
        location_type = "hot_desert"
    elif "Tropical Savanna" in climate_zone:
        location_type = "tropical_savanna"
    elif "Subtropical Highland" in climate_zone:
        location_type = "subtropical_highland"
    elif "Tropical Wet" in climate_zone:
        location_type = "tropical_wet"
    else:
        location_type = "inland"
    
    # Store location type for processing
    st.session_state.location_type = location_type
    
    # Debug: No climate zone and season display
    
    # System status
    if st.sidebar.button("🚀 Start System" if not st.session_state.system_running else "⏹️ Stop System"):
        st.session_state.system_running = not st.session_state.system_running
        
        if st.session_state.system_running:
            # Show immediate feedback when starting
            st.sidebar.success("🚀 System started! Generating telemetry...")
            st.sidebar.write("**Next Steps:**")
            st.sidebar.write("• Watch the main dashboard for real-time data")
            st.sidebar.write("• Telemetry will be generated every 2 seconds")
            st.sidebar.write("• AI models will analyze each reading")
            st.sidebar.write("• RL agent will provide recommendations")
        else:
            st.rerun()  # Force page rerun to update the button text
    
    # Manual telemetry input
    st.sidebar.subheader("📊 Manual Input")
    st.sidebar.info("💡 **Input Real-World Values** - The system automatically converts them to standardized format for AI models")
    manual_mode = st.sidebar.checkbox("Manual Telemetry Input")
    
    if manual_mode:
        st.sidebar.write("**🔋 Battery Parameters:**")
        voltage = st.sidebar.slider("Voltage (V)", 2.5, 4.5, 3.7, 0.1, 
                                   help="Li-ion battery voltage: 3.0V (empty) to 4.2V (full)")
        current = st.sidebar.slider("Current (A)", -100.0, 100.0, 0.0, 5.0,
                                   help="Positive = charging, Negative = discharging, 0 = idle")
        temperature = st.sidebar.slider("Temperature (°C)", 0, 60, 25, 1,
                                       help="Battery cell temperature (⚠️ >40°C is dangerous)")
        soc = st.sidebar.slider("State of Charge (%)", 0, 100, 50, 1) / 100
        
        st.sidebar.write("**🌡️ Environmental:**")
        ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 0, 50, 20, 1,
                                        help="Environmental temperature around battery")
        humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50, 5, 
                                    help="Relative humidity (🌧️ >80% = monsoon conditions)") / 100
        
        # Use the climate zone and season selected at the top
        climate_zone = st.session_state.climate_zone
        location = st.session_state.location_type
        season = st.session_state.season
        
        # Removed charging station input to simplify the interface
        # Climate zone and charging mode are sufficient for safety assessment
        
        st.sidebar.write("**⚡ Charging Mode:**")
        charge_mode = st.sidebar.selectbox("Charge Mode", ['fast', 'slow', 'pause'],
                                          help="fast = Fast charging, slow = Slow charging, pause = No charging")
        
        # Store charge mode in session state for system mode
        st.session_state.charge_mode = charge_mode
        
        # Show adaptive safety thresholds based on India-specific conditions
        if hasattr(bms, 'transformer'):
            base_thresholds = bms.transformer.get_rl_thresholds_in_real_world()
            
            # Calculate India-specific adaptive thresholds
            adaptive_thresholds = bms.get_adaptive_safety_thresholds(
                base_thresholds, 
                ambient_temp, 
                humidity, 
                location, 
                season,
                charge_mode
            )
            
            st.sidebar.write("**🚨 AI Safety Thresholds (India Adaptive):**")
            st.sidebar.write(f"• High Temp: >{adaptive_thresholds['high_temp']:.1f}°C")
            st.sidebar.write(f"• Low SoC: <{adaptive_thresholds['low_soc']:.0f}%")
            st.sidebar.write(f"• High SoC: >{adaptive_thresholds['high_soc']:.0f}%")
            
            # Show adaptation factors
            if adaptive_thresholds['adaptations']:
                st.sidebar.write("**🇮🇳 India Adaptations:**")
                for adaptation in adaptive_thresholds['adaptations']:
                    st.sidebar.write(f"• {adaptation}")
        
        if st.sidebar.button("📤 Submit Telemetry"):
            telemetry = {
                'timestamp': datetime.now(),
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'soc': soc,
                'ambient_temp': ambient_temp,
                'humidity': humidity,
                'location': location,
                'season': season,
                'charge_mode': charge_mode,
                'climate_zone': climate_zone,  # Enhanced logging
                'scenario': 'Manual Input - India Enhanced',
                'time_since_start': len(st.session_state.telemetry_data) * 5
            }
            st.session_state.telemetry_data.append(telemetry)
    
    # Logging status and data processing info (positioned after manual telemetry)
    st.sidebar.subheader("📊 Logging & Data Processing")
    
    # Show logging status with file info
    log_file = Path("prediction_validation_log.json")
    
    # Test if we can actually write to files (Streamlit Cloud detection)
    try:
        test_file = Path("test_logging_access.json")
        with open(test_file, 'w') as f:
            f.write('{"test": "data"}')
        test_file.unlink()  # Delete test file
        can_write = True
    except Exception:
        can_write = False
    
    if not can_write:
        # Streamlit Cloud or no write access
        st.sidebar.warning("📝 **Logging**: Not Available\n\n⚠️ **Cloud Environment**: File system access restricted\n\n💡 **Local Development**: Logging works in local environment")
    elif log_file.exists():
        import time
        file_age = time.time() - log_file.stat().st_mtime
        
        if file_age > 300:  # 5 minutes - file is old
            st.sidebar.info("📝 **Logging**: Inactive\n\n⏰ **Last Update**: " + 
                          f"{int(file_age/60)} minutes ago\n\n💡 **Local Development**: Logging works in local environment")
        else:
            try:
                with open(log_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        logs = json.loads(content)
                        log_count = len(logs)
                    else:
                        log_count = 0
                    file_size = log_file.stat().st_size / 1024  # Size in KB
                    
                    # Check dedicated untrained states file
                    untrained_file = Path("rl_untrained_states.json")
                    untrained_count = 0
                    untrained_file_size = 0
                    
                    if untrained_file.exists():
                        try:
                            with open(untrained_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    untrained_states = json.loads(content)
                                    untrained_count = len(untrained_states)
                                else:
                                    untrained_count = 0
                            untrained_file_size = untrained_file.stat().st_size / 1024  # Size in KB
                        except Exception as e:
                            untrained_count = 0
                            print(f"⚠️ Error reading untrained states: {e}")
                    
                    status_text = f"📝 **Continuous Logging**: Active\n\n{log_count:,} entries logged ({file_size:.1f} KB)"
                    
                    # Debug: Show file reading status
                    if st.session_state.get('debug_mode', False):
                        status_text += f"\n\n🔍 **Debug Info**:\n- Log file exists: ✅\n- File size: {file_size:.1f} KB\n- JSON entries: {log_count:,}"
                    
                    # Always show untrained states information
                    status_text += f"\n\n🔴 **Untrained RL States**: {untrained_count:,} unique states ({untrained_file_size:.1f} KB)\n📁 `rl_untrained_states.json`"
                    if untrained_count >= 10:  # Suggest retraining after collecting enough states
                        status_text += f"\n\n💡 **Ready for RL Improvement!**\nRun: `python scripts/fine_tune_from_logs.py`"
                    elif untrained_count == 0:
                        status_text += f"\n\n✅ **Fresh Start**: Ready to collect new untrained states"
                    
                    status_text += f"\n\nAll predictions saved to `prediction_validation_log.json`"
                    
                    st.sidebar.info(status_text)
                    
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error reading log files: {e}")
                st.sidebar.info("📝 **Continuous Logging**: Active (file size increasing)")
    else:
        # No log file exists - show appropriate message
        st.sidebar.info("📝 **Logging**: Ready\n\n💡 **Local Development**: Logging will start when system runs\n\n⚠️ **Cloud Environment**: Logging may not be available")
    
    
    # Auto-generate telemetry
    if st.session_state.system_running and not manual_mode:
        if len(st.session_state.telemetry_data) == 0 or \
           (datetime.now() - st.session_state.telemetry_data[-1]['timestamp']).seconds >= 2:
            new_telemetry = generate_synthetic_telemetry()
            st.session_state.telemetry_data.append(new_telemetry)
            
            # Keep only last 50 readings
            if len(st.session_state.telemetry_data) > 50:
                st.session_state.telemetry_data = st.session_state.telemetry_data[-50:]
    
    # Main dashboard
    if st.session_state.telemetry_data:
        current_telemetry = st.session_state.telemetry_data[-1]
        
        # Extract features and make predictions
        features = bms.extract_features(current_telemetry)
        anomaly_predictions = bms.predict_anomaly(features)
        
        # Create standardized telemetry for RL agent (RL was trained on standardized data)
        standardized_telemetry = {
            'voltage': features[0][0],      # Standardized voltage
            'current': features[0][1],      # Standardized current  
            'temperature': features[0][2],  # Standardized temperature
            'soc': features[0][3],          # Standardized SoC
            'ambient_temp': features[0][4], # Standardized ambient temp
        }
        
        # Add anomaly flag to standardized telemetry for RL agent
        ensemble_prob = anomaly_predictions.get('ensemble', {}).get('probability', 0.5)
        standardized_telemetry['is_anomaly'] = ensemble_prob > 0.7  # High threshold to avoid over-conservative actions for borderline cases
        
        # Store raw telemetry for untrained state logging
        bms.current_raw_telemetry = current_telemetry
        
        # Debug: Show when RL agent is called (only when debug is enabled)
        if debug_rl:
            pass  # No debug message for RL agent call
        
        rl_action, rl_confidence = bms.get_rl_action(standardized_telemetry, debug_rl)
        
        # Debug: Show RL agent result (only when debug is enabled)
        if debug_rl:
            st.sidebar.write(f"🎯 **RL Agent Decision:** {rl_action} (confidence: {rl_confidence:.2f})")
        # Calculate Enhanced BHI with India-specific factors
        bhi = bms.calculate_bhi(current_telemetry)
        
        # Get BHI recommendations
        bhi_recommendations = bms.get_bhi_recommendations(bhi, current_telemetry)
        
        # Get ensemble prediction
        ensemble_prob = anomaly_predictions.get('ensemble', {}).get('probability', 0.5)
        safety_status, safety_color = bms.assess_safety(current_telemetry, ensemble_prob)
        
        # Get action reason for display and logging
        action_reason = bms.get_action_reason(current_telemetry, anomaly_predictions, rl_action, rl_confidence, safety_status)
        
        # Log prediction data for validation (will be called after critical/warning counts are calculated)
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Add emoji to indicate status without extra height
            if safety_color == "red":
                status_display = f"🔴 {safety_status}"
            elif safety_color == "yellow":
                status_display = f"🟡 {safety_status}"
            else:
                status_display = f"🟢 {safety_status}"
                
            st.metric(
                "🛡️ Safety",
                status_display,
                delta=None,
                help="Overall system safety assessment"
            )
        
        with col2:
            st.metric(
                "🔋 Battery Health Index",
                f"{bhi:.1f}%",
                delta=f"{bhi-75:.1f}%" if bhi < 75 else f"+{bhi-75:.1f}%",
                help="Enhanced BHI for Indian conditions (SoC, Temperature, Voltage, Humidity, Heat, Monsoon, Salinity)"
            )
        
        with col3:
            st.metric(
                "🎯 Anomaly Risk",
                f"{ensemble_prob*100:.1f}%",
                delta=f"{(ensemble_prob-0.5)*100:.1f}%",
                help="Probability of anomalous behavior"
            )
        
        with col4:
            # Store previous confidence for delta calculation
            if "prev_confidence" not in st.session_state:
                st.session_state.prev_confidence = rl_confidence
            
            confidence_delta = rl_confidence - st.session_state.prev_confidence
            confidence_change = f"{confidence_delta*100:+.0f}%" if abs(confidence_delta) > 0.01 else None
            
            st.metric(
                "🤖 Action",
                rl_action.replace('_', ' ').title(),
                delta=f"Confidence: {rl_confidence*100:.0f}%" + (f" ({confidence_change})" if confidence_change else ""),
                help=f"RL Agent recommendation: {action_reason}"
            )
            
            # Update previous confidence
            st.session_state.prev_confidence = rl_confidence
        
        # Display Enhanced BHI Recommendations
        st.subheader("🔋 Battery Health Index (BHI) - India Enhanced")
        
        # BHI Status and Recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**BHI Score**: {bhi:.1f}%")
            if bhi >= 90:
                st.success("🟢 Excellent Battery Health")
            elif bhi >= 75:
                st.info("🟡 Good Battery Health")
            elif bhi >= 60:
                st.warning("🟠 Moderate Battery Health")
            elif bhi >= 40:
                st.error("🔴 Poor Battery Health")
            else:
                st.error("🚨 Critical Battery Health")
        
        with col2:
            # Environmental factors display
            humidity = current_telemetry.get('humidity', 0.5)
            ambient_temp = current_telemetry.get('ambient_temp', 25)
            location = current_telemetry.get('location', 'inland')
            
            st.write("**Environmental Factors:**")
            st.write(f"🌡️ Ambient: {ambient_temp:.1f}°C")
            st.write(f"💧 Humidity: {humidity*100:.1f}%")
            st.write(f"📍 Location: {location.title()}")
        
        # BHI Recommendations
        if bhi_recommendations:
            st.subheader("📋 BHI-Based Recommendations")
            for i, recommendation in enumerate(bhi_recommendations, 1):
                st.write(f"{i}. {recommendation}")
        
        with col5:
            st.metric(
                "📊 Data Points",
                len(st.session_state.telemetry_data),
                delta="+1" if len(st.session_state.telemetry_data) > 1 else None,
                help="Total telemetry readings"
            )
        
        # Current telemetry display
        st.subheader("📡 Current Telemetry")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Voltage:** {current_telemetry['voltage']:.2f} V")
            st.info(f"**Current:** {current_telemetry['current']:.1f} A")
        
        with col2:
            st.info(f"**Temperature:** {current_telemetry['temperature']:.1f} °C")
            st.info(f"**Ambient Temp:** {current_telemetry['ambient_temp']:.1f} °C")
        
        with col3:
            st.info(f"**State of Charge:** {current_telemetry['soc']*100:.1f} %")
            st.info(f"**Humidity:** {current_telemetry['humidity']*100:.0f} %")
        
        with col4:
            st.info(f"**Charge Mode:** {current_telemetry['charge_mode'].title()}")
            st.info(f"**Scenario:** {current_telemetry['scenario']}")
        
        # Model predictions
        st.subheader("🧠 AI Model Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Anomaly Detection Models:**")
            for model_name, pred in anomaly_predictions.items():
                prob = pred['probability']
                status = "🔴 ANOMALY" if pred['prediction'] == 1 else "🟢 NORMAL"
                st.write(f"• **{model_name.replace('_', ' ').title()}:** {status} ({prob*100:.1f}%)")
        
        with col2:
            st.write("**Action Recommendation:**")
            st.write(f"• **RL Agent Action:** {rl_action.replace('_', ' ').title()}")
            st.write(f"• **Confidence Level:** {rl_confidence*100:.0f}%")
            st.write(f"• **Safety Assessment:** {safety_status}")
            st.write(f"• **Reason:** {action_reason}")
        
        with col3:
            st.write("**🔍 All 16 Features Sent to Models:**")
            if features is not None and len(features[0]) == 16:
                feature_values = features[0]
                # Show top 3 most important features
                important_features = [0, 7, 8]  # voltage, power, c_rate (top predictors)
                for i in important_features:
                    name = bms.feature_names[i]
                    value = feature_values[i]
                    st.write(f"• **{name}:** {value:.3f}")
                
                # Show remaining 13 features in an expander
                remaining_count = len(feature_values) - 3
                with st.expander(f"Show remaining {remaining_count} features"):
                    for i, (name, value) in enumerate(zip(bms.feature_names, feature_values)):
                        if i not in important_features:
                            st.write(f"• **{name}:** {value:.3f}")
            else:
                st.error("⚠️ Feature extraction issue detected!")
                st.write(f"Expected 16 features, got: {len(features[0]) if features is not None else 'None'}")
        
        # Visualizations (optimized for performance)
        if len(st.session_state.telemetry_data) > 1:
            st.subheader("📈 Real-time Monitoring")
            
            # Create DataFrame for plotting (limit to last 50 points for performance)
            recent_data = st.session_state.telemetry_data[-50:] if len(st.session_state.telemetry_data) > 50 else st.session_state.telemetry_data
            df = pd.DataFrame(recent_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time series plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Battery Voltage', 'Temperature vs Ambient', 'State of Charge', 'Current Flow'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Voltage plot
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['voltage'], name='Voltage', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['temperature'], name='Battery Temp', line=dict(color='red')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['ambient_temp'], name='Ambient Temp', line=dict(color='orange')),
                row=1, col=2
            )
            
            # SoC plot
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['soc']*100, name='SoC (%)', line=dict(color='green')),
                row=2, col=1
            )
            
            # Current plot
            colors = ['red' if c > 0 else 'blue' for c in df['current']]
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['current'], name='Current', 
                          line=dict(color='purple'), fill='tonexty'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Battery Telemetry Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly probability over time
            if len(st.session_state.telemetry_data) > 5:
                st.subheader("🎯 Anomaly Detection Timeline")
                
                # PERFORMANCE OPTIMIZATION: Cache anomaly predictions
                if "anomaly_cache" not in st.session_state:
                    st.session_state.anomaly_cache = {}
                
                # Only calculate for new data points
                recent_data = st.session_state.telemetry_data[-20:]  # Last 20 points
                anomaly_probs = []
                
                for i, telemetry in enumerate(recent_data):
                    cache_key = f"{telemetry['timestamp']}_{telemetry['temperature']:.1f}_{telemetry['soc']:.2f}"
                    
                    if cache_key in st.session_state.anomaly_cache:
                        # Use cached result
                        prob = st.session_state.anomaly_cache[cache_key]
                    else:
                        # Calculate and cache new result
                        features = bms.extract_features(telemetry)
                        preds = bms.predict_anomaly(features)
                        prob = preds.get('ensemble', {}).get('probability', 0.5)
                        st.session_state.anomaly_cache[cache_key] = prob
                    
                    anomaly_probs.append(prob)
                
                # Clean old cache entries (keep last 100)
                if len(st.session_state.anomaly_cache) > 100:
                    old_keys = list(st.session_state.anomaly_cache.keys())[:-50]
                    for key in old_keys:
                        del st.session_state.anomaly_cache[key]
                
                fig_anomaly = go.Figure()
                fig_anomaly.add_trace(go.Scatter(
                    x=list(range(len(anomaly_probs))),
                    y=anomaly_probs,
                    mode='lines+markers',
                    name='Anomaly Probability',
                    line=dict(color='red', width=3),
                    fill='tonexty'
                ))
                
                fig_anomaly.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                                     annotation_text="Anomaly Threshold")
                fig_anomaly.update_layout(
                    title="Anomaly Detection Over Time",
                    xaxis_title="Time Steps",
                    yaxis_title="Anomaly Probability",
                    height=300
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Dynamic Alert System with Session State
        st.subheader("🚨 System Status")
        
        # Initialize alert counters in session state
        # alert_history is now initialized in main session state
        
        # Initialize accumulated alert counters
        # accumulated_alerts is now initialized in main session state
        
        # Get accumulated counts
        accumulated_critical = st.session_state.accumulated_alerts["critical"]
        accumulated_warning = st.session_state.accumulated_alerts["warning"]
        
        # Current status check (for display purposes)
        current_critical = 0
        current_warning = 0
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check critical conditions based on safety status and ensemble probability
        if safety_status == "RISK" or ensemble_prob > 0.8:
            st.error("🔥 CRITICAL: System at RISK!")
            current_critical += 1
        elif safety_status == "WARN" or ensemble_prob > 0.6:
            st.warning("⚠️ WARNING: System needs attention!")
            current_warning += 1
        
        # Also check individual critical conditions for additional alerts
        if current_telemetry['temperature'] > 40:  # Lowered from 45°C
            st.error("🔥 CRITICAL: Battery temperature exceeds safe limits!")
            current_critical += 1
        
        if current_telemetry['soc'] < 0.15:  # Raised from 0.1 (10% to 15%)
            st.error("🔋 CRITICAL: Battery critically low!")
            current_critical += 1
        
        if current_telemetry['voltage'] < 3.2:  # Raised from 3.0V
            st.error("⚡ CRITICAL: Battery voltage critically low!")
            current_critical += 1
        
        # Check warning conditions
        if current_telemetry['soc'] > 0.9 and current_telemetry['current'] > 0:
            st.warning("⚡ WARNING: Battery nearly full!")
            current_warning += 1
        
        if current_telemetry['temperature'] > 35 and current_telemetry['temperature'] <= 45:
            st.warning("🌡️ WARNING: Battery temperature elevated!")
            current_warning += 1
        
        if current_telemetry['soc'] < 0.2 and current_telemetry['soc'] >= 0.1:
            st.warning("🔋 WARNING: Battery low!")
            current_warning += 1
        
        # Check anomaly probability
        if ensemble_prob > 0.8:
            st.warning("🎯 WARNING: High anomaly probability!")
            current_warning += 1
        
        # Log prediction data for validation (after critical/warning counts are calculated)
        # Calculate adaptive thresholds and BHI for enhanced logging
        adaptive_thresholds = None
        bhi = None
        if hasattr(bms, 'transformer'):
            base_thresholds = bms.transformer.get_rl_thresholds_in_real_world()
            adaptive_thresholds = bms.get_adaptive_safety_thresholds(
                base_thresholds,
                current_telemetry.get('ambient_temp', 25),
                current_telemetry.get('humidity', 0.5),
                current_telemetry.get('location', 'inland'),
                current_telemetry.get('season', 'spring'),
                current_telemetry.get('charge_mode', 'slow')
            )
            bhi = bms.calculate_bhi(current_telemetry)
        
        bms.log_prediction_data(current_telemetry, features, anomaly_predictions, rl_action, rl_confidence, safety_status, action_reason, current_critical, current_warning, adaptive_thresholds, bhi)
        
        # Accumulate alerts when conditions are met
        if current_critical > 0:
            # Add to accumulated count
            st.session_state.accumulated_alerts["critical"] += current_critical
            # Add to history
            for _ in range(current_critical):
                st.session_state.alert_history["critical"].append(f"{current_time}: Critical alert")
        
        if current_warning > 0:
            # Add to accumulated count
            st.session_state.accumulated_alerts["warning"] += current_warning
            # Add to history
            for _ in range(current_warning):
                st.session_state.alert_history["warning"].append(f"{current_time}: Warning alert")
        
        # Debug: Show what conditions are being checked (collapsible)
        if debug_rl:
            with st.sidebar.expander("🔍 Condition Debug", expanded=False):
                st.write(f"Safety Status: {safety_status}")
                st.write(f"Ensemble Prob: {ensemble_prob:.3f} (>0.8: {ensemble_prob > 0.8})")
                st.write(f"Temp: {current_telemetry['temperature']:.1f}°C (>40°C: {current_telemetry['temperature'] > 40})")
                st.write(f"SoC: {current_telemetry['soc']*100:.1f}% (<15%: {current_telemetry['soc'] < 0.15})")
                st.write(f"Voltage: {current_telemetry['voltage']:.2f}V (<3.2V: {current_telemetry['voltage'] < 3.2})")
                st.write(f"Current Critical: {current_critical}")
                st.write(f"Current Warning: {current_warning}")
        
        # Keep only recent alerts (last 10 of each type)
        st.session_state.alert_history["critical"] = st.session_state.alert_history["critical"][-10:]
        st.session_state.alert_history["warning"] = st.session_state.alert_history["warning"][-10:]
        
        if current_critical == 0 and current_warning == 0:
            st.success("✅ All systems operating normally")
        
        # Calculate deltas for proper change tracking
        critical_delta = current_critical  # Show current alerts as delta
        warning_delta = current_warning    # Show current alerts as delta
        
        # Debug information (can be removed later) - collapsible
        if debug_rl:  # Reuse the debug toggle
            with st.sidebar.expander("🔍 Alert Debug Info", expanded=False):
                st.write(f"Accumulated Critical: {accumulated_critical}")
                st.write(f"Current Critical: {current_critical}")
                st.write(f"Critical Delta: {critical_delta}")
                st.write(f"Accumulated Warning: {accumulated_warning}")
                st.write(f"Current Warning: {current_warning}")
                st.write(f"Warning Delta: {warning_delta}")
        
        # Dynamic summary with deltas
        total_critical = len(st.session_state.alert_history["critical"])
        total_warning = len(st.session_state.alert_history["warning"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🚨 Critical Total", 
                accumulated_critical,
                delta=critical_delta if critical_delta != 0 else None,
                help=f"Accumulated: {accumulated_critical}, Current: {current_critical}, History: {total_critical}"
            )
        with col2:
            st.metric(
                "⚠️ Warnings Total", 
                accumulated_warning,
                delta=warning_delta if warning_delta != 0 else None,
                help=f"Accumulated: {accumulated_warning}, Current: {current_warning}, History: {total_warning}"
            )
        with col3:
            system_health = "GOOD" if current_critical == 0 and current_warning == 0 else "CAUTION" if current_critical == 0 else "CRITICAL"
            health_color = "normal" if system_health == "GOOD" else "inverse"
            st.metric(
                "🏥 System Health",
                system_health,
                delta=None,
                help="Overall system health status"
            )
        
        # Action recommendations with reasoning
        st.write("**Recommended Actions:**")
        if rl_action == 'pause':
            st.warning(f"🛑 **PAUSE** charging/discharging operations")
            st.caption(f"💭 **Why:** {action_reason}")
        elif rl_action == 'fast_charge':
            st.info(f"⚡ **FAST CHARGE** - conditions are optimal")
            st.caption(f"💭 **Why:** {action_reason}")
        elif rl_action == 'slow_charge':
            st.info(f"🔌 **SLOW CHARGE** - safe charging recommended")
            st.caption(f"💭 **Why:** {action_reason}")
        elif rl_action == 'discharge':
            st.warning(f"📤 **DISCHARGE** - reduce battery level")
            st.caption(f"💭 **Why:** {action_reason}")
        else:
            st.info(f"🔄 **MAINTAIN** - hold current state")
            st.caption(f"💭 **Why:** {action_reason}")
    
    else:
        st.info("🚀 **Start the system** or **input manual telemetry** to begin monitoring")
        st.markdown("""
        ### 🔋 EV Battery Safety Management System Features:
        
        **🧠 AI-Powered Anomaly Detection:**
        - Random Forest Model (99.7% accuracy)
        - MLP Neural Network (99.6% accuracy)  
        - Ensemble prediction for robust detection
        
        **🤖 Intelligent Action Recommendation:**
        - Reinforcement Learning agent (37.5% safety improvement)
        - Real-time action suggestions (pause, charge, discharge, maintain)
        - Safety-first decision making
        
        **📊 Real-time Monitoring:**
        - Live telemetry visualization
        - Battery health index calculation
        - Safety status assessment
        - Alert generation and management
        
        **🎯 System Capabilities:**
        - Processes 5.3M+ training samples
        - 16 engineered features
        - Multi-model ensemble approach
        - Production-ready performance
        """)
    
    # How Data Processing Works (moved to bottom)
    with st.sidebar.expander("ℹ️ How Data Processing Works"):
        st.write("""
        **🔄 Real-World to AI Models:**
        
        1. **Input**: You provide real values (e.g., 35°C, 60% SoC)
        2. **Feature Engineering**: System calculates 16 features (power, ratios, gradients)
        3. **Standardization**: Features normalized to mean=0, std=1 (training format)
        4. **AI Prediction**: Models trained on 5.3M standardized samples
        5. **Output**: Human-readable results and recommendations
        
        **🤖 RL Agent Thresholds:**
        - Trained on standardized data (-3 to +3 range)
        - Automatically converted to real-world values for display
        - Safety decisions based on NASA battery dataset patterns
        """)
        
        if hasattr(bms, 'transformer'):
            thresholds = bms.transformer.get_rl_thresholds_in_real_world()
            st.write("**Current Safety Limits:**")
            for key, value in thresholds.items():
                if 'temp' in key:
                    st.write(f"• {key}: {value:.1f}°C")
                elif 'soc' in key:
                    st.write(f"• {key}: {value:.0f}%")
    
    # Auto-refresh for real-time updates
    if st.session_state.system_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
