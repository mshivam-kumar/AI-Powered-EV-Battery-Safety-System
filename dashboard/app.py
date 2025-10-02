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

class BatteryManagementSystem:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
        
        # RL Agent actions
        self.ACTIONS = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
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
        """Load all trained models and data transformer"""
        # Always initialize transformer first
        self.transformer = TelemetryTransformer(models_dir=str(self.models_dir))
        
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
            
            # Load RL Agent (action selector) - use our breakthrough model
            rl_path = self.models_dir / "rl_robust_enhanced_v2_q_table.pkl"
            if not rl_path.exists():
                rl_path = self.models_dir / "rl_safety_focused_q_table.pkl"
            if rl_path.exists():
                try:
                    with open(rl_path, 'rb') as f:
                        self.models['rl_agent'] = pickle.load(f)
                    st.success("✅ RL Agent loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not load RL Agent: {e}")
            
            if not self.models:
                st.warning("⚠️ No models loaded - using fallback predictions")
                    
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
        if 'random_forest' in self.models:
            try:
                rf_pred = self.models['random_forest'].predict(features)[0]
                rf_proba = self.models['random_forest'].predict_proba(features)[0, 1]
                predictions['random_forest'] = {'prediction': rf_pred, 'probability': rf_proba}
            except Exception as e:
                st.warning(f"⚠️ Random Forest prediction failed: {e}")
        
        # MLP prediction
        if 'mlp_medium' in self.models:
            try:
                mlp_pred = self.models['mlp_medium'].predict(features)[0]
                mlp_proba = self.models['mlp_medium'].predict_proba(features)[0, 1]
                predictions['mlp_medium'] = {'prediction': mlp_pred, 'probability': mlp_proba}
            except Exception as e:
                st.warning(f"⚠️ MLP prediction failed: {e}")
        
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
        if 'rl_agent' not in self.models:
            return self.get_fallback_action(telemetry)
        
        try:
            # Discretize state (6D for new RL model)
            state = self.discretize_state(telemetry)
            
            # Get Q-values for this state
            q_table = self.models['rl_agent']
            
            # Check if q_table is a numpy array (7D) or dictionary
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
                    
                    # Optional debug output
                    if debug:
                        debug_color = "🔴" if is_untrained else "🟢"
                        debug_status = "UNTRAINED STATE" if is_untrained else "TRAINED STATE"
                        
                        st.sidebar.markdown(f"**{debug_color} RL Agent Debug - {debug_status}**")
                        st.sidebar.write(f"**Standardized Values:**")
                        st.sidebar.write(f"• Temp: {telemetry['temperature']:.2f}")
                        st.sidebar.write(f"• SoC: {telemetry['soc']:.2f}")
                        st.sidebar.write(f"• Voltage: {telemetry.get('voltage', 0.0):.2f}")
                        st.sidebar.write(f"**State:** {state}")
                        st.sidebar.write(f"**Q-values:** [{q_values[0]:.3f}, {q_values[1]:.3f}, {q_values[2]:.3f}, {q_values[3]:.3f}, {q_values[4]:.3f}]")
                        st.sidebar.write(f"**Q-sum:** {np.sum(q_values):.3f}")
                        
                        if is_untrained:
                            st.sidebar.warning("⚠️ State not encountered during training - using safety defaults")
                    
                    # Store debug info for logging
                    self.last_rl_debug_info = debug_info
                    
                    # Log untrained states separately for RL improvement
                    if is_untrained:
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
            
            return action, confidence
            
        except Exception as e:
            st.error(f"⚠️ RL Agent failed: {e}")
            st.error(f"   Using fallback action")
            return self.get_fallback_action(telemetry)
    
    def get_fallback_action(self, telemetry):
        """Rule-based fallback action selection"""
        # Simple safety rules using STANDARDIZED values
        temp = telemetry['temperature']  # Standardized temperature
        soc = telemetry['soc']          # Standardized SoC
        
        # Safety first approach (adjusted for standardized data)
        if temp > 2.0:  # Very hot (standardized)
            return 'pause', 0.9
        elif temp > 1.0:  # Hot (standardized)
            return 'slow_charge', 0.7
        elif soc < -2.0:  # Very low battery (standardized)
            return 'slow_charge', 0.8
        elif soc > 2.0:  # Very full battery (standardized)
            return 'pause', 0.8
        elif soc < -1.0:  # Low battery (standardized)
            return 'fast_charge', 0.6
        else:  # Normal conditions
            return 'maintain', 0.5
    
    def calculate_bhi(self, telemetry):
        """Calculate Battery Health Index"""
        # Simplified BHI calculation
        soc_factor = 1.0 - abs(telemetry['soc'] - 0.5) * 2  # Optimal around 50%
        temp_factor = max(0, 1.0 - abs(telemetry['temperature'] - 25) / 50)  # Optimal around 25°C
        voltage_factor = max(0, 1.0 - abs(telemetry['voltage'] - 3.7) / 2)  # Optimal around 3.7V
        
        bhi = (soc_factor + temp_factor + voltage_factor) / 3 * 100
        return max(0, min(100, bhi))
    
    def get_action_reason(self, telemetry, anomaly_predictions, rl_action, rl_confidence, safety_status):
        """Generate explanation for RL agent action"""
        temp = telemetry['temperature']
        soc = telemetry['soc']
        ensemble_prob = anomaly_predictions.get('ensemble', {}).get('probability', 0.5)
        
        # Determine primary reason for action
        if ensemble_prob > 0.8:
            return f"High anomaly detected ({ensemble_prob*100:.1f}%) - Safety protocol: {rl_action}"
        elif temp > 45:
            return f"Critical temperature ({temp:.1f}°C) - Emergency action: {rl_action}"
        elif soc < 0.1:
            return f"Critical low SoC ({soc*100:.1f}%) - Urgent charging: {rl_action}"
        elif soc > 0.9:
            return f"Battery nearly full ({soc*100:.1f}%) - Reduce charging: {rl_action}"
        elif rl_confidence < 0.5:
            return f"Low confidence ({rl_confidence*100:.0f}%) - Conservative action: {rl_action}"
        elif temp > 35:
            return f"Elevated temperature ({temp:.1f}°C) - Thermal management: {rl_action}"
        elif soc < 0.2:
            return f"Low SoC ({soc*100:.1f}%) - Charging recommended: {rl_action}"
        elif ensemble_prob > 0.6:
            return f"Moderate anomaly risk ({ensemble_prob*100:.1f}%) - Cautious action: {rl_action}"
        else:
            return f"Normal conditions - Optimal action: {rl_action} (confidence: {rl_confidence*100:.0f}%)"
    
    def log_untrained_rl_state(self, debug_info, telemetry):
        """Log untrained RL states separately for targeted training improvement"""
        import json
        from datetime import datetime
        
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
            
            # Check if this exact state already exists (avoid duplicates)
            state_signature = str(debug_info.get('state_bins', []))
            existing_signatures = [str(s.get('state_bins', [])) for s in untrained_states if isinstance(s, dict)]
            
            if state_signature not in existing_signatures:
                untrained_states.append(untrained_entry)
                
                # Write back to file with proper error handling
                with open(untrained_file, 'w') as f:
                    json.dump(untrained_states, f, indent=2, ensure_ascii=False)
                    f.flush()  # Ensure data is written
                    
        except Exception as e:
            # Fail silently to not interrupt dashboard
            pass

    def log_prediction_data(self, telemetry, features, anomaly_predictions, rl_action, rl_confidence, safety_status, action_reason=None, critical_count=0, warning_count=0):
        """Log prediction data for validation and analysis"""
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
            # Fail silently to not interrupt dashboard
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
    
    telemetry = {
        'timestamp': base_time,
        'voltage': np.random.uniform(*scenario['voltage_range']),
        'current': np.random.uniform(-50, 50),  # Negative = discharging, Positive = charging
        'temperature': np.random.uniform(*scenario['temp_range']),
        'soc': np.random.uniform(*scenario['soc_range']),
        'ambient_temp': np.random.uniform(15, 35),
        'humidity': np.random.uniform(0.3, 0.8),
        'charge_mode': np.random.choice(['fast', 'slow', 'pause']),
        'scenario': scenario['name'],
        'time_since_start': len(st.session_state.telemetry_data) * 5  # 5 seconds per reading
    }
    
    return telemetry

def main():
    # Header
    st.title("🔋 EV Battery Safety Management System")
    st.markdown("**Real-time Battery Monitoring, Anomaly Detection & Intelligent Action Recommendation**")
    
    # Initialize BMS
    bms = BatteryManagementSystem()
    
    # Sidebar controls
    st.sidebar.header("🎛️ System Controls")
    
    # Data transformation explanation
    # Debug toggle
    debug_rl = st.sidebar.checkbox("🔧 Debug RL Agent", value=False, 
                                   help="Show RL agent internal state, Q-values, and identify untrained states. Debug info is also logged for training improvements.")
    
    # Test critical conditions button
    if st.sidebar.button("🧪 Test Critical Conditions"):
        st.session_state.accumulated_alerts["critical"] += 1
        st.session_state.alert_history["critical"].append(f"{datetime.now().strftime('%H:%M:%S')}: Test critical alert")
        st.sidebar.success("Added test critical alert!")
    
    # Show logging status with file info
    log_file = Path("prediction_validation_log.json")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            log_count = len(logs)
            file_size = log_file.stat().st_size / 1024  # Size in KB
            
            # Check dedicated untrained states file
            untrained_file = Path("rl_untrained_states.json")
            untrained_count = 0
            untrained_file_size = 0
            
            if untrained_file.exists():
                try:
                    with open(untrained_file, 'r') as f:
                        untrained_states = json.load(f)
                    untrained_count = len(untrained_states)
                    untrained_file_size = untrained_file.stat().st_size / 1024  # Size in KB
                except:
                    untrained_count = 0
            
            status_text = f"📝 **Continuous Logging**: Active\n\n{log_count:,} entries logged ({file_size:.1f} KB)"
            if untrained_count > 0:
                status_text += f"\n\n🔴 **Untrained RL States**: {untrained_count:,} unique states ({untrained_file_size:.1f} KB)\n📁 `rl_untrained_states.json`"
                if untrained_count >= 10:  # Suggest retraining after collecting enough states
                    status_text += f"\n\n💡 **Ready for RL Improvement!**\nRun: `python scripts/retrain_rl_from_untrained_states.py`"
            status_text += f"\n\nAll predictions saved to `prediction_validation_log.json`"
            
            st.sidebar.info(status_text)
        except:
            st.sidebar.info("📝 **Continuous Logging**: Active\n\nAll inputs, predictions, and actions are continuously logged to `prediction_validation_log.json` for long-term analysis and validation.")
    else:
        st.sidebar.info("📝 **Continuous Logging**: Active\n\nAll inputs, predictions, and actions are continuously logged to `prediction_validation_log.json` for long-term analysis and validation.")
    
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
    
    # System status
    if st.sidebar.button("🚀 Start System" if not st.session_state.system_running else "⏹️ Stop System"):
        st.session_state.system_running = not st.session_state.system_running
    
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
                                    help="Relative humidity (affects thermal management)") / 100
        
        st.sidebar.write("**⚡ Charging Mode:**")
        charge_mode = st.sidebar.selectbox("Charge Mode", ['fast', 'slow', 'pause'],
                                          help="fast = Fast charging, slow = Slow charging, pause = No charging")
        
        # Show what the RL agent considers dangerous
        if hasattr(bms, 'transformer'):
            thresholds = bms.transformer.get_rl_thresholds_in_real_world()
            st.sidebar.write("**🚨 AI Safety Thresholds:**")
            st.sidebar.write(f"• High Temp: >{thresholds.get('high_temp_celsius', 41):.1f}°C")
            st.sidebar.write(f"• Low SoC: <{thresholds.get('low_soc_percent', 20):.0f}%")
            st.sidebar.write(f"• High SoC: >{thresholds.get('high_soc_percent', 80):.0f}%")
        
        if st.sidebar.button("📤 Submit Telemetry"):
            telemetry = {
                'timestamp': datetime.now(),
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'soc': soc,
                'ambient_temp': ambient_temp,
                'humidity': humidity,
                'charge_mode': charge_mode,
                'scenario': 'Manual Input',
                'time_since_start': len(st.session_state.telemetry_data) * 5
            }
            st.session_state.telemetry_data.append(telemetry)
    
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
        rl_action, rl_confidence = bms.get_rl_action(standardized_telemetry, debug_rl)
        bhi = bms.calculate_bhi(current_telemetry)
        
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
                "🔋 Battery Health",
                f"{bhi:.1f}%",
                delta=f"{bhi-75:.1f}%" if bhi < 75 else f"+{bhi-75:.1f}%",
                help="Battery Health Index"
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
        if "alert_history" not in st.session_state:
            st.session_state.alert_history = {"critical": [], "warning": []}
        
        # Initialize accumulated alert counters
        if "accumulated_alerts" not in st.session_state:
            st.session_state.accumulated_alerts = {
                "critical": 0,
                "warning": 0
            }
        
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
        bms.log_prediction_data(current_telemetry, features, anomaly_predictions, rl_action, rl_confidence, safety_status, action_reason, current_critical, current_warning)
        
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
        
        # Debug: Show what conditions are being checked
        if debug_rl:
            st.sidebar.write("**🔍 Condition Debug:**")
            st.sidebar.write(f"Safety Status: {safety_status}")
            st.sidebar.write(f"Ensemble Prob: {ensemble_prob:.3f} (>0.8: {ensemble_prob > 0.8})")
            st.sidebar.write(f"Temp: {current_telemetry['temperature']:.1f}°C (>40°C: {current_telemetry['temperature'] > 40})")
            st.sidebar.write(f"SoC: {current_telemetry['soc']*100:.1f}% (<15%: {current_telemetry['soc'] < 0.15})")
            st.sidebar.write(f"Voltage: {current_telemetry['voltage']:.2f}V (<3.2V: {current_telemetry['voltage'] < 3.2})")
            st.sidebar.write(f"Current Critical: {current_critical}")
            st.sidebar.write(f"Current Warning: {current_warning}")
        
        # Keep only recent alerts (last 10 of each type)
        st.session_state.alert_history["critical"] = st.session_state.alert_history["critical"][-10:]
        st.session_state.alert_history["warning"] = st.session_state.alert_history["warning"][-10:]
        
        if current_critical == 0 and current_warning == 0:
            st.success("✅ All systems operating normally")
        
        # Debug information (can be removed later)
        if debug_rl:  # Reuse the debug toggle
            st.sidebar.write("**🔍 Alert Debug Info:**")
            st.sidebar.write(f"Accumulated Critical: {accumulated_critical}")
            st.sidebar.write(f"Current Critical: {current_critical}")
            st.sidebar.write(f"Critical Delta: {critical_delta}")
            st.sidebar.write(f"Accumulated Warning: {accumulated_warning}")
            st.sidebar.write(f"Current Warning: {current_warning}")
            st.sidebar.write(f"Warning Delta: {warning_delta}")
        
        # Calculate deltas for proper change tracking
        critical_delta = current_critical  # Show current alerts as delta
        warning_delta = current_warning    # Show current alerts as delta
        
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
    
    # Auto-refresh for real-time updates
    if st.session_state.system_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
