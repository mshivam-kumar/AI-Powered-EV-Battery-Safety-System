#!/usr/bin/env python3
"""
Comprehensive Feature Analysis for Project Report
Analyze how each of the 16 features contributes to anomaly detection
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json

def analyze_all_features():
    """Analyze all 16 features for anomaly detection patterns"""
    print("🔍 Comprehensive Feature Analysis for Project Report")
    print("=" * 70)
    
    # Load the trained RF model
    models_dir = Path("models")
    rf_path = models_dir / "random_forest_complete.pkl"
    
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    print("✅ Loaded Random Forest Complete model")
    
    # Load test data
    splits_dir = Path("data/processed/splits_complete")
    X_test = pd.read_parquet(splits_dir / "test_features.parquet")
    y_test = pd.read_csv(splits_dir / "test_labels.csv")['anomaly'].values
    
    # Remove metadata columns
    feature_cols = [col for col in X_test.columns if col not in ['battery_id', 'battery_type', 'time']]
    X_test = X_test[feature_cols]
    
    print(f"✅ Loaded test data: {len(X_test):,} samples with {len(feature_cols)} features")
    
    # Get feature importance from the model
    feature_importance = dict(zip(X_test.columns, rf_model.feature_importances_))
    
    # Feature descriptions and physical meanings
    feature_descriptions = {
        'voltage': {
            'description': 'Battery terminal voltage',
            'unit': 'Volts (V)',
            'physical_meaning': 'Electrical potential difference across battery terminals',
            'anomaly_indicator': 'Voltage spikes/drops indicate cell damage or connection issues'
        },
        'current': {
            'description': 'Battery current flow',
            'unit': 'Amperes (A)',
            'physical_meaning': 'Rate of electrical charge flow (+ charging, - discharging)',
            'anomaly_indicator': 'Excessive current indicates overcharging or internal shorts'
        },
        'temperature': {
            'description': 'Battery core temperature',
            'unit': 'Celsius (°C)',
            'physical_meaning': 'Internal heat generation from chemical reactions',
            'anomaly_indicator': 'High temperature indicates thermal runaway risk'
        },
        'soc': {
            'description': 'State of Charge',
            'unit': 'Percentage (0-1)',
            'physical_meaning': 'Available energy capacity relative to maximum',
            'anomaly_indicator': 'Extreme SoC values indicate charging system issues'
        },
        'ambient_temp': {
            'description': 'Environmental temperature',
            'unit': 'Celsius (°C)',
            'physical_meaning': 'External temperature affecting battery performance',
            'anomaly_indicator': 'Extreme ambient conditions stress battery systems'
        },
        'humidity': {
            'description': 'Environmental humidity',
            'unit': 'Percentage (0-1)',
            'physical_meaning': 'Moisture content in surrounding air',
            'anomaly_indicator': 'High humidity increases corrosion and short-circuit risk'
        },
        'charge_mode_encoded': {
            'description': 'Charging mode type',
            'unit': 'Categorical (0,1,2)',
            'physical_meaning': 'Type of charging: maintain, slow_charge, fast_charge',
            'anomaly_indicator': 'Inappropriate charging mode for current conditions'
        },
        'power': {
            'description': 'Electrical power',
            'unit': 'Watts (W)',
            'physical_meaning': 'Rate of energy transfer (voltage × current)',
            'anomaly_indicator': 'Power spikes indicate electrical system stress'
        },
        'c_rate': {
            'description': 'Charge/discharge rate',
            'unit': 'C-rate (1C = 1 hour discharge)',
            'physical_meaning': 'Current relative to battery capacity',
            'anomaly_indicator': 'High C-rates cause heat generation and degradation'
        },
        'temp_diff': {
            'description': 'Temperature differential',
            'unit': 'Celsius (°C)',
            'physical_meaning': 'Difference between battery and ambient temperature',
            'anomaly_indicator': 'Large temperature differential indicates poor heat management'
        },
        'voltage_soc_ratio': {
            'description': 'Voltage to SoC ratio',
            'unit': 'V per unit SoC',
            'physical_meaning': 'Relationship between voltage and charge state',
            'anomaly_indicator': 'Abnormal ratio indicates cell degradation or calibration issues'
        },
        'thermal_stress': {
            'description': 'Thermal stress index',
            'unit': 'Normalized (0-1)',
            'physical_meaning': 'Combined temperature and environmental stress factor',
            'anomaly_indicator': 'High thermal stress accelerates battery aging'
        },
        'temp_gradient': {
            'description': 'Temperature change rate',
            'unit': 'Celsius per time unit',
            'physical_meaning': 'Speed of temperature change over time',
            'anomaly_indicator': 'Rapid temperature changes indicate thermal instability'
        },
        'voltage_gradient': {
            'description': 'Voltage change rate',
            'unit': 'Volts per time unit',
            'physical_meaning': 'Speed of voltage change over time',
            'anomaly_indicator': 'Rapid voltage changes indicate electrical instability'
        },
        'soc_rate': {
            'description': 'SoC change rate',
            'unit': 'Percentage per time unit',
            'physical_meaning': 'Speed of charge/discharge process',
            'anomaly_indicator': 'Abnormal SoC rates indicate charging system issues'
        },
        'env_stress': {
            'description': 'Environmental stress index',
            'unit': 'Normalized (0-1)',
            'physical_meaning': 'Combined environmental factors (temp, humidity)',
            'anomaly_indicator': 'High environmental stress affects battery performance'
        }
    }
    
    # Analyze each feature
    feature_analysis = []
    
    print(f"\n📊 Analyzing each feature for anomaly detection patterns...")
    print("-" * 120)
    
    for i, feature in enumerate(X_test.columns):
        print(f"Analyzing {feature}... ({i+1}/{len(X_test.columns)})")
        
        feature_values = X_test[feature]
        importance = feature_importance[feature]
        
        # Basic statistics
        stats = {
            'mean': feature_values.mean(),
            'std': feature_values.std(),
            'min': feature_values.min(),
            'max': feature_values.max(),
            'range': feature_values.max() - feature_values.min()
        }
        
        # Percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(feature_values, p)
        
        # Anomaly analysis at different thresholds
        thresholds_analysis = {}
        
        # Test different percentile thresholds
        for threshold_pct in [90, 95, 99]:
            threshold_val = np.percentile(feature_values, threshold_pct)
            high_mask = feature_values > threshold_val
            
            if np.sum(high_mask) > 0:
                high_samples = np.sum(high_mask)
                high_anomalies = np.sum(y_test[high_mask])
                anomaly_rate = (high_anomalies / high_samples) * 100 if high_samples > 0 else 0
                
                thresholds_analysis[f'>{threshold_pct}th_percentile'] = {
                    'threshold_value': threshold_val,
                    'sample_count': high_samples,
                    'anomaly_count': high_anomalies,
                    'anomaly_rate': anomaly_rate
                }
        
        # Test low values too (for features where low values might be anomalous)
        for threshold_pct in [1, 5, 10]:
            threshold_val = np.percentile(feature_values, threshold_pct)
            low_mask = feature_values < threshold_val
            
            if np.sum(low_mask) > 0:
                low_samples = np.sum(low_mask)
                low_anomalies = np.sum(y_test[low_mask])
                anomaly_rate = (low_anomalies / low_samples) * 100 if low_samples > 0 else 0
                
                thresholds_analysis[f'<{threshold_pct}th_percentile'] = {
                    'threshold_value': threshold_val,
                    'sample_count': low_samples,
                    'anomaly_count': low_anomalies,
                    'anomaly_rate': anomaly_rate
                }
        
        # Find the threshold with highest anomaly rate
        best_threshold = None
        max_anomaly_rate = 0
        for thresh_name, thresh_data in thresholds_analysis.items():
            if thresh_data['anomaly_rate'] > max_anomaly_rate and thresh_data['sample_count'] > 100:
                max_anomaly_rate = thresh_data['anomaly_rate']
                best_threshold = {
                    'name': thresh_name,
                    'value': thresh_data['threshold_value'],
                    'anomaly_rate': thresh_data['anomaly_rate'],
                    'sample_count': thresh_data['sample_count']
                }
        
        # Compile feature analysis
        analysis = {
            'feature_name': feature,
            'importance_rank': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True).index((feature, importance)) + 1,
            'importance_score': importance,
            'description': feature_descriptions.get(feature, {}).get('description', 'Unknown feature'),
            'unit': feature_descriptions.get(feature, {}).get('unit', 'Unknown'),
            'physical_meaning': feature_descriptions.get(feature, {}).get('physical_meaning', 'Unknown'),
            'anomaly_indicator': feature_descriptions.get(feature, {}).get('anomaly_indicator', 'Unknown'),
            'statistics': stats,
            'percentiles': percentiles,
            'thresholds_analysis': thresholds_analysis,
            'best_anomaly_threshold': best_threshold
        }
        
        feature_analysis.append(analysis)
    
    # Sort by importance
    feature_analysis.sort(key=lambda x: x['importance_score'], reverse=True)
    
    # Create comprehensive table
    print(f"\n📋 COMPREHENSIVE FEATURE ANALYSIS TABLE")
    print("=" * 150)
    
    # Header
    header = f"{'Rank':<4} {'Feature':<20} {'Importance':<10} {'Unit':<15} {'Best Anomaly Threshold':<25} {'Anomaly Rate':<12} {'Physical Meaning':<30}"
    print(header)
    print("-" * 150)
    
    # Rows
    for analysis in feature_analysis:
        rank = analysis['importance_rank']
        feature = analysis['feature_name']
        importance = f"{analysis['importance_score']:.3f}"
        unit = analysis['unit'][:14]  # Truncate if too long
        
        if analysis['best_anomaly_threshold']:
            threshold_desc = f"{analysis['best_anomaly_threshold']['name']}"[:24]
            anomaly_rate = f"{analysis['best_anomaly_threshold']['anomaly_rate']:.1f}%"
        else:
            threshold_desc = "No clear threshold"
            anomaly_rate = "N/A"
        
        physical = analysis['physical_meaning'][:29]  # Truncate if too long
        
        row = f"{rank:<4} {feature:<20} {importance:<10} {unit:<15} {threshold_desc:<25} {anomaly_rate:<12} {physical:<30}"
        print(row)
    
    # Save detailed analysis to JSON
    results_path = Path("models/comprehensive_feature_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(feature_analysis, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: {results_path}")
    
    # Create summary for project report
    print(f"\n📊 SUMMARY FOR PROJECT REPORT")
    print("=" * 50)
    
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, analysis in enumerate(feature_analysis[:10]):
        print(f"{i+1:2d}. {analysis['feature_name']:<20} | {analysis['importance_score']:.3f} | {analysis['description']}")
    
    print(f"\n🔥 FEATURES WITH HIGHEST ANOMALY DETECTION RATES:")
    # Sort by best anomaly rate
    anomaly_sorted = [a for a in feature_analysis if a['best_anomaly_threshold']]
    anomaly_sorted.sort(key=lambda x: x['best_anomaly_threshold']['anomaly_rate'], reverse=True)
    
    for i, analysis in enumerate(anomaly_sorted[:5]):
        thresh = analysis['best_anomaly_threshold']
        print(f"{i+1}. {analysis['feature_name']:<20} | {thresh['anomaly_rate']:.1f}% | {thresh['name']}")
    
    return feature_analysis

if __name__ == "__main__":
    feature_analysis = analyze_all_features()
