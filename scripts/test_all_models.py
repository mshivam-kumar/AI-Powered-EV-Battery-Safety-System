#!/usr/bin/env python3
"""
Comprehensive Model Testing & Ensemble
Test all trained models and create ensemble predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load test data for evaluation"""
        print("📁 Loading test data...")
        
        # Load features
        features_file = self.data_dir / "processed" / "features" / "extracted_features.parquet"
        features_df = pd.read_parquet(features_file)
        
        # Load labels
        labels_file = self.data_dir / "processed" / "labels" / "consensus_labels.npy"
        labels = np.load(labels_file)
        
        # Extract features (same as training)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        X = features_df[feature_columns].values
        
        # Use same train/test split as training
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, 
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        print(f"✅ Data loaded:")
        print(f"   • Training: {len(X_train):,} samples")
        print(f"   • Testing: {len(X_test):,} samples")
        print(f"   • Features: {X_test.shape[1]}")
        print(f"   • Test Anomaly Rate: {np.mean(y_test == 1):.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def load_model(self, model_name: str):
        """Load a trained model"""
        model_file = self.models_dir / f"{model_name}.pkl"
        
        if not model_file.exists():
            print(f"❌ Model not found: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded {model_name}")
            return model
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def evaluate_model(self, model, model_name: str, X_test, y_test):
        """Evaluate a single model"""
        print(f"\n🧪 Testing {model_name}...")
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # AUC if probabilities available
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 
                    'fn': int(fn), 'tp': int(tp)
                },
                'predictions': y_pred.tolist() if len(y_pred) < 10000 else None,
                'probabilities': y_proba.tolist() if y_proba is not None and len(y_proba) < 10000 else None
            }
            
            print(f"   ✅ Results:")
            print(f"      • Accuracy: {accuracy:.3f}")
            print(f"      • Precision: {precision:.3f}")
            print(f"      • Recall: {recall:.3f}")
            print(f"      • F1-Score: {f1:.3f}")
            if auc:
                print(f"      • AUC: {auc:.3f}")
            print(f"      • Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {e}")
            return None
    
    def create_ensemble_prediction(self, X_test):
        """Create ensemble predictions from all models"""
        print(f"\n🎯 Creating Ensemble Predictions...")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)[:, 1]
                    pred = (proba > 0.5).astype(int)
                    probabilities[model_name] = proba
                else:
                    pred = model.predict(X_test)
                    probabilities[model_name] = pred.astype(float)
                
                predictions[model_name] = pred
                
            except Exception as e:
                print(f"   ⚠️  Skipping {model_name}: {e}")
        
        if not predictions:
            print("   ❌ No valid predictions for ensemble")
            return None, None
        
        # Ensemble methods
        ensemble_results = {}
        
        # 1. Simple Majority Vote
        if len(predictions) >= 2:
            pred_array = np.array(list(predictions.values()))
            majority_vote = (np.mean(pred_array, axis=0) > 0.5).astype(int)
            ensemble_results['majority_vote'] = majority_vote
        
        # 2. Weighted Average (based on individual performance)
        if len(probabilities) >= 2:
            weights = {}
            for model_name in probabilities.keys():
                if model_name in self.results:
                    weights[model_name] = self.results[model_name]['f1_score']
                else:
                    weights[model_name] = 0.5  # Default weight
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weighted_proba = np.zeros(len(X_test))
                for model_name, proba in probabilities.items():
                    weight = weights[model_name] / total_weight
                    weighted_proba += weight * proba
                
                weighted_pred = (weighted_proba > 0.5).astype(int)
                ensemble_results['weighted_average'] = weighted_pred
                ensemble_results['weighted_probabilities'] = weighted_proba
        
        # 3. Conservative Ensemble (require higher confidence)
        if len(probabilities) >= 2:
            proba_array = np.array(list(probabilities.values()))
            conservative_proba = np.mean(proba_array, axis=0)
            conservative_pred = (conservative_proba > 0.7).astype(int)  # Higher threshold
            ensemble_results['conservative'] = conservative_pred
        
        print(f"   ✅ Created {len(ensemble_results)} ensemble methods")
        return ensemble_results, probabilities
    
    def run_comprehensive_testing(self):
        """Run comprehensive testing of all models"""
        print("🚀 Comprehensive Model Testing & Ensemble")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Load all available models
        model_files = {
            'random_forest': 'Random Forest',
            'mlp_small': 'MLP Small',
            'mlp_medium': 'MLP Medium'
        }
        
        print(f"\n📦 Loading Models...")
        print("-" * 30)
        
        for model_key, model_name in model_files.items():
            model = self.load_model(model_key)
            if model is not None:
                self.models[model_key] = model
        
        if not self.models:
            print("❌ No models loaded! Cannot proceed.")
            return None
        
        # Test each model individually
        print(f"\n🧪 Individual Model Testing...")
        print("-" * 40)
        
        for model_key, model in self.models.items():
            model_name = model_files.get(model_key, model_key)
            results = self.evaluate_model(model, model_name, X_test, y_test)
            if results:
                self.results[model_key] = results
        
        # Create ensemble predictions
        ensemble_preds, individual_probas = self.create_ensemble_prediction(X_test)
        
        if ensemble_preds:
            print(f"\n🎯 Ensemble Testing...")
            print("-" * 25)
            
            for ensemble_name, ensemble_pred in ensemble_preds.items():
                if ensemble_name == 'weighted_probabilities':
                    continue  # Skip probabilities
                
                accuracy = accuracy_score(y_test, ensemble_pred)
                precision = precision_score(y_test, ensemble_pred)
                recall = recall_score(y_test, ensemble_pred)
                f1 = f1_score(y_test, ensemble_pred)
                
                print(f"   {ensemble_name.replace('_', ' ').title()}:")
                print(f"      • Accuracy: {accuracy:.3f}")
                print(f"      • Precision: {precision:.3f}")
                print(f"      • Recall: {recall:.3f}")
                print(f"      • F1-Score: {f1:.3f}")
                
                self.results[f'ensemble_{ensemble_name}'] = {
                    'model_name': f'Ensemble {ensemble_name.replace("_", " ").title()}',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': None
                }
        
        # Model comparison
        self.print_model_comparison()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def print_model_comparison(self):
        """Print comprehensive model comparison"""
        print(f"\n🏆 Model Performance Comparison")
        print("=" * 60)
        
        # Sort by F1-score
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['f1_score'], 
                             reverse=True)
        
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10}")
        print("-" * 70)
        
        for model_key, results in sorted_models:
            model_name = results['model_name'][:24]
            accuracy = results['accuracy']
            precision = results['precision']
            recall = results['recall']
            f1 = results['f1_score']
            
            print(f"{model_name:<25} {accuracy:<10.3f} {precision:<11.3f} {recall:<8.3f} {f1:<10.3f}")
        
        # Best model
        if sorted_models:
            best_model_key, best_results = sorted_models[0]
            print(f"\n🥇 Best Model: {best_results['model_name']}")
            print(f"   F1-Score: {best_results['f1_score']:.3f}")
    
    def save_results(self):
        """Save all results to JSON"""
        results_file = self.models_dir / "comprehensive_test_results.json"
        
        # Convert numpy types for JSON
        json_results = {}
        for model_key, results in self.results.items():
            json_results[model_key] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in results.items()
                if k not in ['predictions', 'probabilities']  # Skip large arrays
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")

def main():
    """Main testing function"""
    tester = ModelEnsemble()
    results = tester.run_comprehensive_testing()
    
    if results:
        print(f"\n✅ Comprehensive Testing Complete!")
        print(f"📊 Tested {len([k for k in results.keys() if not k.startswith('ensemble')])} individual models")
        print(f"🎯 Created {len([k for k in results.keys() if k.startswith('ensemble')])} ensemble methods")
    else:
        print("❌ Testing failed!")

if __name__ == "__main__":
    main()
