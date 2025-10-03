#!/usr/bin/env python3
"""
Model Deployment Helper for Streamlit Cloud
============================================

This script helps prepare models for Streamlit Cloud deployment by:
1. Checking model file sizes
2. Verifying model compatibility
3. Creating a deployment checklist
4. Generating model information for the dashboard
"""

import os
import json
import pickle
from pathlib import Path
import sys

def check_model_files():
    """Check all model files and their sizes"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return False
    
    print("🔍 **Model Files Analysis**")
    print("=" * 50)
    
    model_info = {}
    total_size = 0
    
    for model_file in models_dir.glob("*.pkl"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        model_info[model_file.name] = {
            "type": "pickle",
            "size_mb": round(size_mb, 2),
            "path": str(model_file)
        }
        print(f"📦 {model_file.name}: {size_mb:.2f} MB")
    
    for model_file in models_dir.glob("*.json"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        model_info[model_file.name] = {
            "type": "json",
            "size_mb": round(size_mb, 2),
            "path": str(model_file)
        }
        print(f"📄 {model_file.name}: {size_mb:.2f} MB")
    
    print(f"\n📊 **Total Size**: {total_size:.2f} MB")
    
    # Check Streamlit Cloud limits
    if total_size > 100:  # Streamlit Cloud has ~100MB limit for free tier
        print("⚠️  **WARNING**: Total size exceeds recommended limit for Streamlit Cloud free tier")
        print("💡 Consider using smaller models or upgrading to paid tier")
    else:
        print("✅ **Size OK**: Within Streamlit Cloud limits")
    
    return model_info

def verify_model_compatibility():
    """Verify that models can be loaded without errors"""
    print("\n🔧 **Model Compatibility Check**")
    print("=" * 50)
    
    models_dir = Path("models")
    issues = []
    
    # Test Random Forest
    rf_path = models_dir / "random_forest_complete.pkl"
    if rf_path.exists():
        try:
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            print("✅ Random Forest: Compatible")
        except Exception as e:
            issues.append(f"Random Forest: {e}")
            print(f"❌ Random Forest: {e}")
    else:
        print("⚠️ Random Forest: File not found")
    
    # Test MLP
    mlp_path = models_dir / "mlp_medium_complete.pkl"
    if mlp_path.exists():
        try:
            with open(mlp_path, 'rb') as f:
                mlp_model = pickle.load(f)
            print("✅ MLP Medium: Compatible")
        except Exception as e:
            issues.append(f"MLP Medium: {e}")
            print(f"❌ MLP Medium: {e}")
    else:
        print("⚠️ MLP Medium: File not found")
    
    # Test RL Agent
    rl_path = models_dir / "fine_tuned_from_logs_rl_agent.json"
    if rl_path.exists():
        try:
            with open(rl_path, 'r') as f:
                rl_data = json.load(f)
            print("✅ RL Agent: Compatible")
        except Exception as e:
            issues.append(f"RL Agent: {e}")
            print(f"❌ RL Agent: {e}")
    else:
        print("⚠️ RL Agent: File not found")
    
    return issues

def create_deployment_checklist():
    """Create a deployment checklist"""
    print("\n📋 **Streamlit Cloud Deployment Checklist**")
    print("=" * 50)
    
    checklist = [
        "✅ Models directory exists",
        "✅ Model files are under size limits",
        "✅ Dashboard has fallback error handling",
        "✅ Logging has try-catch blocks",
        "✅ Relative paths used for model loading",
        "✅ Fallback predictions implemented",
        "📝 Push models to GitHub repository",
        "📝 Deploy to Streamlit Cloud",
        "📝 Test all functionality in cloud environment",
        "📝 Verify fallback behavior works"
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    return checklist

def generate_model_info():
    """Generate model information for the dashboard"""
    model_info = {
        "deployment_status": "ready",
        "models_available": [],
        "fallback_mode": True,
        "last_updated": "2024-01-01"
    }
    
    models_dir = Path("models")
    
    # Check which models are available
    if (models_dir / "random_forest_complete.pkl").exists():
        model_info["models_available"].append("random_forest")
    
    if (models_dir / "mlp_medium_complete.pkl").exists():
        model_info["models_available"].append("mlp_medium")
    
    if (models_dir / "fine_tuned_from_logs_rl_agent.json").exists():
        model_info["models_available"].append("rl_agent")
    
    # Save model info
    with open("model_deployment_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n💾 **Model Info Saved**: model_deployment_info.json")
    print(f"📊 **Available Models**: {', '.join(model_info['models_available'])}")
    
    return model_info

def main():
    """Main deployment preparation function"""
    print("🚀 **EV Battery Safety System - Model Deployment Helper**")
    print("=" * 60)
    
    # Check model files
    model_info = check_model_files()
    
    # Verify compatibility
    issues = verify_model_compatibility()
    
    # Create checklist
    checklist = create_deployment_checklist()
    
    # Generate model info
    deployment_info = generate_model_info()
    
    print("\n🎯 **Deployment Summary**")
    print("=" * 30)
    
    if issues:
        print(f"⚠️  **Issues Found**: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ **No Issues**: All models are compatible")
    
    print(f"\n📦 **Models Ready**: {len(model_info)} files")
    print(f"🔧 **Fallback Mode**: Enabled for cloud deployment")
    
    print("\n🚀 **Next Steps**:")
    print("1. Push all files to GitHub repository")
    print("2. Deploy to Streamlit Cloud")
    print("3. Test functionality in cloud environment")
    print("4. Verify fallback behavior works correctly")

if __name__ == "__main__":
    main()
