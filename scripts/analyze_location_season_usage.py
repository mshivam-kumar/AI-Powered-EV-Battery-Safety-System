#!/usr/bin/env python3
"""
Analyze how location and season data is used throughout the system
"""

import json
import re
from pathlib import Path

def analyze_location_season_usage():
    """Analyze how location and season data is used in the system"""
    print("🔍 Location and Season Data Usage Analysis")
    print("=" * 60)
    
    # Check prediction validation log for location/season usage
    print("📊 PREDICTION VALIDATION LOG ANALYSIS:")
    print("-" * 40)
    
    try:
        with open('prediction_validation_log.json', 'r') as f:
            prediction_logs = json.load(f)
        
        print(f"📈 Total prediction entries: {len(prediction_logs)}")
        
        # Analyze location/season usage in logs
        location_usage = {}
        season_usage = {}
        climate_zone_usage = {}
        
        for entry in prediction_logs[:100]:  # Sample first 100 entries
            # Check input_telemetry
            if 'input_telemetry' in entry:
                telemetry = entry['input_telemetry']
                
                # Location usage
                if 'location' in telemetry:
                    location = telemetry['location']
                    location_usage[location] = location_usage.get(location, 0) + 1
                
                # Season usage
                if 'season' in telemetry:
                    season = telemetry['season']
                    season_usage[season] = season_usage.get(season, 0) + 1
                
                # Climate zone usage
                if 'climate_zone' in telemetry:
                    climate_zone = telemetry['climate_zone']
                    climate_zone_usage[climate_zone] = climate_zone_usage.get(climate_zone, 0) + 1
        
        print(f"🌍 Location Usage:")
        for location, count in sorted(location_usage.items()):
            print(f"   • {location}: {count} entries")
        
        print(f"🌦️ Season Usage:")
        for season, count in sorted(season_usage.items()):
            print(f"   • {season}: {count} entries")
        
        print(f"🌡️ Climate Zone Usage:")
        for climate_zone, count in sorted(climate_zone_usage.items()):
            print(f"   • {climate_zone}: {count} entries")
        
    except Exception as e:
        print(f"❌ Error analyzing prediction logs: {e}")
    
    print()
    
    # Analyze dashboard code for location/season usage
    print("📊 DASHBOARD CODE ANALYSIS:")
    print("-" * 40)
    
    try:
        with open('dashboard/app.py', 'r') as f:
            dashboard_code = f.read()
        
        # Find location/season usage patterns
        location_patterns = [
            r'climate_zone',
            r'season',
            r'location',
            r'get_adaptive_safety_thresholds',
            r'calculate_bhi',
            r'apply_climate_aware_adjustments',
            r'get_bhi_recommendations',
            r'get_fallback_action'
        ]
        
        print("🔍 Location/Season Usage in Dashboard:")
        for pattern in location_patterns:
            matches = re.findall(pattern, dashboard_code)
            if matches:
                print(f"   • {pattern}: {len(matches)} occurrences")
        
        # Find specific usage examples
        print("\n📋 Specific Usage Examples:")
        
        # Climate zone usage
        climate_zone_examples = re.findall(r'climate_zone.*?=.*?["\'].*?["\']', dashboard_code)
        if climate_zone_examples:
            print(f"   • Climate zone assignments: {len(climate_zone_examples)}")
        
        # Season usage
        season_examples = re.findall(r'season.*?=.*?["\'].*?["\']', dashboard_code)
        if season_examples:
            print(f"   • Season assignments: {len(season_examples)}")
        
        # Adaptive thresholds usage
        adaptive_examples = re.findall(r'get_adaptive_safety_thresholds.*?\(', dashboard_code)
        if adaptive_examples:
            print(f"   • Adaptive thresholds calls: {len(adaptive_examples)}")
        
        # BHI calculation usage
        bhi_examples = re.findall(r'calculate_bhi.*?\(', dashboard_code)
        if bhi_examples:
            print(f"   • BHI calculation calls: {len(bhi_examples)}")
        
    except Exception as e:
        print(f"❌ Error analyzing dashboard code: {e}")
    
    print()
    
    # Analyze how location/season affects system behavior
    print("📊 SYSTEM BEHAVIOR ANALYSIS:")
    print("-" * 40)
    
    print("1️⃣ ADAPTIVE SAFETY THRESHOLDS:")
    print("   • Location affects temperature thresholds")
    print("   • Season affects sensitivity adjustments")
    print("   • Climate zone determines base thresholds")
    print("   • Charging mode affects threshold adjustments")
    print()
    
    print("2️⃣ BATTERY HEALTH INDEX (BHI):")
    print("   • Climate zone factor (10% weight)")
    print("   • Season factor (3% weight)")
    print("   • Environmental health (25% weight)")
    print("   • Charging health (2% weight)")
    print()
    
    print("3️⃣ RL AGENT ADJUSTMENTS:")
    print("   • 20 climate/season combinations")
    print("   • If-else logic for action adjustments")
    print("   • Safety overrides for critical conditions")
    print("   • Confidence boosts for climate-aware actions")
    print()
    
    print("4️⃣ RECOMMENDATIONS:")
    print("   • Climate-specific charging advice")
    print("   • Season-specific safety recommendations")
    print("   • Location-specific environmental factors")
    print("   • Adaptive threshold explanations")
    print()
    
    print("5️⃣ LOGGING AND ANALYSIS:")
    print("   • Climate context in prediction logs")
    print("   • Location/season in telemetry data")
    print("   • Adaptive thresholds tracking")
    print("   • BHI calculations with climate factors")
    print()

def explain_location_season_benefits():
    """Explain the benefits of location and season data"""
    print("🎯 LOCATION AND SEASON DATA BENEFITS:")
    print("=" * 60)
    
    print("1️⃣ ADAPTIVE SAFETY THRESHOLDS:")
    print("-" * 40)
    print("🌡️ Temperature Thresholds:")
    print("   • Hot Desert: 35°C (more sensitive)")
    print("   • Tropical Monsoon: 37°C (more sensitive)")
    print("   • Subtropical Highland: 44°C (less sensitive)")
    print("   → Prevents thermal runaway in extreme climates")
    print()
    
    print("🌦️ Season Adjustments:")
    print("   • Summer: -2.5°C threshold (more sensitive)")
    print("   • Monsoon: -1.5°C threshold (more sensitive)")
    print("   • Winter: +2.0°C threshold (less sensitive)")
    print("   → Adapts to seasonal temperature variations")
    print()
    
    print("2️⃣ ENHANCED BATTERY HEALTH INDEX:")
    print("-" * 40)
    print("🔋 Climate Zone Health (10% weight):")
    print("   • Hot Desert: 0.80-0.85 (reduced in extreme heat)")
    print("   • Tropical Monsoon: 0.85-0.90 (reduced in humidity)")
    print("   • Subtropical Highland: 0.90-0.95 (good conditions)")
    print("   → Reflects climate impact on battery health")
    print()
    
    print("🌦️ Season Health (3% weight):")
    print("   • Summer: 0.80-0.90 (reduced in extreme heat)")
    print("   • Monsoon: 0.85-0.90 (reduced in humidity)")
    print("   • Winter: 0.88-0.95 (good in moderate cold)")
    print("   • Spring: 1.00 (optimal conditions)")
    print("   → Accounts for seasonal variations")
    print()
    
    print("3️⃣ RL AGENT CLIMATE-AWARE ACTIONS:")
    print("-" * 40)
    print("🌍 Climate Zone + Season Combinations (20 total):")
    print("   • Hot Desert + Summer: Fast charge → Pause (extreme heat)")
    print("   • Tropical Monsoon + Monsoon: Fast charge → Slow charge (humidity)")
    print("   • Subtropical Highland + Winter: Pause → Slow charge (cold)")
    print("   → Prevents dangerous actions in extreme conditions")
    print()
    
    print("4️⃣ REAL-WORLD SAFETY BENEFITS:")
    print("-" * 40)
    print("🚨 Prevents Thermal Runaway:")
    print("   • Desert conditions: More sensitive temperature monitoring")
    print("   • Monsoon conditions: Humidity-aware charging")
    print("   • Highland conditions: Cold weather protection")
    print()
    
    print("🔋 Optimizes Charging Strategy:")
    print("   • Summer: Reduces charging speed in extreme heat")
    print("   • Winter: Ensures charging in cold conditions")
    print("   • Monsoon: Protects against humidity damage")
    print()
    
    print("📊 Provides Context-Aware Recommendations:")
    print("   • Climate-specific advice")
    print("   • Season-specific warnings")
    print("   • Location-specific environmental factors")
    print("   • Adaptive threshold explanations")
    print()
    
    print("5️⃣ SYSTEM INTELLIGENCE:")
    print("-" * 40)
    print("🧠 Context-Aware Decision Making:")
    print("   • Same battery state → Different actions based on climate")
    print("   • Same temperature → Different thresholds based on location")
    print("   • Same SoC → Different recommendations based on season")
    print()
    
    print("📈 Continuous Learning:")
    print("   • Logs climate context for analysis")
    print("   • Tracks adaptive threshold effectiveness")
    print("   • Monitors BHI trends by climate zone")
    print("   • Improves recommendations over time")
    print()
    
    print("🎯 SUMMARY:")
    print("Location and season data makes the system:")
    print("   ✅ More intelligent (context-aware decisions)")
    print("   ✅ More safe (adaptive thresholds)")
    print("   ✅ More accurate (climate-specific BHI)")
    print("   ✅ More practical (real-world conditions)")
    print("   ✅ More robust (handles diverse environments)")

def main():
    """Main analysis function"""
    analyze_location_season_usage()
    explain_location_season_benefits()
    
    print("\n✅ Location and season usage analysis complete!")
    print("🎯 The system uses location/season data for intelligent, adaptive safety!")

if __name__ == "__main__":
    main()


