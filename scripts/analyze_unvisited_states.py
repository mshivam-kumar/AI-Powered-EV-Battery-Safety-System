#!/usr/bin/env python3
"""
Analyze unvisited states and generate targeted scenarios for fine-tuning
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set

def analyze_unvisited_states(q_table_path: str = 'models/climate_aware_rl_agent.json'):
    """Analyze which states were not visited during training"""
    
    # Load the trained agent
    try:
        with open(q_table_path, 'r') as f:
            agent_data = json.load(f)
        q_table = agent_data['q_table']
        print(f"✅ Loaded agent with {len(q_table)} states")
    except FileNotFoundError:
        print(f"❌ Agent file not found: {q_table_path}")
        return None
    
    # Define state space dimensions (5x5x5x4 = 500 states)
    state_dims = [5, 5, 5, 4]
    total_states = np.prod(state_dims)
    
    # Find visited states (non-zero Q-values)
    visited_states = set()
    
    # Q-table is stored as nested list [soc][temp][ambient][voltage][action]
    for soc in range(5):
        for temp in range(5):
            for ambient in range(5):
                for voltage in range(4):
                    # Check if any action has non-zero Q-value
                    q_values = q_table[soc][temp][ambient][voltage]
                    if any(q != 0 for q in q_values):
                        visited_states.add((soc, temp, ambient, voltage))
    
    # Find unvisited states
    all_possible_states = set()
    for soc in range(5):
        for temp in range(5):
            for ambient in range(5):
                for voltage in range(4):
                    all_possible_states.add((soc, temp, ambient, voltage))
    
    unvisited_states = all_possible_states - visited_states
    
    print(f"📊 State Coverage Analysis:")
    print(f"   • Total possible states: {len(all_possible_states)}")
    print(f"   • Visited states: {len(visited_states)}")
    print(f"   • Unvisited states: {len(unvisited_states)}")
    print(f"   • Coverage: {len(visited_states)/len(all_possible_states)*100:.1f}%")
    
    return visited_states, unvisited_states, state_dims

def generate_targeted_scenarios(unvisited_states: Set[Tuple], num_scenarios: int = 1000) -> List[Dict]:
    """Generate targeted scenarios for unvisited states"""
    
    scenarios = []
    
    # Convert unvisited states to scenarios
    for state in list(unvisited_states)[:num_scenarios]:
        soc_bin, temp_bin, ambient_bin, voltage_bin = state
        
        # Convert bins back to actual values
        soc = soc_bin / 4.0  # 0-1 range
        temp = (temp_bin / 4.0) * 6 - 3  # -3 to 3 range, then denormalize
        ambient = (ambient_bin / 4.0) * 6 - 3  # -3 to 3 range, then denormalize  
        voltage = (voltage_bin / 3.0) * 6 - 3  # -3 to 3 range, then denormalize
        
        # Add some variation to make scenarios more realistic
        soc += np.random.uniform(-0.05, 0.05)
        temp += np.random.uniform(-2, 2)
        ambient += np.random.uniform(-2, 2)
        voltage += np.random.uniform(-0.1, 0.1)
        
        # Clamp to realistic ranges
        soc = max(0, min(1, soc))
        temp = max(-3, min(3, temp))
        ambient = max(-3, min(3, ambient))
        voltage = max(-3, min(3, voltage))
        
        scenario = {
            'soc': soc,
            'temperature': temp,
            'ambient_temp': ambient,
            'voltage': voltage,
            'current': np.random.uniform(-2, 2),
            'humidity': np.random.uniform(0.3, 0.9),
            'climate_zone': np.random.choice([
                'Tropical Monsoon (Mumbai, Kolkata, Chennai, Goa)',
                'Hot Desert (Rajasthan, Gujarat, Punjab)',
                'Tropical Savanna (Delhi, Bangalore, Hyderabad)',
                'Subtropical Highland (Shimla, Kashmir, Himachal)',
                'Tropical Wet (Kerala, Assam, Meghalaya)'
            ]),
            'season': np.random.choice(['summer', 'monsoon', 'winter', 'spring']),
            'is_anomaly': np.random.random() < 0.3,  # 30% anomaly rate
            'data_type': 'targeted_synthetic',
            'target_state': state
        }
        
        scenarios.append(scenario)
    
    print(f"✅ Generated {len(scenarios)} targeted scenarios for unvisited states")
    return scenarios

def save_targeted_scenarios(scenarios: List[Dict], output_path: str = 'data/targeted_scenarios.json'):
    """Save targeted scenarios to file"""
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"💾 Saved {len(scenarios)} targeted scenarios to {output_path}")

def main():
    """Main analysis function"""
    print("🔍 Analyzing Unvisited States for Targeted Fine-tuning")
    print("=" * 60)
    
    # Analyze unvisited states
    result = analyze_unvisited_states()
    if result is None:
        return
    
    visited_states, unvisited_states, state_dims = result
    
    # Generate targeted scenarios
    print(f"\n🎯 Generating Targeted Scenarios...")
    targeted_scenarios = generate_targeted_scenarios(unvisited_states, num_scenarios=2000)
    
    # Save scenarios
    save_targeted_scenarios(targeted_scenarios)
    
    # Show some examples
    print(f"\n📋 Sample Targeted Scenarios:")
    for i, scenario in enumerate(targeted_scenarios[:5]):
        print(f"   {i+1}. State {scenario['target_state']}: SoC={scenario['soc']:.2f}, Temp={scenario['temperature']:.1f}, Ambient={scenario['ambient_temp']:.1f}, Voltage={scenario['voltage']:.1f}")
    
    print(f"\n✅ Analysis complete! Use targeted scenarios for fine-tuning.")

if __name__ == "__main__":
    main()
