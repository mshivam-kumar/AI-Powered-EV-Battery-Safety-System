#!/usr/bin/env python3
"""
Cleanup Untrained States File
=============================

Simple utility to delete the untrained states file after successful retraining.

Usage:
    python cleanup_untrained_states.py
"""

import os
from pathlib import Path
from datetime import datetime

def cleanup_untrained_states():
    """Delete the untrained states file and create a backup if needed"""
    
    untrained_file = Path("../rl_untrained_states.json")
    
    if not untrained_file.exists():
        print("✅ No untrained states file found - already clean!")
        return
    
    # Get file info
    file_size = untrained_file.stat().st_size / 1024  # KB
    
    # Ask for confirmation
    print(f"📁 Found: rl_untrained_states.json ({file_size:.1f} KB)")
    response = input("🗑️  Delete this file? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        try:
            # Create backup first
            backup_name = f"rl_untrained_states_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = Path("../temp") / backup_name
            
            # Ensure temp directory exists
            backup_path.parent.mkdir(exist_ok=True)
            
            # Copy to backup
            import shutil
            shutil.copy2(untrained_file, backup_path)
            print(f"💾 Backup created: {backup_path}")
            
            # Delete original
            untrained_file.unlink()
            print(f"✅ Deleted: rl_untrained_states.json")
            print(f"🎯 Ready for new untrained state collection!")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    else:
        print("❌ Cleanup cancelled")

if __name__ == "__main__":
    print("🧹 UNTRAINED STATES CLEANUP UTILITY")
    print("=" * 40)
    cleanup_untrained_states()
