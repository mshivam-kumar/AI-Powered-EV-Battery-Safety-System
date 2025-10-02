#!/usr/bin/env python3
"""
Master Pipeline Script
Run the complete EV Battery Safety System pipeline from start to finish
"""

import sys
import subprocess
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.scripts_dir = Path("scripts")
        self.pipeline_steps = [
            ("01_data_preparation.py", "Data Preparation"),
            ("02_preprocessing.py", "Preprocessing & Feature Engineering"),
            ("03_synthetic_labels_complete.py", "Complete Dataset Label Generation (All 5.3M samples)"),
            ("train_rf_complete.py", "Random Forest Training (Parallel)"),
            ("train_mlp_complete.py", "MLP Training (Parallel)"),
            ("05_testing.py", "Model Testing & Validation"),
            ("06_inference.py", "Inference & Deployment Demo")
        ]
    
    def run_step(self, script_name: str, step_name: str) -> bool:
        """Run a single pipeline step"""
        logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            script_path = self.scripts_dir / script_name
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, check=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ {step_name} completed successfully in {duration:.1f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {step_name} failed!")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("="*60)
        logger.info("EV BATTERY SAFETY SYSTEM - FULL PIPELINE")
        logger.info("="*60)
        
        start_time = time.time()
        successful_steps = 0
        
        for script_name, step_name in self.pipeline_steps:
            logger.info(f"\n--- {step_name} ---")
            
            if self.run_step(script_name, step_name):
                successful_steps += 1
            else:
                logger.error(f"Pipeline stopped at {step_name}")
                break
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total steps: {len(self.pipeline_steps)}")
        logger.info(f"Successful steps: {successful_steps}")
        logger.info(f"Failed steps: {len(self.pipeline_steps) - successful_steps}")
        logger.info(f"Total duration: {total_duration:.1f}s")
        
        if successful_steps == len(self.pipeline_steps):
            logger.info("🎉 Full pipeline completed successfully!")
            return True
        else:
            logger.error("❌ Pipeline failed!")
            return False
    
    def run_single_step(self, step_number: int):
        """Run a single pipeline step"""
        if 1 <= step_number <= len(self.pipeline_steps):
            script_name, step_name = self.pipeline_steps[step_number - 1]
            return self.run_step(script_name, step_name)
        else:
            logger.error(f"Invalid step number: {step_number}")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EV Battery Safety System Pipeline")
    parser.add_argument("--step", type=int, help="Run specific step (1-6)")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.step:
        success = runner.run_single_step(args.step)
    elif args.full:
        success = runner.run_full_pipeline()
    else:
        # Default: run full pipeline
        success = runner.run_full_pipeline()
    
    if success:
        print("\n✅ Pipeline execution completed successfully!")
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
