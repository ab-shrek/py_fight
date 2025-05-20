import subprocess
import time
import argparse
import logging
import os
import glob
import json
import requests
import sys
import gflags
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# Define gflags
FLAGS = gflags.FLAGS
# http://localhost:5001 for testing locally
gflags.DEFINE_string('server_url', 'http://localhost:5000', 'URL of the training server (can be local or remote)')
gflags.DEFINE_integer('instances', 1, 'Number of parallel game instances to run')
gflags.DEFINE_integer('cycles', 1, 'Number of cycles to run for each instance')

# Configure logging to only write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parallel_games.log", mode='a')  # 'a' for append mode
    ]
)


import os
import subprocess
import logging


# python run_parallel_games.py --server_url=http://localhost:5000 --instances=1 --cycles=1

def start_server():
    """Start the GPU training server"""
    logging.info("Building Go files...")
    build_process = subprocess.run(["go", "build", "gpu_training_server.go", "base_model.go"], check=True)
    if build_process.returncode != 0:
        raise RuntimeError("Failed to build Go files")
    
    logging.info("Starting GPU training server...")
    server_process = subprocess.Popen(["./base_model"])
    return server_process

def save_model(cycle_num):
    """Save the current model state"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"model_cycle{cycle_num}_{timestamp}.json"
    
    try:
        logging.info(f"Saving model to {checkpoint_path}")
        response = requests.post(
            f"{FLAGS.server_url}/save_model",
            json={"path": checkpoint_path},
            timeout=300000000
        )
        if response.status_code == 200:
            logging.info(f"Successfully saved model to {checkpoint_path}")
        else:
            logging.error(f"Error saving model: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def run_game_instance(instance_id, cycle, is_first_in_cycle=False):
    """Run a single game instance for one cycle"""
    logging.info(f"Instance {instance_id} starting cycle {cycle + 1} ({'first' if is_first_in_cycle else 'random'} in cycle)")
    # subprocess.run(["python", "game.py", "--use_ai", "--server_url=http://localhost:5001", f"--headless={not is_first_in_cycle}", f"--is_first_in_cycle={is_first_in_cycle}"])
    subprocess.run(["python", "game.py", "--use_ai", f"--server_url={FLAGS.server_url}", f"--headless=True"])
    logging.info(f"Instance {instance_id} completed cycle {cycle + 1}")

def train_from_files():
    """Train on all experience files and then delete them"""
    experience_files = glob.glob("*.jsonl")
    if not experience_files:
        logging.info("No experience files found for training")
        return
    
    all_experiences = []
    for file_path in experience_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    experience = json.loads(line.strip())
                    all_experiences.append(experience)
            logging.info(f"Loaded {len(all_experiences)} experiences from {file_path}")
        except Exception as e:
            logging.error(f"Error reading experience file {file_path}: {e}")
            continue
    
    if not all_experiences:
        logging.info("No experiences loaded for training")
        return
    
    # Send experiences to training server
    try:
        logging.info(f"Sending {len(all_experiences)} experiences for training")
        response = requests.post(
            f"{FLAGS.server_url}/train",
            json={"experiences": all_experiences},
            timeout=3000000
        )
        if response.status_code == 200:
            logging.info("Successfully trained on experiences")
            # Only delete files after successful training
            for file_path in experience_files:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted processed file: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {e}")
        else:
            logging.error(f"Error during training: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error sending experiences for training: {e}")
        return

def main():
    try:
        argv = FLAGS(sys.argv)
    except gflags.FlagsError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Start the training server
    server_process = start_server()
    try:
        # Give the server time to start
        time.sleep(2)
        
        # Run cycles sequentially
        for cycle in range(FLAGS.cycles):
            logging.info(f"Starting cycle {cycle + 1}/{FLAGS.cycles}")
            
            # Run all instances in parallel for this cycle
            with ThreadPoolExecutor(max_workers=FLAGS.instances) as executor:
                # Submit all instances to run in parallel
                futures = [
                    executor.submit(run_game_instance, instance, cycle, instance == 0)  # First instance uses model
                    for instance in range(FLAGS.instances)
                ]
                
                # Wait for all instances to complete this cycle
                for future in futures:
                    future.result()
            
            # After all instances complete this cycle:
            # 1. Train on collected experiences
            train_from_files()
            
            # 2. Save model checkpoint
            save_model(cycle + 1)
            
            logging.info(f"Completed cycle {cycle + 1}/{FLAGS.cycles}")
            
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        # Clean up server process
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main() 