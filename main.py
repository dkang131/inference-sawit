import os
import logging
import sys

# =============================================================================
# GPU CONFIGURATION - EASY TOGGLE
# =============================================================================
# Set USE_GPU to True to enable GPU acceleration, False for CPU-only mode
USE_GPU = False  # Change this to True to enable GPU acceleration

# Set CUDA environment variables before importing any CUDA-dependent modules
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting

if not USE_GPU:
    # Force CPU-only mode to prevent CUDA crashes during video processing
    # RTX 5060 TI has compatibility issues with current PyTorch version
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    print("INFO: Running in CPU-only mode (USE_GPU = False)")
    print("INFO: RTX 5060 TI CUDA compatibility issues cause silent failures during video processing")
    print("INFO: To enable GPU: Change USE_GPU = True in main.py and restart")
else:
    print("INFO: GPU acceleration enabled (USE_GPU = True)")
    print("INFO: Will use CUDA if available and compatible")
    print("INFO: WARNING: Ensure PyTorch supports your GPU's compute capability!")
    print("INFO: If processing fails at ~60%, change USE_GPU = False and restart")

from dashboard.app import app
from extensions import db
from models import Video
from flask_migrate import Migrate
from flask import Flask  # pyright: ignore[reportMissingImports]
from config import dbConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Set specific log levels for different modules
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

app.config.from_object(dbConfig)
db.init_app(app)

migrate = Migrate(app, db)

def run_migrations():
    """Run database migrations to set up the database schema"""
    with app.app_context():
        try:
            print("Running database migrations...")
            
            # Use Flask-Migrate CLI commands programmatically
            from flask_migrate import upgrade
            
            # Run migrations to head
            upgrade()
            print("Database migrations completed successfully!")
        except Exception as e:
            print(f"Error running migrations: {e}")

if __name__ == '__main__':
    run_migrations()    
    app.run(debug=True, port=4786, host='0.0.0.0')