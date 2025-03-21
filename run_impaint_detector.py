"""Inpainting detection in artwork module."""
import sys
import os
from pathlib import Path

# Set up path
FILEPATH = Path(__file__).parent
sys.path.append(f'{FILEPATH}')

# Import the InpaintingDetector class
from inpainting_detector import InpaintingDetector

# Set GPU device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # Step 1: Create and configure the detector
    detector = InpaintingDetector(
        config_name='cnn_baseline',
        image_size=256,
        batch_size=16,
        max_epochs=25,
        learning_rate=1e-4,
        model_type="cnn"  # Options: "cnn", "resnet", "efficientnet"
    )
    
    # Step 2: Prepare datasets - extract and process images
    PREPARE_DATA = False
    if PREPARE_DATA:
        print("Preparing datasets...")
        detector.prepare_datasets(source_dir='inpainting')
    
    # Step 3: Train the model
    print(f"Training {detector.model_type} model...")
    detector.train_model()
    
    # Step 4: Visualize training results
    detector.plot_training_history()
    
    # Step 5: Evaluate the model
    print("Evaluating model...")
    metrics = detector.evaluate_model()
    
    print(f"Done! All results saved to {detector.save_path}")