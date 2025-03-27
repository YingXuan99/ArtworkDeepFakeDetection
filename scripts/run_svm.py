"""Train an SVM classifier on features extracted from a trained model."""
import sys
import os
from pathlib import Path

# Set up path
FILEPATH = Path(__file__).parent.parent
sys.path.append(f'{FILEPATH}')

# Import the InpaintingDetector class
from src.inpainting_detector import InpaintingDetector

# Set GPU device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    MODEL_PATH =  Path(f'{FILEPATH}/experiments/24032025-1547_cnn/best_model_weights.h5') # Path to the trained model
    FEATURE_LAYER = 'global_average_pooling2d'  # Name of the layer to extract features from (None for default)
    DATA_DIR = "dataset"  # Path to the dataset directory
    IMAGE_SIZE = 256  # Image size for model input
    BATCH_SIZE = 16  # Batch size for feature extraction
    
    # Create detector
    detector = InpaintingDetector(
        config_name='cnn_svm',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        model_type="cnn"  # This doesn't matter for feature extraction
    )
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}")
    detector.load_model(MODEL_PATH)
    
    # Make sure train_test_split is prepared
    print("Preparing dataset split...")
    detector.prepare_train_test_split()
    
    # Train SVM
    print("Training SVM classifier...")
    detector.train_svm(feature_layer_name=FEATURE_LAYER)
    
    print(f"Done! SVM results saved to {detector.save_path}/svm/")