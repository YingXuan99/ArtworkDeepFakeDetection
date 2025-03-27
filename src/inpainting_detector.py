"""Inpainting Detection in Artwork."""
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import List, Any, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from src.model_architecture import build_cnn_model, build_resnet_model, build_efficientnetv2_model
from src.preprocess_image import PreprocessImage
from src.visualize import plot_confusion_matrix, plot_mask_type_comparison, plot_roc_curve, plot_training_history
from src.training import create_callbacks, create_data_generators
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from tqdm import tqdm

# Constants
FILEPATH = Path(__file__).parent.parent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

@dataclass
class InpaintingDetector:
    """Detection of inpainting in artwork images."""
    config_name: str = "cnn_baseline"
    image_size: int = 256
    batch_size: int = 16
    num_classes: int = 2
    max_epochs: int = 25
    unfreeze: bool = False
    unfreeze_epoch: int = 5
    unfreeze_block: List[str] = field(default_factory=lambda: ['block5', 'block6', 'block7'])
    learning_rate: float = 1e-4
    phase: str = field(init=False)
    save_path: Path = field(init=False)
    dataset_dir: Path = field(init=False)
    train_paths: List[str] = field(init=False)
    test_paths: List[str] = field(init=False)
    train_labels: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)
    model: Any = field(init=False)
    base_model: Any = field(init=False)
    checkpoint_path: str = field(init=False)
    model_type: str = "cnn"  # Options: "cnn", "resnet", "efficientnet"
    
    def __post_init__(self) -> None:
        """Initialize the paths and configurations."""
        self.dataset_dir = Path(f'{FILEPATH}/dataset')
        self.original_dir = Path(f'{FILEPATH}/dataset/mask_images/original')
        self.inpainting_dir = Path(f'{FILEPATH}/dataset/mask_images/inpainting')
        
        # Create timestamp for experiment directory
        current_time = datetime.now().strftime("%d%m%Y-%H%M")
        self.save_path = Path(f'{FILEPATH}/experiments/{current_time}_{self.config_name}')
        if not self.save_path.exists():
            os.makedirs(self.save_path)
            
        # Create model weights directory
        self.weights_dir = self.save_path / 'weights'
        if not self.weights_dir.exists():
            os.makedirs(self.weights_dir)
            
        # Set checkpoint path
        self.checkpoint_path = f'{self.weights_dir}/best_model.keras'
        
        # Set figure save path
        self.fig_path = self.save_path / 'figures'
        if not self.fig_path.exists():
            os.makedirs(self.fig_path)
        
        # Save configuration
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration parameters to a JSON file."""
        config = {
            "config_name": self.config_name,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "model_type": self.model_type,
            "timestamp": datetime.now().strftime("%d%m%Y-%H%M"),
        }
        
        with open(self.save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def process_folder(self, folder_path: str) -> Tuple[bool, Optional[str]]:
        return PreprocessImage.process_folder(
            folder_path, 
            self.original_dir, 
            self.inpainting_dir
        )

    def prepare_datasets(self, source_dir: str = 'inpainting') -> None:

        # Ensure destination directories exist
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.inpainting_dir, exist_ok=True)
        
        # Get folders from source directory
        folders = [os.path.join(source_dir, d) for d in os.listdir(source_dir) 
                if os.path.isdir(os.path.join(source_dir, d))]
        
        # Initialize mask type counters
        mask_counts = {
            'top': 0, 'bottom': 0, 'left': 0, 'right': 0,
            'diagonal_top_left': 0, 'diagonal_top_right': 0,
            'diagonal_bottom_left': 0, 'diagonal_bottom_right': 0,
            'random': 0
        }
        
        # Process directory with thread pool
        print("Processing images and extracting masks...")
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.process_folder, folders),
                total=len(folders)
            ))
        
        # Count results
        for success, mask_type in results:
            if success and mask_type:
                mask_counts[mask_type] += 1
        
        # Print summary
        total_processed = sum(mask_counts.values())
        print(f"\nProcessed {total_processed} total images")
        print("\nMask type distribution:")
        for mask_type, count in mask_counts.items():
            percentage = (count / total_processed * 100) if total_processed > 0 else 0
            print(f"{mask_type}: {count} images ({percentage:.1f}%)")
        
        # Save mask counts to file
        with open(self.save_path / 'mask_counts.json', 'w') as f:
            json.dump(mask_counts, f, indent=4)
    
    def prepare_train_test_split(self) -> None:

        real_image_paths, real_labels = self._load_from_folder(self.original_dir, label=1)
        fake_image_paths, fake_labels = self._load_from_folder(self.inpainting_dir, label=0)
        
        # Combine real and fake image paths and labels
        image_paths = real_image_paths + fake_image_paths
        labels = real_labels + fake_labels
        
        # Convert labels to NumPy array
        labels = np.array(labels)
        
        # Print dataset info
        print(f"Number of real images: {len(real_image_paths)}")
        print(f"Number of fake images: {len(fake_image_paths)}")
        print(f"Total number of images: {len(image_paths)}")
        
        # Perform train-test split
        self.train_paths, self.test_paths, self.train_labels, self.test_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        
        # Save dataset split info
        dataset_info = {
            "num_real_images": len(real_image_paths),
            "num_fake_images": len(fake_image_paths),
            "total_images": len(image_paths),
            "num_train_images": len(self.train_paths),
            "num_test_images": len(self.test_paths),
            "train_real_count": int(np.sum(self.train_labels == 1)),
            "train_fake_count": int(np.sum(self.train_labels == 0)),
            "test_real_count": int(np.sum(self.test_labels == 1)),  
            "test_fake_count": int(np.sum(self.test_labels == 0))   
        }
        
        with open(self.save_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)
    
    def _load_from_folder(self, folder: str, label: int) -> Tuple[List[str], List[int]]:
        image_paths, labels = [], []
        
        for filename in os.listdir(folder):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, filename)
                image_paths.append(img_path)
                labels.append(label)
        
        return image_paths, labels
    
    def load_and_preprocess_image(
        self, 
        image_path: str, 
        target_size: Tuple[int, int] = (256, 256), 
        preserve_texture: bool = True
    ) -> np.ndarray:
        
        img = Image.open(image_path)
        img = img.convert('RGB')  # Convert to RGB
        
        if preserve_texture:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(img)
            
            # Use Lanczos resampling which better preserves high-frequency details
            img_array = cv2.resize(
                img_array, 
                target_size, 
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            img = img.resize(target_size)  # Resize the image
            img_array = np.array(img)

        img_array = img_array.astype(np.float32)
        img_array /= 255.0  # Normalize to [0, 1]
        
        return img_array
    
    def build_model(self) -> Model:

        if self.model_type == "resnet":
            return build_resnet_model()
        elif self.model_type == "efficientNet":
            return build_efficientnetv2_model()
        else:  # Default to CNN
            return build_cnn_model()
    
    def train_model(self) -> Dict:
        
        self.prepare_train_test_split()
        # Build the model
        self.model, self.base_model = self.build_model()
        
        # Create data generators
        train_generator, test_generator = create_data_generators(
            train_paths=self.train_paths,
            train_labels=self.train_labels,
            test_paths=self.test_paths,
            test_labels=self.test_labels,
            batch_size=self.batch_size,
            image_size=self.image_size,
            preprocess_fn=PreprocessImage.load_and_preprocess_image
        )
        
        # Create callbacks
        callbacks = create_callbacks(
            save_path=self.save_path,
            learning_rate=self.learning_rate,
            model=self.model,
            base_model=self.base_model,
            model_type=self.model_type,
            unfreeze=self.unfreeze,
            unfreeze_epoch=self.unfreeze_epoch,
            unfreeze_block=self.unfreeze_block
        )
        
        # Train the model
        print(f"\nStarting model training with {self.model_type} architecture...")
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(self.train_paths) // self.batch_size,
            epochs=self.max_epochs,
            validation_data=test_generator,
            validation_steps=len(self.test_paths) // self.batch_size,
            callbacks=callbacks
        )
        
        # Save the best model weights
        final_weights_path = self.save_path / 'best_model_weights.h5'
        self.model.save_weights(final_weights_path)
        print(f"Best model weights saved to {final_weights_path}")

        plot_training_history(
            self.save_path / 'training_history.csv', 
            self.fig_path
        )
    
    def load_model(self, model_path: str) -> None:

        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

    def evaluate_model(self) -> Dict:

        # Prepare test data
        X_test = np.array([
            self.load_and_preprocess_image(
                img_path, 
                target_size=(self.image_size, self.image_size)
            ) 
            for img_path in tqdm(self.test_paths, desc="Preprocessing test images")
        ])
        y_test = self.test_labels
        
        # Evaluate model
        print("Evaluating model...")
        evaluation = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Extract mask types from filenames
        mask_types = []
        for path in self.test_paths:
            filename = os.path.basename(path)
            # Extract mask type from filename (assuming format like "top_original_folder.png")
            mask_type = filename.split('_')[0]
            mask_types.append(mask_type)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Compute metrics by mask type
        mask_type_metrics = {}
        unique_mask_types = set(mask_types)
        
        for mask_type in unique_mask_types:
            # Get indices for this mask type
            indices = [i for i, t in enumerate(mask_types) if t == mask_type]
            
            if len(indices) == 0:
                continue
                
            # Extract predictions and ground truth for this mask type
            mask_y_test = y_test[indices]
            mask_y_pred = y_pred[indices]
            
            # Calculate metrics
            if len(np.unique(mask_y_test)) < 2:
                # Handle case with only one class in this mask type
                mask_metrics = {
                    'count': len(indices),
                    'accuracy': accuracy_score(mask_y_test, mask_y_pred),
                    'precision': 'N/A' if all(mask_y_test == 0) else 1.0,
                    'recall': 'N/A' if all(mask_y_test == 1) else 1.0,
                    'f1_score': 'N/A'
                }
            else:
                # Normal case with both classes present
                mask_metrics = {
                    'count': len(indices),
                    'accuracy': accuracy_score(mask_y_test, mask_y_pred),
                    'precision': precision_score(mask_y_test, mask_y_pred),
                    'recall': recall_score(mask_y_test, mask_y_pred),
                    'f1_score': f1_score(mask_y_test, mask_y_pred)
                }
            
            mask_type_metrics[mask_type] = mask_metrics
        
        # Compile overall results
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(roc_auc),
            'loss': float(evaluation[0]),
            'model_accuracy': float(evaluation[1]),
            'confusion_matrix': cm.tolist(),
            'mask_type_metrics': mask_type_metrics
        }
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, target_names=['Inpainted', 'Original'])
        
        # Save results
        with open(self.save_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        with open(self.save_path / 'classification_report.txt', 'w') as f:
            f.write(class_report)
        
        plot_confusion_matrix(cm, self.fig_path)
        plot_roc_curve(fpr, tpr, roc_auc, self.fig_path)
        plot_mask_type_comparison(mask_type_metrics, self.fig_path)
        
        return metrics