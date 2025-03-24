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
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
)
from model_architecture import build_cnn_model, build_resnet_model, build_efficientnetv2_model
from unfreeze_layers import UnfreezeCallback
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from tqdm import tqdm

# Constants
FILEPATH = Path(__file__).parent
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
        self.original_dir = Path(f'{FILEPATH}/mask_images/original')
        self.inpainting_dir = Path(f'{FILEPATH}/mask_images/inpainting')
        
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
    
    def determine_mask_type(self, mask_array: np.ndarray) -> str:
        """Determine the type of mask applied to an image."""
        height, width = mask_array.shape
        binary_mask = (mask_array == 255).astype(float)
        
        # Calculate total white pixel percentage
        total_percentage = np.mean(binary_mask)
        
        # Quick reject for very sparse or dense masks
        if total_percentage < 0.1 or total_percentage > 0.9:
            return 'random'
        
        # Define regions
        regions = {
            'top': binary_mask[:height//2, :],
            'bottom': binary_mask[height//2:, :],
            'left': binary_mask[:, :width//2],
            'right': binary_mask[:, width//2:]
        }
        
        # Calculate percentages for each region
        percentages = {name: np.mean(region) for name, region in regions.items()}
        
        # Calculate diagonal regions
        y, x = np.ogrid[:height, :width]
        diag_regions = {
            'diagonal_top_left': binary_mask[y < (-height/width * (x - width))],
            'diagonal_bottom_right': binary_mask[y > (-height/width * (x - width))],
            'diagonal_top_right': binary_mask[y < (height/width * x)],
            'diagonal_bottom_left': binary_mask[y > (height/width * x)]
        }
        
        for name, region in diag_regions.items():
            percentages[name] = np.mean(region)
        
        # Define threshold criteria
        thresholds = {'main': 0.5, 'secondary': 0.2, 'diagonal': 0.45}
        
        # Check standard directions
        directions = [
            ('top', 'bottom'),
            ('bottom', 'top'),
            ('left', 'right'),
            ('right', 'left')
        ]
        
        for main, opposite in directions:
            if (percentages[main] > thresholds['main'] and 
                percentages[opposite] < thresholds['secondary']):
                return main
        
        # Check diagonal directions
        diagonals = [
            ('diagonal_top_left', 'diagonal_bottom_right'),
            ('diagonal_top_right', 'diagonal_bottom_left'),
            ('diagonal_bottom_left', 'diagonal_top_right'),
            ('diagonal_bottom_right', 'diagonal_top_left')
        ]
        
        for main, opposite in diagonals:
            if (percentages[main] > thresholds['diagonal'] and 
                percentages[opposite] < thresholds['secondary']):
                return main
        
        return 'random'
    
    def apply_mask(self, image_path: str, mask_path: str) -> Tuple[Image.Image, str]:

        image = Image.open(image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L')
        
        # Convert images to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Determine mask type
        mask_type = self.determine_mask_type(mask_array)
        
        # Create an all transparent image
        result = np.zeros_like(image_array)
        
        # Copy only the pixels where the mask is white (255)
        result[mask_array == 255] = image_array[mask_array == 255]
        
        return Image.fromarray(result), mask_type
    
    def process_folder(self, folder_path: str) -> Tuple[bool, Optional[str]]:

        folder_name = os.path.basename(folder_path)
        original_file = os.path.join(folder_path, 'original.png')
        inpainting_file = os.path.join(folder_path, 'inpainting.png')
        mask_file = os.path.join(folder_path, 'mask.png')
        
        if all(os.path.exists(f) for f in [original_file, inpainting_file, mask_file]):
            try:
                # Apply mask and get mask type
                masked_original, mask_type = self.apply_mask(original_file, mask_file)
                masked_inpainting, _ = self.apply_mask(inpainting_file, mask_file)
                
                # Save with mask type and directory identifier
                original_filename = f'{mask_type}_original_{folder_name}.png'
                inpainting_filename = f'{mask_type}_inpainting_{folder_name}.png'
                
                masked_original.save(os.path.join(self.original_dir, original_filename))
                masked_inpainting.save(os.path.join(self.inpainting_dir, inpainting_filename))
                
                return True, mask_type
            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                return False, None
        return False, None
    
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

        # Load real image paths (label = 1)
        real_image_paths, real_labels = self._load_from_folder(self.original_dir, label=1)
        
        # Load fake image paths (label = 0)
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
        image_paths = []
        labels = []
        
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
    
    def create_data_generators(self) -> Tuple[Any, Any]:

        # Define data augmentation for training
        train_gen = ImageDataGenerator(
            rotation_range=5,    
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip = True,
            zoom_range=0.1,
            shear_range=0.0,
            fill_mode='constant',
            cval=0,
        )
        
        # No augmentation for validation
        test_gen = ImageDataGenerator()
        
        # Create generators
        train_generator = self.custom_generator(
            self.train_paths, 
            self.train_labels, 
            self.batch_size, 
            train_gen, 
            is_training=True
        )
        
        test_generator = self.custom_generator(
            self.test_paths, 
            self.test_labels, 
            self.batch_size, 
            test_gen, 
            is_training=False
        )
        
        return train_generator, test_generator
    
    def custom_generator(
        self, 
        features: List[str], 
        labels: np.ndarray, 
        batch_size: int, 
        gen: ImageDataGenerator, 
        is_training: bool = False
    ):
        while True:
            batch_indices = np.random.choice(len(features), batch_size)
            batch_features = np.array([
                self.load_and_preprocess_image(
                    features[i], 
                    target_size=(self.image_size, self.image_size)
                ) 
                for i in batch_indices
            ])
            batch_labels = labels[batch_indices]
            
            # Apply data augmentation only for training generator
            if is_training:
                batch_features, batch_labels = next(gen.flow(batch_features, batch_labels, batch_size=batch_size))
            
            yield batch_features, batch_labels
    
    def create_callbacks(self) -> List[Callback]:

        # Define the file path for saving the model weights
        checkpoint_filepath = os.path.join(
            self.weights_dir, 
            "model_weights_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"
        )
        
        # Create a ModelCheckpoint callback to save the best model weights
        # model_checkpoint_callback = ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_best_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     verbose=1
        # )
        
        # Create an EarlyStopping callback to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create a ReduceLROnPlateau callback to reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=5,  # Steps per restart
            t_mul=2.0,  # Each restart period gets longer
            m_mul=0.9,  # Each restart peak is lower
            alpha=0.1   # Minimum learning rate factor
        )
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_schedule(epoch), 
            verbose=1
        )

        # Create a CSV logger to save training history
        csv_logger = tf.keras.callbacks.CSVLogger(
            self.save_path / 'training_history.csv'
        )
        
        callbacks =  [
            # model_checkpoint_callback,
            early_stopping,
            lr_scheduler,
            csv_logger
        ]

        if self.model_type != 'cnn' and self.unfreeze:
            unfreeze_callback = UnfreezeCallback(
            model=self.model,
            base_model=self.base_model,
            unfreeze_epoch=self.unfreeze_epoch,
            unfreeze_block = self.unfreeze_block,
            new_lr=self.learning_rate * 0.1
        )
        callbacks.append(unfreeze_callback)

        return callbacks
    
    def train_model(self) -> Dict:
        
        self.prepare_train_test_split()
        # Build the model
        self.model, self.base_model = self.build_model()
        
        # Create data generators
        train_generator, test_generator = self.create_data_generators()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
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
        
    
    def load_model(self, model_path: str) -> None:

        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def plot_training_history(self) -> None:

        # Load training history from CSV
        history_df = pd.read_csv(self.save_path / 'training_history.csv')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot training & validation loss
        ax1.plot(history_df['epoch'], history_df['loss'], label='Train Loss')
        ax1.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Plot training & validation accuracy
        ax2.plot(history_df['epoch'], history_df['accuracy'], label='Train Accuracy')
        ax2.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.fig_path / 'training_history.png')
        plt.close()
        
        # Plot AUC curve if available
        if 'auc' in history_df.columns and 'val_auc' in history_df.columns:
            plt.figure(figsize=(10, 8))
            plt.plot(history_df['epoch'], history_df['auc'], label='Train AUC')
            plt.plot(history_df['epoch'], history_df['val_auc'], label='Validation AUC')
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(self.fig_path / 'auc_history.png')
            plt.close()
        
        print(f"Training history plots saved to {self.fig_path}")
    
    def evaluate_model(self) -> Dict:
        """Evaluate the model on the test set with breakdown by mask type."""
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
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Inpainted', 'Original'], rotation=45)
        plt.yticks(tick_marks, ['Inpainted', 'Original'])
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.fig_path / 'confusion_matrix.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.fig_path / 'roc_curve.png')
        plt.close()
        
        # Print summary of results
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{class_report}")
        
        # Create mask type performance comparison plot
        self._plot_mask_type_comparison(mask_type_metrics)
        
        print("\nPerformance by Mask Type:")
        for mask_type, metrics_dict in mask_type_metrics.items():
            print(f"  {mask_type} (count: {metrics_dict['count']}):")
            print(f"    Accuracy: {metrics_dict['accuracy']:.4f}")
            print(f"    Precision: {metrics_dict['precision'] if isinstance(metrics_dict['precision'], str) else metrics_dict['precision']:.4f}")
            print(f"    Recall: {metrics_dict['recall'] if isinstance(metrics_dict['recall'], str) else metrics_dict['recall']:.4f}")
            print(f"    F1 Score: {metrics_dict['f1_score'] if isinstance(metrics_dict['f1_score'], str) else metrics_dict['f1_score']:.4f}")
        
        print(f"\nClassification Report:\n{class_report}")
        
        return metrics

    def _plot_mask_type_comparison(self, mask_type_metrics):
        """Plot performance comparison across different mask types."""
        mask_types = list(mask_type_metrics.keys())
        
        # Extract metrics for plotting
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        counts = []
        
        for mask_type in mask_types:
            metrics = mask_type_metrics[mask_type]
            accuracies.append(metrics['accuracy'])
            
            # Handle potential string values for precision, recall, f1
            if isinstance(metrics['precision'], str):
                precisions.append(0)  # Use 0 as placeholder for 'N/A'
            else:
                precisions.append(metrics['precision'])
                
            if isinstance(metrics['recall'], str):
                recalls.append(0)
            else:
                recalls.append(metrics['recall'])
                
            if isinstance(metrics['f1_score'], str):
                f1_scores.append(0)
            else:
                f1_scores.append(metrics['f1_score'])
                
            counts.append(metrics['count'])
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(mask_types))
        width = 0.2
        
        # Create bars
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy')
        ax.bar(x - 0.5*width, precisions, width, label='Precision')
        ax.bar(x + 0.5*width, recalls, width, label='Recall')
        ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score')
        
        # Add labels and title
        ax.set_xlabel('Mask Type')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Mask Type')
        ax.set_xticks(x)
        ax.set_xticklabels(mask_types, rotation=45, ha='right')
        ax.legend()
        
        # Add sample counts as text above the bars
        for i, count in enumerate(counts):
            ax.annotate(f'n={count}', 
                    xy=(i, max(accuracies[i], precisions[i], recalls[i], f1_scores[i]) + 0.02),
                    ha='center')
        
        # Add grid lines for readability
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.fig_path / 'mask_type_performance.png')
        plt.close()
        
        # Create a second plot showing sample distribution
        plt.figure(figsize=(10, 6))
        plt.bar(mask_types, counts, color='skyblue')
        plt.xlabel('Mask Type')
        plt.ylabel('Number of Samples')
        plt.title('Test Set Distribution by Mask Type')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(self.fig_path / 'mask_type_distribution.png')
        plt.close()
