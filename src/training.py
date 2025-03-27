"""Training utilities for inpainting detection."""
from typing import List, Any, Tuple, Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, Callback)
from pathlib import Path
from src.unfreeze_layers import UnfreezeCallback

def create_data_generators(
    train_paths: List[str],
    train_labels: np.ndarray,
    test_paths: List[str],
    test_labels: np.ndarray,
    batch_size: int,
    image_size: int,
    preprocess_fn: Callable
) -> Tuple[Any, Any]:
    """Create data generators for training and validation.
    
    Args:
        train_paths: List of training image paths
        train_labels: Array of training labels
        test_paths: List of test image paths
        test_labels: Array of test labels
        batch_size: Batch size for training
        image_size: Target image size
        preprocess_fn: Function to preprocess images
        
    Returns:
        Tuple of (train_generator, test_generator)
    """
    # Define data augmentation for training
    train_gen = ImageDataGenerator(
        rotation_range=5,    
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        shear_range=0.0,
        fill_mode='constant',
        cval=0,
    )
    
    # No augmentation for validation
    test_gen = ImageDataGenerator()
    
    # Create generators
    train_generator = custom_generator(
        train_paths, 
        train_labels, 
        batch_size, 
        train_gen, 
        image_size,
        preprocess_fn,
        is_training=True
    )
    
    test_generator = custom_generator(
        test_paths, 
        test_labels, 
        batch_size, 
        test_gen,
        image_size,
        preprocess_fn,
        is_training=False
    )
    
    return train_generator, test_generator

def custom_generator(
    features: List[str], 
    labels: np.ndarray, 
    batch_size: int, 
    gen: ImageDataGenerator,
    image_size: int,
    preprocess_fn: Callable,
    is_training: bool = False
):
    """Create a custom generator for training or validation.
    
    Args:
        features: List of image paths
        labels: Array of labels
        batch_size: Batch size
        gen: ImageDataGenerator to use
        image_size: Target image size
        preprocess_fn: Function to preprocess images
        is_training: Whether this is for training (apply augmentation)
        
    Yields:
        Batches of (features, labels)
    """
    while True:
        batch_indices = np.random.choice(len(features), batch_size)
        batch_features = np.array([
            preprocess_fn(
                features[i], 
                target_size=(image_size, image_size)
            ) 
            for i in batch_indices
        ])
        batch_labels = labels[batch_indices]
        
        # Apply data augmentation only for training generator
        if is_training:
            batch_features, batch_labels = next(gen.flow(batch_features, batch_labels, batch_size=batch_size))
        
        yield batch_features, batch_labels

def create_callbacks(
    save_path: Path,
    learning_rate: float,
    model=None,
    base_model=None,
    model_type: str = "cnn",
    unfreeze: bool = False,
    unfreeze_epoch: int = 5,
    unfreeze_block: List[str] = None
) -> List[Callback]:
    """Create callbacks for model training.
    
    Args:
        save_path: Path to save training artifacts
        learning_rate: Initial learning rate
        model: The model being trained (needed for unfreeze callback)
        base_model: Base model (needed for unfreeze callback) 
        model_type: Type of model ("cnn", "resnet", etc.)
        unfreeze: Whether to unfreeze layers during training
        unfreeze_epoch: Epoch at which to unfreeze layers
        unfreeze_block: List of block names to unfreeze
        
    Returns:
        List of callbacks for training
    """
    if unfreeze_block is None:
        unfreeze_block = ['block5', 'block6', 'block7']
        
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

    # Learning rate schedule
    lr_schedule = tf.keras.experimental.CosineDecayRestarts(
        learning_rate,
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
        save_path / 'training_history.csv'
    )
    
    callbacks = [
        early_stopping,
        lr_scheduler,
        csv_logger
    ]

    # Add unfreeze callback if needed
    if model_type != 'cnn' and unfreeze and model is not None and base_model is not None:
        unfreeze_callback = UnfreezeCallback(
            model=model,
            base_model=base_model,
            unfreeze_epoch=unfreeze_epoch,
            unfreeze_block=unfreeze_block,
            new_lr=learning_rate * 0.1
        )
        callbacks.append(unfreeze_callback)

    return callbacks