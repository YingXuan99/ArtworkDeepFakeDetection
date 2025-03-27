"""Visualization utilities for inpainting detection results."""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def plot_mask_type_comparison(mask_type_metrics: Dict, fig_path: Path) -> None:

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
    plt.savefig(fig_path / 'mask_type_performance.png')
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
    plt.savefig(fig_path / 'mask_type_distribution.png')
    plt.close()

def plot_training_history(history_path: Path, fig_path: Path) -> None:

    # Load training history from CSV
    history_df = pd.read_csv(history_path)
    
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
    plt.savefig(fig_path / 'training_history.png')
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
        plt.savefig(fig_path / 'auc_history.png')
        plt.close()
    
    print(f"Training history plots saved to {fig_path}")

def plot_confusion_matrix(cm: np.ndarray, fig_path: Path, title: str = 'Confusion Matrix') -> None:

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
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
    plt.savefig(fig_path / 'confusion_matrix.png')
    plt.close()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, fig_path: Path) -> None:
    """Plot ROC curve.
    
    Args:
        fpr: False positive rate array
        tpr: True positive rate array
        roc_auc: Area under the ROC curve
        fig_path: Directory to save the generated figure
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(fig_path / 'roc_curve.png')
    plt.close()

def plot_sample_images(original_images: List[np.ndarray], inpainted_images: List[np.ndarray], 
                    fig_path: Path, num_samples: int = 5) -> None:

    num_samples = min(num_samples, len(original_images), len(inpainted_images))
    
    # Create a figure with subplots for each sample pair
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    
    for i in range(num_samples):
        # Plot original image
        if num_samples > 1:
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
            
        ax1.imshow(original_images[i])
        ax1.set_title(f'Original Image {i+1}')
        ax1.axis('off')
        
        # Plot inpainted image
        ax2.imshow(inpainted_images[i])
        ax2.set_title(f'Inpainted Image {i+1}')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(fig_path / 'sample_image_comparison.png')
    plt.close()

def plot_model_comparison(cnn_metrics: Dict, svm_metrics: Dict, save_path: Path, 
                        svm_kernel: Optional[str] = None) -> None:
    try:
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Metrics to compare
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        cnn_values = [cnn_metrics[m] for m in metrics]
        svm_values = [svm_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        kernel_label = f' ({svm_kernel})' if svm_kernel else ''
        ax.bar(x - width/2, cnn_values, width, label='CNN')
        ax.bar(x + width/2, svm_values, width, label=f'SVM{kernel_label}')
        
        ax.set_ylabel('Score')
        ax.set_title('CNN vs SVM Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / 'cnn_vs_svm_comparison.png')
        plt.close()
        
        # Also try to compare by mask type if available
        if ('mask_type_metrics' in cnn_metrics and 
            'mask_type_metrics' in svm_metrics):
            
            cnn_mask_metrics = cnn_metrics['mask_type_metrics']
            svm_mask_metrics = svm_metrics['mask_type_metrics']
            
            common_mask_types = set(cnn_mask_metrics.keys()) & set(svm_mask_metrics.keys())
            
            for metric in ['accuracy', 'f1_score']:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                cnn_values = []
                svm_values = []
                mask_types = []
                
                for mask_type in common_mask_types:
                    cnn_value = cnn_mask_metrics[mask_type][metric]
                    svm_value = svm_mask_metrics[mask_type][metric]
                    
                    # Skip if either value is not numeric
                    if isinstance(cnn_value, str) or isinstance(svm_value, str):
                        continue
                    
                    cnn_values.append(cnn_value)
                    svm_values.append(svm_value)
                    mask_types.append(mask_type)
                
                x = np.arange(len(mask_types))
                width = 0.35
                
                ax.bar(x - width/2, cnn_values, width, label='CNN')
                ax.bar(x + width/2, svm_values, width, label=f'SVM{kernel_label}')
                
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'CNN vs SVM {metric.capitalize()} by Mask Type')
                ax.set_xticks(x)
                ax.set_xticklabels(mask_types, rotation=45)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(save_path / f'cnn_vs_svm_{metric}_by_mask_type.png')
                plt.close()
    
    except Exception as e:
        print(f"Could not create comparison plots: {str(e)}")