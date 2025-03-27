"""SVM classifier for inpainting detection."""
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from src.visualize import plot_mask_type_comparison, plot_model_comparison, plot_confusion_matrix
from tensorflow.keras.models import Model
from tqdm import tqdm
import time

class SVMClassifier:
    """SVM classifier using features extracted from a deep learning model."""
    
    def __init__(
        self, 
        model: Model,
        layer_name: str,
        save_path: Path,
        random_state: int = 42
    ):
        self.model = model
        self.layer_name = layer_name
        self.save_path = save_path
        self.random_state = random_state
        self.feature_extractor = None
        self.scaler = StandardScaler()
        self.svm_models = {}
        self.best_model = None
        self.best_kernel = None
        self.best_f1 = 0.0
        
        # Create SVM directory
        self.svm_dir = self.save_path / 'svm'
        os.makedirs(self.svm_dir, exist_ok=True)
        
        # Create feature extractor
        self._create_feature_extractor()
    
    def _create_feature_extractor(self) -> None:
        """Create a feature extractor from the given model and layer name."""
        try:
            self.feature_extractor = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(self.layer_name).output
            )
            print(f"Feature extractor created from layer: {self.layer_name}")
            print(f"Feature shape: {self.feature_extractor.output_shape}")
        except Exception as e:
            print(f"Error creating feature extractor: {str(e)}")
            # List available layers
            print("Available layers:")
            for i, layer in enumerate(self.model.layers):
                print(f"{i}: {layer.name}, Output Shape: {layer.output_shape}")
            raise
    
    def extract_features(
        self, 
        image_paths: List[str], 
        preprocess_fn: Any,
        batch_size: int = 32
    ) -> np.ndarray:
        features = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            # Preprocess images
            batch_images = np.array([
                preprocess_fn(img_path) for img_path in batch_paths
            ])
            # Extract features
            batch_features = self.feature_extractor.predict(batch_images, verbose=0)
            # Flatten features if needed
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
            
            features.append(batch_features)
        
        # Concatenate all batches
        features = np.vstack(features)
        print(f"Extracted features with shape: {features.shape}")
        return features
    
    def train_multiple_svms(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        kernels: List[str] = ['linear', 'poly', 'rbf', 'sigmoid'],
        C_values: List[float] = [0.1, 1.0, 10.0, 100.0],
    ) -> Dict:
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Save scaler
        joblib.dump(self.scaler, self.svm_dir / 'scaler.pkl')
        
        # Create parameter grid
        param_grid = {
            'kernel': kernels,
            'C': C_values,
            'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            'degree': [2, 3, 4, 5, 6]
        }
        
        start_time = time.time()
        
        # Create and fit HalvingGridSearchCV
        print("Starting hyperparameter search with HalvingGridSearchCV...")
        svm_search = HalvingGridSearchCV(
            SVC(probability=True, random_state=self.random_state),
            param_grid,
            cv=3,
            factor=3,
            resource='n_samples',
            n_jobs=-1,
            verbose=2,
            scoring='f1'  # Optimize for F1 score
        )
        
        # Fit search
        svm_search.fit(X_train_scaled, y_train)
        
        # Get best model and parameters
        self.best_model = svm_search.best_estimator_
        self.best_kernel = self.best_model.kernel
        self.best_C = self.best_model.C
        
        total_time = time.time() - start_time
        
        # Save only the best model
        joblib.dump(self.best_model, self.svm_dir / 'best_svm_model.pkl')
        
        # Prepare results for all tested configurations
        results_list = []
        for params, mean_score, rank in zip(
            svm_search.cv_results_['params'],
            svm_search.cv_results_['mean_test_score'],
            svm_search.cv_results_['rank_test_score']
        ):
            result = {
                **params,
                'mean_f1': mean_score,
                'rank': rank
            }
            results_list.append(result)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(self.svm_dir / 'hyperparameter_search_results.csv', index=False)
        
        print(f"\nBest SVM model: {svm_search.best_params_}")
        print(f"Total search time: {total_time:.2f}s")
        
        # Return basic info about the best model
        return {
            "best_params": svm_search.best_params_,
            "search_time": total_time
        }
    
    def evaluate_best_model(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        mask_types: Optional[List[str]] = None
    ) -> Dict:

        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # AUC
        if hasattr(self.best_model, "predict_proba"):
            y_pred_prob = self.best_model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
        else:
            y_pred_prob = y_pred
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=['Inpainted', 'Original'])
        plot_confusion_matrix(cm, self.svm_dir)
        
        # Compute metrics by mask type if provided
        mask_type_metrics = {}
        if mask_types is not None:
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
                    # Handle case with only one class
                    mask_metrics = {
                        'count': len(indices),
                        'accuracy': accuracy_score(mask_y_test, mask_y_pred),
                        'precision': 'N/A' if all(mask_y_test == 0) else 1.0,
                        'recall': 'N/A' if all(mask_y_test == 1) else 1.0,
                        'f1_score': 'N/A'
                    }
                else:
                    # Normal case with both classes
                    mask_metrics = {
                        'count': len(indices),
                        'accuracy': accuracy_score(mask_y_test, mask_y_pred),
                        'precision': precision_score(mask_y_test, mask_y_pred),
                        'recall': recall_score(mask_y_test, mask_y_pred),
                        'f1_score': f1_score(mask_y_test, mask_y_pred)
                    }
                
                mask_type_metrics[mask_type] = mask_metrics
        
        # Compile results
        metrics = {
            'model_type': 'svm',
            'kernel': self.best_kernel,
            'C': self.best_C,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
        }
        
        if mask_types is not None:
            metrics['mask_type_metrics'] = mask_type_metrics
        
        # Save results
        with open(self.svm_dir / 'evaluation_metrics.json', 'w') as f:
            import json
            json.dump(metrics, f, indent=4)
        
        with open(self.svm_dir / 'classification_report.txt', 'w') as f:
            f.write(class_report)
        
        # Create comparison plot showing SVM vs CNN performance if possible
        try:
            cnn_metrics_path = self.save_path / 'evaluation_metrics.json'
            if cnn_metrics_path.exists():
                with open(cnn_metrics_path, 'r') as f:
                    cnn_metrics = json.load(f)
                
                # Create comparison plots
                plot_model_comparison(
                    cnn_metrics=cnn_metrics,
                    svm_metrics=metrics,
                    save_path=self.svm_dir,
                    svm_kernel=self.best_kernel
                )
        except Exception as e:
            print(f"Error creating comparison plots: {str(e)}")
        
        print(f"SVM Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        if mask_types is not None:
            plot_mask_type_comparison(mask_type_metrics, self.svm_dir)
        
        return metrics