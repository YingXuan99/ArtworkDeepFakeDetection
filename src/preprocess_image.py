"""Image preprocessing utilities for inpainting detection."""
import os
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class PreprocessImage:
    """Class for preprocessing images for inpainting detection."""
    
    @staticmethod
    def determine_mask_type(mask_array: np.ndarray) -> str:
        """Determine the type of mask applied to an image.
        
        Args:
            mask_array: NumPy array of the mask image
            
        Returns:
            str: Mask type (top, bottom, left, right, diagonal_top_left, etc.)
        """
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
    
    @staticmethod
    def apply_mask(image_path: str, mask_path: str) -> Tuple[Image.Image, str]:
        """Apply mask to an image and determine the mask type.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file
            
        Returns:
            Tuple containing the masked image and the mask type
        """
        image = Image.open(image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L')
        
        # Convert images to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Determine mask type
        mask_type = PreprocessImage.determine_mask_type(mask_array)
        
        # Create an all transparent image
        result = np.zeros_like(image_array)
        
        # Copy only the pixels where the mask is white (255)
        result[mask_array == 255] = image_array[mask_array == 255]
        
        return Image.fromarray(result), mask_type
    
    @staticmethod
    def load_and_preprocess_image(
        image_path: str, 
        target_size: Tuple[int, int] = (256, 256), 
        preserve_texture: bool = True
    ) -> np.ndarray:
        """Load and preprocess an image for model input.
        
        Args:
            image_path: Path to the image file
            target_size: Target dimensions (height, width)
            preserve_texture: Whether to preserve texture using Lanczos resampling
            
        Returns:
            Preprocessed image as numpy array
        """
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
    
    @staticmethod
    def process_folder(folder_path: str, original_dir: str, inpainting_dir: str) -> Tuple[bool, Optional[str]]:
        """Process a folder containing original, inpainting, and mask images.
        
        Args:
            folder_path: Path to the folder containing image files
            original_dir: Directory to save original masked images
            inpainting_dir: Directory to save inpainting masked images
            
        Returns:
            Tuple indicating success (bool) and mask type (str)
        """
        folder_name = os.path.basename(folder_path)
        original_file = os.path.join(folder_path, 'original.png')
        inpainting_file = os.path.join(folder_path, 'inpainting.png')
        mask_file = os.path.join(folder_path, 'mask.png')
        
        if all(os.path.exists(f) for f in [original_file, inpainting_file, mask_file]):
            try:
                # Apply mask and get mask type
                masked_original, mask_type = PreprocessImage.apply_mask(original_file, mask_file)
                masked_inpainting, _ = PreprocessImage.apply_mask(inpainting_file, mask_file)
                
                # Save with mask type and directory identifier
                original_filename = f'{mask_type}_original_{folder_name}.png'
                inpainting_filename = f'{mask_type}_inpainting_{folder_name}.png'
                
                masked_original.save(os.path.join(original_dir, original_filename))
                masked_inpainting.save(os.path.join(inpainting_dir, inpainting_filename))
                
                return True, mask_type
            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                return False, None
        return False, None
    
    @staticmethod
    def process_folders_with_threading(folders: List[str], original_dir: str, inpainting_dir: str, max_workers: int = None):
        """Process multiple folders with threading for performance.
        
        Args:
            folders: List of folder paths to process
            original_dir: Directory to save original masked images
            inpainting_dir: Directory to save inpainting masked images
            max_workers: Maximum number of worker threads
            
        Returns:
            List of results (success, mask_type) for each folder
        """
        # Process directory with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_folder = {
                executor.submit(
                    PreprocessImage.process_folder, 
                    folder, 
                    original_dir, 
                    inpainting_dir
                ): folder for folder in folders
            }
            
            for future in tqdm(
                future_to_folder, 
                total=len(folders), 
                desc="Processing images"
            ):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error in thread: {str(e)}")
                    results.append((False, None))
        
        return results
    
    @staticmethod
    def load_from_folder(folder: str, label: int) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from a folder.
        
        Args:
            folder: Path to the folder containing images
            label: Label to assign to all images in the folder
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        for filename in os.listdir(folder):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, filename)
                image_paths.append(img_path)
                labels.append(label)
        
        return image_paths, labels