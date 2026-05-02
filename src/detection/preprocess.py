


import numpy as np
import cv2

class Preprocess:

    def letter_box(self,
                   img: np.ndarray,
                   target_size: tuple = (160, 160),
                   color: tuple=(0, 0, 0),
                   ) -> np.ndarray:
        """
        Resize and pad image while maintaining aspect ratio.
        
        Args:
            img: Input image (numpy array)
            target_size: Target size (width, height) tuple
            color: Padding color (BGR tuple)
        
        Returns:
            letterboxed_img: Padded and resized image
        """
        h, w = img.shape[:2]
        
        scale = min(target_size[0] / h, target_size[1] / w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = (target_size[0] - new_h) // 2
        pad_w = (target_size[1] - new_w) // 2
        
        # Create padded image
        letterboxed_img = cv2.copyMakeBorder(
            resized_img, 
            pad_h, 
            target_size[0] - new_h - pad_h, 
            pad_w, 
            target_size[1] - new_w - pad_w, 
            cv2.BORDER_CONSTANT, 
            value=color
        )
        
        return letterboxed_img