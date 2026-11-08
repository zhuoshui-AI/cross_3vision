from transformers import BitImageProcessor as BaseProcessor
import numpy as np

from transformers import AutoImageProcessor
from transformers.image_utils import PILImageResampling

class CustomDinov2Processor(BaseProcessor):
    model_type = "dinov2"

    def preprocess(self, images, **kwargs):
        # 1. Handle "Already Normalized" (0-1) check
        test_img = images[0] if isinstance(images, list) else images
        
        # Determine max value to see if we should rescale
        if hasattr(test_img, "getextrema"): # PIL
            extrema = test_img.getextrema()
            max_val = max([b[1] for b in extrema]) if isinstance(extrema[0], tuple) else extrema[1]
        elif hasattr(test_img, "max"): # Numpy/Tensor
            max_val = test_img.max()
        else:
            max_val = 255 

        # If already 0-1, disable rescaling (1/255)
        if max_val <= 1.0:
            kwargs["do_rescale"] = False
        else:
            kwargs["do_rescale"] = True
            
        # 2. Force RGB Conversion (handles grayscale)
        kwargs["do_convert_rgb"] = True
            
        return super().preprocess(images, **kwargs)

    def resize(self, image: np.ndarray, size=None, resample=PILImageResampling.BILINEAR, **kwargs) -> np.ndarray:
        # 3. Your dynamic "nearest 14" logic
        h, w = image.shape[:2]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        
        return super().resize(
            image, 
            size={"height": new_h, "width": new_w}, 
            resample=resample, 
            **kwargs
        )

# Register the class so it saves to the Hub correctly
CustomDinov2Processor.register_for_auto_class("AutoImageProcessor")
