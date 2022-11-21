import numpy as np
from PIL import Image
import cv2
import torch

class MotionBlur(object):
    """Crop randomly the image in a sample.

    Args:
        p (float): probability of the image getting a motion blur.
            Default value is 0.5 Should be float value between [0, 1].
        size (int): Desired motion blur size of the crop. If size is an
            int instead of sequence like (min, max), a constant size is
            applied.
    """
    def __init__(self, p=0.15, size=(5, 13)):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
        self.p = p
  
    def __call__(self, image):
        if torch.rand(1) <= self.p:
            # Convert from PIL image to Numpy's
            image = np.asarray(image)
            image_dtype = image.dtype
            image = image.astype('float32')

            # Random size
            size = torch.randint(self.size[0], self.size[1], (1,)).item()
            # Random angle
            angle = torch.randint(1, 359, (1,))
            
            # Apply motion blur
            image = self.__apply_motion_blur(image, size=size)
            
            # Convert back to PIL image from Numpy's
            image = image.astype(image_dtype)
            image = Image.fromarray(image)
            
        return image

    def __apply_motion_blur(self, image, size=5, angle=90):
        """ 
        Source: https://stackoverflow.com/a/57629531/10281627
        size - in pixels, size of motion blur
        angel - in degrees, direction of motion blur
        """
        k = np.zeros((size, size), dtype=np.float32)
        k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
        k = k * ( 1.0 / np.sum(k) )
        return cv2.filter2D(image, -1, k)
    
    def __repr__(self):
        return self.__class__.__name__+'()'