import numpy as np


class NumpyImage(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    @property
    def width(self):
        return self.shape[1] if len(self.shape) > 1 else 1
    
    @property
    def height(self):
        return self.shape[0]
    
    @property
    def depth(self):
        return self.shape[2] if len(self.shape) > 2 else 1
    
    def as_array(self):
        """Convert back to regular numpy array for compatibility"""
        return np.asarray(self)