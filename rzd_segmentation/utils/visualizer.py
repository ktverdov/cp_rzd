import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt


class SegmVisualizer():
    def __init__(self, num_classes=4):
        colours = cm.get_cmap('viridis', num_classes)
        cmap = colours(np.linspace(0, 1, num_classes))
        cmap[0,-1] = 0
        cmap[1:,-1] = 0.3
        
        self.cmap = cmap
    
    def vis_mask_on_image(self, image, mask):
        image = image.copy()
        mask = mask.copy()
        
        output = self.cmap[mask.flatten()]

        R, C = mask.shape[:2]
        output = (output.reshape((R, C, -1))[:,:,:-1] * 255).astype(np.uint8)

        image[mask > 0] = image[mask > 0] * 0.5 + output[mask > 0] * 0.5

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()
    
    def vis_mistakes(self, image, mask, mask_gt):
        image = image.copy()
        
        mistakes_mask = (mask_gt != mask)
        image[mistakes_mask] = image[mistakes_mask] * 0.6 + np.array([255, 0, 0]) * 0.4

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()
