from PIL import Image
import matplotlib.pyplot as plt
from app import segment_image
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use Agg backend instead of Qt5Agg

def create_overlay_image(image_path):
    """
    Creates an overlay image with the segmentation mask on the original image.
    Returns the overlay image as a PIL Image object.
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Get segmentation mask
    mask = segment_image(image_path)
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    
    # Create overlay
    plt.imshow(original_image)
    plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    
    # Convert plot to image
    fig.canvas.draw()
    
    # Get the RGBA buffer from the SSfigure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    
    # Convert to RGB
    buf = buf[:,:,:3]
    
    # Convert to PIL Image
    plot_image = Image.fromarray(buf)
    
    plt.close(fig)
    
    return plot_image

def display_segmentation(image_path):
    """
    Displays the segmentation result.
    """
    result = create_overlay_image(image_path)
    return result

# Example usage:
# if __name__ == "__main__":
#     display_segmentation('Vishnu_img.jpg') 