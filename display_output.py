from PIL import Image
import matplotlib.pyplot as plt
from app import segment_image
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of Qt5Agg

def create_overlay_image(image_path):
    """
    Creates an overlay image with the segmentation mask.
    Returns the overlay image as a PIL Image object.
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Get segmentation mask
    mask = segment_image(image_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.title('Pupil Segmentation Overlay')
    plt.axis('off')
    
    # Convert plot to image
    fig.canvas.draw()
    plot_image = Image.frombytes('RGB', 
                                fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    plt.close(fig)
    
    return plot_image

def display_segmentation(image_path):
    """
    Displays the segmentation result.
    """
    result = create_overlay_image(image_path)
    result.show()

# # Example usage:
# display_segmentation('Vishnu_img.jpg') 