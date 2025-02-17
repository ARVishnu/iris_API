from PIL import Image
import matplotlib.pyplot as plt
from app import segmenter  # Import the global segmenter instance
import matplotlib
import numpy as np
import base64
from io import BytesIO
matplotlib.use('Agg')  # Use Agg backend instead of Qt5Agg

def create_overlay_image(image_path):
    """
    Creates an overlay image with the segmentation mask on the original image.
    Returns the overlay image as a PIL Image object.
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Get segmentation mask using the global segmenter instance
    mask = segmenter.segment_image(image_path)
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    
    # Create overlay
    plt.imshow(original_image)
    plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    
    # Convert plot to image
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    
    # Convert to RGB
    buf = buf[:,:,:3]
    
    # Convert to PIL Image
    plot_image = Image.fromarray(buf)
    
    plt.close(fig)
    
    return plot_image

def display_segmentation(image_path, return_json=False):
    """
    Displays the segmentation result.
    If return_json is True, returns a dictionary with base64 encoded image.
    Otherwise returns the PIL Image object.
    """
    result_image = create_overlay_image(image_path)
    
    if return_json:
        # Convert PIL image to base64 string
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create response dictionary
        response = {
            "image_data": img_str,
            "format": "PNG",
            "encoding": "base64",
            "message": "Successfully processed image"
        }
        return response
    
    return result_image

# Example usage:
# if __name__ == "__main__":
#     result = display_segmentation('Vishnu_img.jpg', return_json=True)
#     print(result['image_data'][:100])  # Print first 100 chars of base64 string 