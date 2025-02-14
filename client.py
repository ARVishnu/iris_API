import requests
import os
from PIL import Image
from io import BytesIO

def process_image(image_path, op_path):
    """
    Send an image to the API for processing
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        Image: The processed image
    """
    url = "http://127.0.0.1:8000/process-image/"
    
    # Ensure the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Verify file type
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Invalid file format. Only PNG and JPG files are allowed.")
    
    # Prepare the file for upload
    with open(image_path, 'rb') as image_file:
        files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
        
        try:
            # Make the request to the API
            response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()
            print(response.content)
            
            # Convert response content to image
            processed_image_path = f'output/{op_path}.jpg'
            with open(processed_image_path, 'wb') as processed_image_file:
                processed_image_file.write(response.content)
            print(f'Processed image saved to {processed_image_path}')
            
        except requests.exceptions.Timeout:
            print("Request timed out. Server took too long to respond.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error making request to API: {e}")
            return None

# Example usage
if __name__ == "__main__":
    image_path = "Vishnu_img.jpg"  # Replace with your image path
    result_image = process_image(image_path,'VI')
    
    

