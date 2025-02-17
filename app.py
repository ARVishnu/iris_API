from PIL import Image
import numpy as np
import onnxruntime as ort
import gc
# from display_output import display_segmentation  # Import the display function

class IrisSegmenter:
    def __init__(self, model_path='iris_semseg_upp_scse_mobilenetv2.onnx'):
        """Initialize the segmentation model."""
        self.session = ort.InferenceSession(model_path)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def segment_image(self, image_path):
        """
        Segments the pupil from the given image path using a pre-trained ONNX model.
        Returns both the original image and the mask for overlay purposes.
        """
        try:
            # Load and resize the image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            image = image.resize((640, 480))

            # Convert to numpy array and normalize
            image_np = np.array(image).astype(np.float32) / 255.0
            image_np = (image_np - self.mean) / self.std

            # Transpose to match the model's input shape
            image_np = np.transpose(image_np, (2, 0, 1))
            image_np = np.expand_dims(image_np, axis=0)

            # Run inference
            inputs = {self.session.get_inputs()[0].name: image_np}
            outputs = self.session.run(None, inputs)

            # Extract the pupil probability map
            pupil_prob_map = outputs[0][0, 2, :, :]

            # Apply threshold to get binary mask
            threshold = 0.5
            pupil_mask = (pupil_prob_map > threshold).astype(np.uint8)

            # Create overlay image
            overlay = Image.fromarray(pupil_mask * 255).resize(original_size)
            
            return overlay
            
        finally:
            # Clean up resources
            gc.collect()
            
            # Clear large variables
            if 'image_np' in locals():
                del image_np
            if 'outputs' in locals():
                del outputs
            if 'pupil_prob_map' in locals():
                del pupil_prob_map
            if 'pupil_mask' in locals():
                del pupil_mask
            if 'image' in locals():
                del image

# Create a single instance of the segmenter
segmenter = IrisSegmenter()

# Example usage:
# image_path = 'Vishnu_img.jpg'
# display_segmentation(image_path)  # Use the display function to show the output