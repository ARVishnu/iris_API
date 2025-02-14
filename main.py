from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import os
import logging

from io import BytesIO
from PIL import Image
from pathlib import Path
from app import segment_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model session globally
try:
    model_path = 'iris_semseg_upp_scse_mobilenetv2.onnx'
    session = ort.InferenceSession(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG and JPG files are allowed.")
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    
    try:
        # Open the uploaded image file
        image = Image.open(file.file)

        # Define the path to save the image
        image_path = output_dir / f"temp_{file.filename}"

        # Save the image
        image.save(image_path)
        
        logger.info(f"File saved temporarily at: {image_path}")
        
        # Create overlay image with both original and segmentation
        result_image = segment_image(image_path)
        
            # Save the processed image to a BytesIO object
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format=image.format)
        img_byte_arr.seek(0)

        # Return the processed image as a streaming response
        return StreamingResponse(img_byte_arr, media_type=file.content_type)
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # finally:
        # # Clean up the temporary file
        # if os.path.exists(image_path):
        #     os.remove(image_path)
        #     logger.info(f"Temporary file removed: {image_path}")
        
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-name/")
async def process_image(file: UploadFile = File(...)):
    image_path = os.path.join("output", f"temp_{file.filename}")
    return {"Name":f'{file.filename}',"Path":f'{image_path}'}

# Run the app with: uvicorn main:app --reload 
