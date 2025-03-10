import logging
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("BiRefNetFastAPI")

app = FastAPI()

# Set torch matmul precision
try:
    logger.info("Setting torch float32 matmul precision to 'high'")
    torch.set_float32_matmul_precision('high')
except Exception as e:
    logger.exception("Error setting torch matmul precision")

# Device selection: use CUDA if available; otherwise, use CPU.
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.exception("Error during device setup")
    raise e

# Load the BiRefNet model on the selected device.
try:
    logger.info("Loading BiRefNet model from Hugging Face")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet', trust_remote_code=True
    )
    birefnet.to(device).eval()
    # Use half precision only if using CUDA.
    if device.type == "cuda":
        birefnet.half()
        logger.info("Converted model to half precision for CUDA")
    else:
        logger.info("Using full precision (float32) on CPU")
except Exception as e:
    logger.exception("Error loading the BiRefNet model")
    raise e


def extract_object(image: Image.Image) -> Image.Image:
    """Process the image using BiRefNet to extract an object and apply an alpha mask."""
    try:
        logger.debug("Starting image preprocessing")
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        image_rgb = image.convert("RGB")
        logger.debug("Converted image to RGB")
    except Exception as e:
        logger.exception("Error during image conversion/preprocessing")
        raise HTTPException(status_code=400, detail="Invalid image format or preprocessing error")

    try:
        if device.type == "cuda":
            input_tensor = transform_image(image_rgb).unsqueeze(0).to(device).half()
            logger.debug("Transformed image tensor to half precision on CUDA")
        else:
            input_tensor = transform_image(image_rgb).unsqueeze(0).to(device)
            logger.debug("Transformed image tensor to full precision")
    except Exception as e:
        logger.exception("Error transforming image to tensor")
        raise HTTPException(status_code=500, detail="Error processing image tensor")

    try:
        logger.debug("Running model prediction")
        with torch.no_grad():
            output = birefnet(input_tensor)
            preds = output[-1].sigmoid().cpu().squeeze()
        logger.debug("Model prediction completed")
    except Exception as e:
        logger.exception("Error during model prediction")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    try:
        # Ensure the predicted mask tensor has shape (C, H, W)
        if preds.ndim == 2:
            preds = preds.unsqueeze(0)
        # Swap dimensions: PIL image.size is (width, height) but tensor resize expects (height, width)
        target_size = (image.height, image.width)
        pred_mask_resized = transforms.functional.resize(preds, target_size)
        mask_pil = transforms.ToPILImage()(pred_mask_resized)
        mask_pil = mask_pil.convert('L')
        logger.debug(
            f"Resized mask and converted to PIL 'L' mode; mask size: {mask_pil.size}, image size: {image.size}")
    except Exception as e:
        logger.exception("Error resizing mask")
        raise HTTPException(status_code=500, detail="Error resizing mask")

    try:
        # Convert original image to RGBA before applying alpha mask.
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask_pil)
        logger.debug("Applied alpha mask to image")
    except Exception as e:
        logger.exception("Error applying alpha mask to image")
        raise HTTPException(status_code=500, detail="Error applying mask to image")

    return image_rgba


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    logger.info("Received image processing request")
    try:
        file_bytes = await file.read()
        logger.debug("File read successfully")
    except Exception as e:
        logger.exception("Error reading uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        logger.debug("Opened image successfully")
    except Exception as e:
        logger.exception("Error opening image file")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        processed_image = extract_object(image)
        logger.info("Image processed successfully")
    except HTTPException as http_e:
        logger.error("HTTP error during image processing")
        raise http_e
    except Exception as e:
        logger.exception("Unexpected error during image processing")
        raise HTTPException(status_code=500, detail="Image processing failed")

    try:
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        logger.debug("Processed image converted to PNG byte stream")
    except Exception as e:
        logger.exception("Error converting processed image to PNG")
        raise HTTPException(status_code=500, detail="Failed to convert image to PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
