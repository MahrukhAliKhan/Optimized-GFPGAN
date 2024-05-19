#Import necessary libraries
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import asyncio
import torch
import multiprocessing 


# Load Model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model,
    tile=300,
    tile_pad=10,
    pre_pad=0,
    half=True  # need to set False in CPU mode
)

restorer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler
)

app = FastAPI()

multiprocessing.set_start_method("spawn", force=True)

#global ProcessPoolExecutor:: for multiprocessing
executor_process = ProcessPoolExecutor()

async def async_enhance_image(image_input):
    _, _, restored_img = restorer.enhance(
        image_input, has_aligned=False, only_center_face=False, paste_back=True
    )

    restored_img = cv2.cvtColor(np.array(restored_img), cv2.COLOR_BGR2RGB)
    res_image = Image.fromarray(restored_img)
    output_image = BytesIO()
    res_image.save(output_image, format="JPEG")
    output_image.seek(0)
    return output_image.getvalue()

def sync_enhance_image(image_input):   #function to run async functions
    return asyncio.run(async_enhance_image(image_input))

def cleanup_image_processing_resources():
    #Closing the ProcessPoolExecutor used for parallel processing
    executor_process.shutdown()
    cv2.destroyAllWindows()

    #Clear the CUDA memory if using PyTorch with GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    pass

## AI-Enhancer Endpoint ##
@app.post("/Enhance_Image")
async def paint(image: UploadFile = File(...)): 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    image_bytes = await image.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_input = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Use ProcessPoolExecutor to run the enhancement in a separate process
    enhanced_image = await asyncio.get_event_loop().run_in_executor(executor_process, sync_enhance_image, image_input)
    return StreamingResponse(content=BytesIO(enhanced_image), media_type="image/jpeg")


async def on_shutdown():
    # Clean up resources related to image processing libraries or models
    cleanup_image_processing_resources()

app.add_event_handler("shutdown", on_shutdown)




    