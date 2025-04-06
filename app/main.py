# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import base64
# import logging
# from PIL import Image
# import io
# import google.generativeai as genai
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from inference_sdk import InferenceHTTPClient
# from sklearn.cluster import KMeans
# from typing import List
# import asyncio
# from itertools import cycle

# app = FastAPI()

# # Setup CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Google Generative AI Setup
# API_KEYS = [
#     "AIzaSyBD74O8ZZYcqXIBOS554L2qyfE44QHHyds",
#     "AIzaSyAckv8FLYmYiZ0MJclXeYcRXoF-Sb6303w",
#     "AIzaSyBHmJPU5c0dDNgtGWZnbdUL39K_1D3LXPM",
#     "AIzaSyAtokft4iVP5F_XaSvb8IJruHZhbWq0OJA",
#     "AIzaSyAObGZyw_XuJuEqY9nSN61vE09NN9AZziA"
# ]
# api_key_cycle = cycle(API_KEYS)

# generation_config = {
#     "temperature": 0.7,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 1024,
# }

# safety_settings = {
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# }

# # Initialize InferenceHTTPClient
# CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="BlsORQTDLd2n4rILvyUU")

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def get_next_api_key():
#     return next(api_key_cycle)

# async def generate_content_with_retry(model, prompt, blob, max_retries=5):
#     for i in range(max_retries):
#         try:
#             response = await model.generate_content([prompt, blob])
#             return response.text
#         except Exception as e:
#             logger.warning(f"API call failed (attempt {i+1}/{max_retries}): {str(e)}")
#             if i == max_retries - 1:
#                 raise
#             await asyncio.sleep(2 ** i)  # Exponential backoff

# def process_image(image_file: UploadFile) -> Image.Image:
#     image = Image.open(image_file.file)
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     return image

# def image_to_base64(image: Image.Image) -> str:
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# @app.post("/analyze_faces")
# async def analyze_faces(image: UploadFile = File(...)):
#     try:
#         img = process_image(image)
#         img_array = np.array(img)
#         gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         result_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
#         predicted_image_str = image_to_base64(result_image)

#         return JSONResponse(content={"face_count": len(faces), "predicted_image": predicted_image_str})
#     except Exception as e:
#         logger.error(f"Error in analyze_faces: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/analyze_objects")
# async def analyze_objects(image: UploadFile = File(...)):
#     try:
#         img = process_image(image)
#         img_base64 = image_to_base64(img)
#         result = CLIENT.infer(img_base64, model_id="yolov8n-640")
        
#         if not result['predictions']:
#             return JSONResponse(content={"objects": []})

#         object_descriptions = []
#         genai.configure(api_key=get_next_api_key())
#         model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827", generation_config=generation_config, safety_settings=safety_settings)

#         for obj in result['predictions']:
#             label = obj['class']
#             prompt = f"Describe the object '{label}' in detail, focusing on its typical characteristics, uses, and any interesting facts. Keep the description concise but informative."
#             description = await generate_content_with_retry(model, prompt, img)
#             object_descriptions.append({"label": label, "description": description})

#         return JSONResponse(content={"objects": object_descriptions})
#     except Exception as e:
#         logger.error(f"Error in analyze_objects: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/blur_sensitive_info")
# async def blur_sensitive_info(image: UploadFile = File(...)):
#     try:
#         img = process_image(image)
#         img_array = np.array(img)
#         gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             face = img_array[y:y+h, x:x+w]
#             blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
#             img_array[y:y+h, x:x+w] = blurred_face

#         result_image = Image.fromarray(img_array)
#         blurred_image_str = image_to_base64(result_image)

#         return JSONResponse(content={"blurred_image": blurred_image_str, "faces_blurred": len(faces)})
#     except Exception as e:
#         logger.error(f"Error in blur_sensitive_info: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate_image_caption")
# async def generate_image_caption(image: UploadFile = File(...)):
#     try:
#         img = process_image(image)
#         genai.configure(api_key=get_next_api_key())
#         model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827", generation_config=generation_config, safety_settings=safety_settings)

#         prompt = """
#         Analyze the image and provide a detailed, engaging caption that describes:
#         1. The main subject or focus of the image
#         2. Any notable actions, emotions, or interactions
#         3. The setting or background, if relevant
#         4. Any striking visual elements or composition details
#         Keep the caption concise but informative, aiming for 2-3 sentences.
#         """
#         caption = await generate_content_with_retry(model, prompt, img)
#         return JSONResponse(content={"generated_caption": caption})
#     except Exception as e:
#         logger.error(f"Error in generate_image_caption: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/analyze_color_palette")
# async def analyze_color_palette(image: UploadFile = File(...), num_colors: int = 5):
#     try:
#         img = process_image(image)
#         img_array = np.array(img.resize((100, 100)))
#         pixels = img_array.reshape(-1, 3)

#         kmeans = KMeans(n_clusters=num_colors, n_init=10)
#         kmeans.fit(pixels)
#         colors = kmeans.cluster_centers_.astype(int).tolist()
#         hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors]

#         return JSONResponse(content={"color_palette": hex_colors})
#     except Exception as e:
#         logger.error(f"Error in analyze_color_palette: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Image Analysis API"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pillmodel import get_prediction
import base64
from fastapi.staticfiles import StaticFiles
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.ai.generativelanguage as glm
from PIL import Image
import io
import random
import re
import json
from dotenv import load_dotenv
load_dotenv()

api_keys = os.getenv('GEMINI_API_KEYS').split(',')
print(api_keys)

from inference_sdk import InferenceHTTPClient



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the image to a temporary location
    # temp_image_path = "temp_image.jpg"
    # cv2.imwrite(temp_image_path, img)

    # Prediction
    predicted_image, count_dict = get_prediction(img)
    # Encode predicted image to base64
    _, buffer = cv2.imencode('.jpg', predicted_image)
    predicted_image_str = base64.b64encode(buffer).decode('utf-8')

    # Send a confirmation message
    message_to_send = (
        f"There are {count_dict.get('capsules', 0)} capsules and {count_dict.get('tablets', 0)} tablets. "
        f"A total of {count_dict.get('capsules', 0) + count_dict.get('tablets', 0)} pills."
    )
    
    return JSONResponse(content={"message": message_to_send, "count": count_dict, "predicted_image": predicted_image_str})






@app.post("/predict_wheat")
async def predict_wheat(image: UploadFile = File(...), model_id: str = "grian/1"):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # delete the image if exists
    try:
        os.remove("temp_image.jpg")
    except:
        print("temp_image.jpg does not exist")

    # Save the image to a temporary location
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, img)
    
    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="PpEebXofNuob5VSx7YP3"
    )
    

    result = CLIENT.infer("temp_image.jpg", model_id=model_id)
    # Prediction
    predicted_count = len(result['predictions'])
    message_to_send = (
        f"There are {predicted_count} wheat grains."
    )

    for prediction in result['predictions']:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    # Encode predicted image to base64
    _, buffer = cv2.imencode('.jpg', img)
    predicted_image_str = base64.b64encode(buffer).decode('utf-8')

    
    return JSONResponse(content={"message": message_to_send, "count": predicted_count, "predicted_image": predicted_image_str})



def process_image(file: UploadFile):
    image = Image.open(file.file)
    
    # Convert the image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the image to a byte array
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    
    # Create a Blob object
    blob = glm.Blob(
        mime_type='image/jpeg',
        data=img_byte_arr.getvalue()
    )
    
    return blob

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    selected_api_key = random.choice(api_keys)
    print(f"Selected API Key: {selected_api_key}")
    genai.configure(api_key=selected_api_key)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    # Process the image
    blob = process_image(file)
    
    # Initialize the Generative Model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Prompt for content generation
    prompt = """
    give a safety score for a website called unipall which is a olx, now when a user is uploading a product,
    tell me this in json like:
    only give this json nothing else not be too harmful
    when a picture contains some accessories in a scene focus on them and don't flag it
    don't flag text on the product
    {
        useable_on_website: true/false,
        safety_score: /100,
        category: "",
        reason: "",
        suggested_product_title: "",
        suggested_product_description: ""
    }
    """

    # Generate content using the AI model
    response = model.generate_content([prompt, blob])

    if '```json' not in response.text:
        return JSONResponse(content=response.text ,media_type="application/json")
        
    # Extract JSON string from Markdown-formatted JSON string
    json_string = re.search(r'```json(.*?)```', response.text, re.DOTALL).group(1)
    
    # Clean JSON string
    cleaned_response = json_string.strip()

    # Parse the cleaned string as JSON
    data = json.loads(cleaned_response)
    fd = json.dumps(data, indent=4)

    # Return the AI-generated response
    return JSONResponse(content=fd ,media_type="application/json")
    


app.mount("/", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return HTMLResponse(content="<html><head><meta http-equiv='refresh' content='0; url=/index.html'></head></html>")


