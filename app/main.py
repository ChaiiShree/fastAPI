from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import logging
from PIL import Image
import io
import pytesseract
import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from inference_sdk import InferenceHTTPClient
from sklearn.cluster import KMeans

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Generative AI Setup
genai.configure(api_key="AIzaSyBD74O8ZZYcqXIBOS554L2qyfE44QHHyds")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize InferenceHTTPClient
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="BlsORQTDLd2n4rILvyUU")


def process_image(image_file: UploadFile):
    image = Image.open(image_file.file)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    
    blob = glm.Blob(
        mime_type='image/jpeg',
        data=img_byte_arr.getvalue()
    )
    
    return blob


def convert_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.post("/analyze_faces")
async def analyze_faces(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode image with bounding boxes to base64
        predicted_image_str = convert_image_to_base64(img)

        return JSONResponse(content={"face_count": len(faces), "predicted_image": predicted_image_str})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/analyze_objects")
async def analyze_objects(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert image to base64
        img_base64 = convert_image_to_base64(img)

        # Infer objects in the image using the model
        result = CLIENT.infer(img_base64, model_id="yolov8n-640")

        # Log the result for debugging
        logging.info(f"Inference result: {result}")

        # Check if any objects are detected
        if not result['predictions']:
            return JSONResponse(content={"objects": []})

        object_descriptions = []
        for obj in result['predictions']:
            label = obj['class']

            # Process the image and use Generative AI to describe the object
            blob = process_image(image)

            # Initialize the Generative Model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-exp-0827",
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Prompt for object description
            prompt = f"Describe the object: {label}."
            response = model.generate_content([prompt, blob])
            description = response.text

            # Append the object label and description to the list
            object_descriptions.append({"label": label, "description": description})

        # Return the object names and descriptions
        return JSONResponse(content={"objects": object_descriptions})

    except Exception as e:
        logging.error(f"Error during object analysis: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/blur_sensitive_info")
async def blur_sensitive_info(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale and detect faces
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Blur detected faces
        for (x, y, w, h) in faces:
            # Extract the face from the image
            face = img[y:y+h, x:x+w]

            # Apply Gaussian blur to the face
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the face with the blurred version in the original image
            img[y:y+h, x:x+w] = blurred_face

        # Encode the blurred image to base64
        blurred_image_str = convert_image_to_base64(img)

        return JSONResponse(content={"blurred_image": blurred_image_str, "faces_blurred": len(faces)})
    except Exception as e:
        logging.error(f"Error during blurring sensitive info: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/generate_image_caption")
async def generate_image_caption(image: UploadFile = File(...)):
    try:
        # Process the image and create a Blob
        blob = process_image(image)

        # Initialize the Generative Model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-exp-0827",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Send prompt to AI to generate a caption for the image
        prompt = """
        Write a detailed caption for the following image content:
        """
        response = model.generate_content([prompt, blob])

        return JSONResponse(content={"generated_caption": response.text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/analyze_color_palette")
async def analyze_color_palette(image: UploadFile = File(...), num_colors: int = 5):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize image to speed up the KMeans algorithm
        resized_img = cv2.resize(img, (100, 100))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # Reshape image to a 2D array of pixels
        img_reshaped = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))

        # Apply KMeans clustering to find dominant colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(img_reshaped)
        colors = kmeans.cluster_centers_.astype(int).tolist()

        # Convert colors to hex for easy display
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in colors]

        return JSONResponse(content={"color_palette": hex_colors})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/")
async def root():
    return {"Hello": "World"}
