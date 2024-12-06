from fastapi import FastAPI, HTTPException, Request
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# Initialize the FastAPI app
app = FastAPI()

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLOv11 model path

# Firebase configuration
cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://despro-halte-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

@app.post("/detect")
async def predict(request: Request):
    # Read raw image data from request body
    image_data = await request.body()

    # Convert image data to NumPy array using PIL
    try:
        img = Image.open(io.BytesIO(image_data))
        img = np.array(img)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # Run YOLO prediction
    results = model.predict(img, conf=0.1, iou=0.8)
    boxes = results[0].boxes

    # Count the number of "person" detections
    person_count = 0
    for box in boxes:
        class_id = int(box.cls[0])  # Class ID
        if class_id != 0:  # Assuming "person" class is ID 0
            continue

        # Increment the person count
        person_count += 1

    # Update the 'person_count' at a fixed location (e.g., '/Crowd Density/total_person_count')
    ref = db.reference("/Crowd Density/Person Count")  # Specify the fixed location for the count
    ref.set(person_count)  # Update the value of person_count

    # Return response with the counts of persons detected
    return {"person_count": person_count}
