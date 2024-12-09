from fastapi import FastAPI, HTTPException, Request
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from supabase import create_client, Client

# Initialize the FastAPI app
app = FastAPI()

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLO model path

# Firebase configuration
cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://despro-halte-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Supabase configuration
SUPABASE_URL = "https://zfkzncklxqjyejpavvys.supabase.co"  # Ganti dengan URL API Anda
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpma3puY2tseHFqeWVqcGF2dnlzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMzc1NTQ2NSwiZXhwIjoyMDQ5MzMxNDY1fQ.JYeHpSZneSreNzg1uSLV2fTnhI9rRIk-wmHIAW1geAk"  # Ganti dengan kunci API Anda
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(file_path, bucket_name, file_name):
    """
    Mengunggah file ke Supabase Storage bucket publik.
    Tidak mengembalikan apapun.
    """
    with open(file_path, "rb") as file:
        supabase.storage.from_(bucket_name).upload(file_name, file)
    
    print(f"File uploaded to {bucket_name} as {file_name}")

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
    results = model.predict(img, conf=0.05, iou=0.8)
    boxes = results[0].boxes

    # Count the number of "person" detections
    person_count = 0
    for box in boxes:
        class_id = int(box.cls[0])  # Class ID
        if class_id != 0:  # Assuming "person" class is ID 0
            continue

        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score

        # Draw the bounding box on the image
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Increment the person count
        person_count += 1

    # Save the processed image with bounding boxes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for unique filename
    output_path = f"detected_{timestamp}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # Update the 'person_count' at a fixed location (e.g., '/Crowd Density/Person Count')
    ref = db.reference("/Crowd Density/Person Count")  # Specify the fixed location for the count
    ref.set(person_count)  # Update the value of person_count

    # Upload the image to Supabase (without returning any response)
    bucket_name = "detected-image"  # Nama bucket yang dipakai
    file_name = f"detected_{timestamp}.jpg"
    try:
        upload_to_supabase(output_path, bucket_name, file_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image to Supabase: {str(e)}")

    # Return only the person count (no image URL, no response related to upload)
    return {"person_count": person_count}
