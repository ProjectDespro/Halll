from fastapi import FastAPI, HTTPException, Request
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
import os
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

# Output directory for saved images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

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

    # Draw bounding boxes for "person" class only
    person_count = 0
    for box in boxes:
        class_id = int(box.cls[0])  # Class ID
        if class_id != 0:  # Assuming "person" class is ID 0
            continue

        # Count the person detection
        person_count += 1

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        confidence = float(box.conf[0])        

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label = f"Person ({confidence:.2f})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes
    output_path = os.path.join(output_dir, "output_image.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Update the 'person_count' at a fixed location (e.g., '/Crowd Density/total_person_count')
    ref = db.reference("/Crowd Density/Person Count")  # Specify the fixed location for the count
    ref.set(person_count)  # Update the value of person_count

    # Return response with the counts of persons detected
    return {"person_count": person_count, "saved_path": output_path}
