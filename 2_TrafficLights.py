from ultralytics import YOLO
import cv2
import cvzone
from urllib.request import urlopen
import json
import math
import time
import os
from datetime import datetime
from firebase_admin import credentials, storage
from firebase_admin import firestore
import firebase_admin

# Initialize Firebase Admin SDK for Authentication and Firestore
cred = credentials.Certificate("techsavants-bdf20-firebase-adminsdk-flfvo-d5dce4bf44.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'techsavants-bdf20.appspot.com'
})

# Initialize Firebase Storage
bucket = storage.bucket()

# Initialize Firestore
db = firestore.client()

# Constants for distance estimation
KNOWN_DISTANCE = 1.0  # Distance from camera to object in some unit (like meters)
KNOWN_HEIGHT = 0.32  # Height of the object in the same unit

# Initialize
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("traff.pt")
yolo_model = YOLO("Weights/yolov8l.pt")

# Define the allowed classes for the 'traff.pt' model
allowed_classes = ["notworking", "working"]

# Define the allowed classes for the 'yolov8l.pt' model
yolo_allowed_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck", "motorcycle"]

prev_frame_time = 0
new_frame_time = 0

# Flag to indicate if the 'yolov8l.pt' model should be active
yolo_active = False

# Time when the count becomes 0
time_zero_count = None

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        break

    # Reset the object count for each frame
    object_count = 0

    if not yolo_active:
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            detected_objects = []  # To store detected object data

            for box in boxes:
                cls = int(box.cls[0])

                # Check if the class is in the allowed classes
                class_name = model.names[cls]
                if class_name in allowed_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Store detected object data
                    detected_object = {
                        'class': allowed_classes[cls],
                        'confidence': conf
                    }
                    detected_objects.append(detected_object)

                    # If 'notworking' class is detected, set yolo_active to True to activate 'yolov8l.pt' model
                    if class_name == 'notworking':
                        yolo_active = True

                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        c_time = datetime.now().strftime("%H:%M:%S")
                        c_date = datetime.now().strftime("%d-%m-%Y")

                        url = "http://ipinfo.io/json"
                        response = urlopen(url)
                        data = json.load(response)
                        lat, lon = map(float, data['loc'].split(','))

                        # Create the directory if it doesn't exist
                        os.makedirs("image/tl", exist_ok=True)

                        # Rest of your code remains the same
                        image_filename = f'image/tl/{timestamp}.jpg'

                        # Save the image locally
                        cv2.imwrite(image_filename, img)

                        # Upload the image to Firebase Storage
                        blob = bucket.blob(image_filename)
                        blob.upload_from_filename(image_filename)

                        # Get the URL of the uploaded image
                        image_url = blob.public_url

                        # Access Firestore and add the detected_objects data with the image URL
                        detected_objects_collection = db.collection("tl")
                        detected_objects_collection.add({
                            'Date': c_date,
                            'Time': c_time,
                            'Issue': allowed_classes[cls],
                            'Confidence': conf,
                            'Lattitude': lat,
                            'Longitude': lon,
                            'timestamp': timestamp,
                            'image_url': image_url
                        })

                    # Calculate distance based on the bounding box height
                    distance = (KNOWN_HEIGHT * img.shape[0]) / (h * KNOWN_DISTANCE)
                    distance = round(distance, 2)  # Round to 2 decimal places for better readability

                    # Set the color of the bounding box based on the distance
                    if distance < 2.00:
                        box_color = (0, 0, 255)  # Red
                        object_count += 1  # Increment object count for red boxes
                    else:
                        box_color = (0, 255, 0)  # Green

                    # Display class name and confidence
                    cvzone.putTextRect(img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=box_color,
                                       colorT=(0, 255, 255), colorR=box_color,
                                       offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)  # Decreased line thickness to 1

        # If object count is 0, update the time
        if object_count == 0:
            if time_zero_count is None:
                time_zero_count = time.time()
            else:
                elapsed_time = time.time() - time_zero_count
                # If the count remains 0 for 1 minute, change back to traffic light model
                if elapsed_time >= 60:
                    yolo_active = False
                    time_zero_count = None
        else:
            # Reset the time if the count is not 0
            time_zero_count = None

    else:
        # Use 'yolov8l.pt' model for general object detection
        yolo_results = yolo_model(img, stream=True)

        for r in yolo_results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])

                # Check if the class is in the allowed classes
                class_name = yolo_model.names[cls]
                if class_name in yolo_allowed_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Calculate distance based on the bounding box height
                    distance = (KNOWN_HEIGHT * img.shape[0]) / (h * KNOWN_DISTANCE)
                    distance = round(distance, 2)  # Round to 2 decimal places for better readability

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Set the color of the bounding box based on the distance
                    if distance < 2.00:
                        box_color = (0, 0, 255)  # Red
                        object_count += 1  # Increment object count for red boxes
                    else:
                        box_color = (0, 255, 0)  # Green

                    # Display class name and confidence
                    cvzone.putTextRect(img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=box_color,
                                       colorT=(0, 255, 255), colorR=box_color,
                                       offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)  # Decreased line thickness to 1

        # Display object count for each frame with a larger and bold font
        cvzone.putTextRect(img, f'Red Boxes Detected: {object_count}', (10, 30), scale=0.5, thickness=2,
                           colorB=(0, 0, 255),
                           colorT=(0, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(1)
