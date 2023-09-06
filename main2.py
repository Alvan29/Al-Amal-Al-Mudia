from ubidots import ApiClient
from ultralytics import YOLO
import cv2
import math 
import time
import RPi.GPIO as GPIO

# Inisialisasi koneksi ke API Ubidots
api = ApiClient(token="BBFF-veCVwcOYvgAkrvBByLAlFlfTdPqnLt")

# Konfigurasi pin LED
LED_A = 17  # Ganti dengan pin yang sesuai pada Raspberry Pi
LED_B = 27
LED_C = 22
LED_D = 23
LED_E = 24
BUTTON_PIN = 16
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_A, GPIO.OUT)
GPIO.setup(LED_B, GPIO.OUT)
GPIO.setup(LED_C, GPIO.OUT)
GPIO.setup(LED_D, GPIO.OUT)
GPIO.setup(LED_E, GPIO.OUT)

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# model
model = YOLO("best.pt")

# object classes
classNames = ["Bendera", "Dasi", "Kelas", "Logo", "Osis"]

#Varibel yang sesuai dengan Ubidots
variable_ids = {
    "Bendera": "64edeb70e38025222b2a13e4",  # Ganti dengan ID variabel sebenarnya
    "Logo": "64edca52ff9abc1ca571e573",
    "Dasi": "64edeb6f38e6b1000d319773",
    "Kelas": "64edeb65fa3844000fc99efb",
    "Osis": "64edeba6b7f744000d88988d"
}

while True:
    button_state = GPIO.input(BUTTON_PIN)
    if button_state == GPIO.LOW:
        print("ditekan")
        GPIO.output(LED_A, GPIO.HIGH)
        GPIO.output(LED_B, GPIO.HIGH)
        GPIO.output(LED_C, GPIO.HIGH)
        GPIO.output(LED_D, GPIO.HIGH)
        GPIO.output(LED_E, GPIO.HIGH)
    success, img = cap.read()
    results = model(img, stream=True)
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            

            if classNames[cls] == "Bendera":
                GPIO.output(LED_A, GPIO.LOW)
                print("led a mati")
            elif classNames[cls] == "Logo":
                GPIO.output(LED_B, GPIO.LOW)
                print("led b mati")
            elif classNames[cls] == "Dasi":
                GPIO.output(LED_C, GPIO.LOW)
                print("led c mati")
            elif classNames[cls] == "Kelas":
                GPIO.output(LED_D, GPIO.LOW)
                print("led d mati")
            elif classNames[cls] == "Osis":
                GPIO.output(LED_E, GPIO.LOW)
                print("led e mati")
                

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            # Dapatkan nama kelas
            class_name = classNames[cls]
            # Dapatkan ID variabel Ubidots yang sesuai
            variable_id = variable_ids.get(class_name)

            if variable_id:
                # Send data to Ubidots
                value = confidence
                context = {"class": class_name, "coordinates": f"{x1},{y1},{x2},{y2}"}
                new_data = api.save_collection([
                    {
                        "variable": variable_id,
                        "value": value,
                        "context": context
                    }
                ])

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()