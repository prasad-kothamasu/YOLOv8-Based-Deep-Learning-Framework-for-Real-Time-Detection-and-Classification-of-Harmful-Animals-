from ultralytics import YOLO
import requests
import numpy as np
import cv2
import time
import smtplib
import geocoder
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import serial

model = YOLO("last.pt")
cap = cv2.VideoCapture(0)

target_animals = [
    'bear', 'elephant', 'fox', 'hippopotamus',
    'hyena', 'leopard', 'lion', 'pig',
    'tiger', 'wolf'
]

class_names = model.names
target_ids = [i for i, name in class_names.items() if name in target_animals]

detection_start_time = None
email_sent = False

def serial_pass():
    SerialObj = serial.Serial('COM8')
    SerialObj.baudrate = 9600
    SerialObj.bytesize = 8
    SerialObj.parity   ='N'
    SerialObj.stopbits = 1
    SerialObj.write(b'a')
    time.sleep(2)
    SerialObj.close()

def get_location():
    g = geocoder.ip('me')
    if g.ok:
        lat, lng = g.latlng
        return lat, lng
    return None, None

def send_email(image_path, lat, lng):
    sender = "vtu24185@veltech.edu.in"
    receiver = "jayaprasadkothamasu@gmail.com"
    password = "vbrpfxooqoifzqft"

    msg = MIMEMultipart()
    msg['Subject'] = "⚠️ Wild Animal Detected!"
    msg['From'] = sender
    msg['To'] = receiver

    
    body = f"""
    Dangerous animal detected continuously for 5 seconds.
    """


    msg.attach(MIMEText(body))

    # Attach image
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename="detected.jpg")
        msg.attach(img)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("Email sent with location!")
    except Exception as e:
        print("Email failed:", e)



stream_url = "http://10.31.62.243:81/stream"

try:
    stream = requests.get(stream_url, stream=True)
    buffer = bytes()

    for chunk in stream.iter_content(chunk_size=8192):
        if not chunk:
            print("Received empty chunk.")
            continue

        buffer += chunk
        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')

        if start != -1 and end != -1:
            jpg_data = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            if len(jpg_data) > 0:
                nparr = np.frombuffer(jpg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    results = model(frame, conf=0.3, classes=target_ids)
                    detected = len(results[0].boxes) > 0

                    current_time = time.time()

                    if detected:
                        if detection_start_time is None:
                            detection_start_time = current_time

                        elapsed = current_time - detection_start_time

                        if elapsed >= 5 and not email_sent:
                            image_path = "detected.jpg"
                            annotated_frame = results[0].plot()
                            cv2.imwrite(image_path, annotated_frame)

                            lat, lng = get_location()
                            serial_pass()
                            send_email(image_path, lat, lng)

                            email_sent = True

                    else:
                        detection_start_time = None
                        email_sent = False

                    annotated_frame = results[0].plot()
                    cv2.imshow("Animal Detection", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to decode frame.")
            else:
                print("Received empty JPEG data.")
        else:
            print("Waiting for complete JPEG data...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except requests.exceptions.RequestException as e:
    print(f"Error connecting to the stream: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    cv2.destroyAllWindows()


            
