from ultralytics import YOLO
import cv2
import time
import smtplib
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

def send_email(image_path):
    sender = "vtu24185@veltech.edu.in"
    receiver = "jayaprasadkothamasu@gmail.com"
    password = "vbrpfxooqoifzqft"

    msg = MIMEMultipart()
    msg['Subject'] = "⚠️ Wild Animal Detected!"
    msg['From'] = sender
    msg['To'] = receiver

    # Google Maps link
    #maps_link = f"https://www.google.com/maps?q={lat},{lng}" if lat else "Location not available"

    body = f"""
    Dangerous animal detected continuously for 5 seconds.
    """
##    📍 Location:
##    Latitude: {lat}
##    Longitude: {lng}
##
##    🗺️ Map:
##    {maps_link}

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, classes=target_ids)
    detected = len(results[0].boxes) > 0

    current_time = time.time()

    if detected:
        if detection_start_time is None:
            detection_start_time = current_time

        elapsed = current_time - detection_start_time

        if elapsed >= 2 and not email_sent:
            image_path = "detected.jpg"
            annotated_frame = results[0].plot()
            cv2.imwrite(image_path, annotated_frame)

            #lat, lng = get_location()
            serial_pass()
            send_email(image_path)

            email_sent = True

    else:
        detection_start_time = None
        email_sent = False

    annotated_frame = results[0].plot()
    cv2.imshow("Dangerous Animal Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
