import cv2
import numpy as np
import os

# Load model đã train
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load file cascade để phát hiện khuôn mặt
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font để hiển thị tên
font = cv2.FONT_HERSHEY_SIMPLEX

# Danh sách tên, chỉ số phải khớp ID
names = ['None', 'Putin', 'Jisoo']  # ví dụ ID=1 là Putin, ID=2 là Jisoo

# Mở camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # chiều rộng
cam.set(4, 480)  # chiều cao

# Kích thước khuôn mặt tối thiểu để phát hiện
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        id_pred, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 60:
            name = names[id_pred] if id_pred < len(names) else "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        # Hiển thị tên và độ tin cậy
        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)

    cv2.imshow('Nhan dien khuon mat', img)

    k = cv2.waitKey(10) & 0xff  # giảm thời gian chờ để mượt hơn
    if k == 27:
        break

# Giải phóng tài nguyên
print("\n[INFO] Thoat")
cam.release()
cv2.destroyAllWindows()
