import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn thư mục chứa dữ liệu khuôn mặt
path = 'dataset'

# Khởi tạo bộ nhận diện LBPH và bộ phát hiện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Mở ảnh, chuyển sang ảnh xám
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Lấy ID từ tên file, ví dụ User.1.1.jpg -> id=1
        id = int(os.path.split(imagePath)[-1].split('.')[1])

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n[INFO] Đang training dữ liệu, vui lòng chờ ...")
faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

# Lưu model đã train vào thư mục trainer
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.write('trainer/trainer.yml')

print(f"\n[INFO] Hoan tat training. Da train {len(np.unique(ids))} khuon mat. Thoat.")
