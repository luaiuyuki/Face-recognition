import cv2
import os

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set chiều rộng khung hình
cam.set(4, 480)  # set chiều cao khung hình

# Load file cascade để nhận diện khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập ID khuôn mặt
face_id = input('\n Nhap ID Khuon Mat <return> ==> ')

print("\n [INFO] Khoi tao Camera ...")
count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Khong doc duoc frame tu camera")
        break

    # Lật ảnh cho tự nhiên
    img = cv2.flip(frame, 1)
    # Chuyển ảnh sang đen trắng để nhận diện khuôn mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)  # Bỏ qua mặt nhỏ hơn 100x100
    )

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Cắt và resize khuôn mặt về 200x200
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200, 200))

        # Lưu ảnh khuôn mặt
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face_resized)

        print(f"[INFO] Da chup anh thu {count}")

    # Hiển thị khung hình
    cv2.imshow('image', img)

    # Nhấn ESC để thoát hoặc chụp đủ 30 ảnh
    k = cv2.waitKey(100) & 0xff  # Chờ lâu hơn một chút để không chụp quá nhanh
    if k == 27:
        break
    elif count >= 30:
        break

# Giải phóng tài nguyên
print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()
