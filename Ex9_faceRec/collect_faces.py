import cv2
import os

person_name = input("Enter person name: ")
save_path = f"dataset/{person_name}"
os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(f"{save_path}/{count}.jpg", face)
            count += 1
            print(f"Saved image {count}")

    cv2.imshow("Press S to Save, Q to Quit", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()