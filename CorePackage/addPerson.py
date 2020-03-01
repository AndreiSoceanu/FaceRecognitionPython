import cv2
import dlib

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 60)

[faceCenter, faceLeft, faceRight, faceUp, faceDown] = [False, False, False, False, False]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Data/shape_predictor_68_face_landmarks.dat")
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=15, sigmaColor=15, sigmaSpace=75)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        ROI = frame[y1:y2, x1:x2]

        landmarks = predictor(gray, face)
        noseTip = (landmarks.part(30).x, landmarks.part(30).y)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)
        cv2.circle(frame, noseTip, 6, (255, 255, 255), -1)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 27:
        break

# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
