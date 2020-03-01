import cv2
import dlib
import numpy as np

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
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        leftEyeLeftCorner = (landmarks.part(36).x, landmarks.part(36).y)
        rightEyeRightCorner = (landmarks.part(45).x, landmarks.part(45).y)
        leftMouthCorner = (landmarks.part(48).x, landmarks.part(48).y)
        rightMouthCorner = (landmarks.part(54).x, landmarks.part(54).y)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # Head Pose Est
        image_points = np.array([
            noseTip,  # Nose tip
            chin,  # Chin
            leftEyeLeftCorner,  # Left eye left corner
            rightEyeRightCorner,  # Right eye right corner
            leftMouthCorner,  # Left Mouth corner
            rightMouthCorner  # Right mouth corner
        ], dtype="double")

        # 3D model points.

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)

        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # ---

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 27:
        break

# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
