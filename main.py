import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller
import time
from math import radians


mouse = Controller()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model_points = np.array([
    (0.0, 0.0, 0.0),  # Нос
    (0.0, -330.0, -65.0),  # Подбородок
    (-225.0, 170.0, -135.0),  # Левый глаз
    (225.0, 170.0, -135.0),  # Правый глаз
    (-150.0, -150.0, -125.0),  # Левый угол рта
    (150.0, -150.0, -125.0)  # Правый угол рта
], dtype=np.float64)

point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                (255, 255, 0), (0, 255, 255), (255, 0, 255)]
point_labels = ['Nose', 'Chin', 'Right Eye',
                'Left Eye', 'Right Mouth', 'Left Mouth']

landmark_indices = [
    4,    # Нос
    152,  # Подбородок
    263,  # Правый глаз
    33,   # Левый глаз
    287,  # Правый угол рта
    57    # Левый угол рта
]

WINDOW_NAME = 'Face Controlled Cursor'
difference_to_move = 12
step = radians(1)
difference_up = difference_to_move
difference_down = difference_to_move
difference_left = difference_to_move
difference_right = difference_to_move
calibration = True
calibration_y = []
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = image.shape[:2]
            image_points = []
            for idx in landmark_indices:
                lm = face_landmarks.landmark[idx]
                image_points.append((lm.x * w, lm.y * h))

            image_points = np.array(image_points, dtype=np.float64)

            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            _, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                None
            )

            # Расчет углов Эйлера
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Определение направления движения
            y_angle = angles[0]
            x_angle = angles[1]

            if calibration:
                current_time = time.time()
                cv2.putText(image, f"Calibration: {5 - int(current_time - start_time)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, "Look straight, do not move", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                dif_time = current_time - start_time
                if dif_time >= 2:
                    calibration_y.append(y_angle)
                if dif_time >= 5:
                    avg_y = sum(calibration_y) / len(calibration_y)
                    del calibration_y
                    difference_down = abs(int(avg_y + difference_to_move))
                    difference_up = abs(int(avg_y - difference_to_move))
                    calibration = False
            else:
                # движение мыши
                if x_angle < -difference_right: # вправо
                    mouse.move(-int(x_angle) - difference_right, 0)
                elif x_angle > difference_left: # влево
                    mouse.move(-int(x_angle) + difference_left, 0)

                if y_angle < -difference_up: # вверх
                    mouse.move(0, int(y_angle) + difference_up)
                elif y_angle > difference_down: # вниз
                    mouse.move(0, int(y_angle) - difference_down)

            # Проекция 3D точек на 2D плоскость
            (projected_points, _) = cv2.projectPoints(
                model_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                None
            )

            # Отрисовка проекций точек
            for i, p in enumerate(projected_points):
                x, y = p.ravel().astype(int)
                cv2.circle(image, (x, y), 7, point_colors[i], -1)
                cv2.putText(image, point_labels[i], (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[i], 1)

    cv2.imshow(WINDOW_NAME, image)

    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()