import cv2
import mediapipe as mp
import numpy as np
from log_settings import logger
from math import atan2, degrees





class FaceMeshClassifier:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5)
        self.mp_drawing = mp.solutions.drawing_utils

    def classify_jaw(self, image):
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return "No face detected"

        landmarks = results.multi_face_landmarks[0].landmark

        # Отримати координати ключових точок лівої скроні
        left_temple = np.array([landmarks[234].x, landmarks[234].y])

        # Отримати координати ключових точок правої скроні
        right_temple = np.array([landmarks[454].x, landmarks[454].y])

        # Отримати координати ключових точок підборіддя
        jaw_point = np.array([landmarks[152].x, landmarks[152].y])

        # Отримати точку на лобі
        forehead_point = np.array([landmarks[10].x, landmarks[10].y])

        # Отримати координати нижньої частини губи
        lips = np.array([landmarks[0].x, landmarks[0].y])

        # Використати усі ці точки для визначення класа щелепи

        # Визначити тип щелепи на основі кута між висками і підборіддям
        chin_lips_distance = np.linalg.norm(lips - jaw_point)

        # Визначити тип щелепи на основі кута між висками і підборіддям
        angle = degrees(atan2(jaw_point[1] - left_temple[1], jaw_point[0] - left_temple[0]) - atan2(jaw_point[1] - right_temple[1], jaw_point[0] - right_temple[0]))


        if 83.3 <= abs(angle) <= 90:
            jaw_type = "wide"
        elif 80 <= abs(angle) < 83.3:
            jaw_type = "medium"
        elif abs(angle) < 80 or abs(angle) > 90:
            jaw_type = "narrow"
        else:
            jaw_type = "unknown"

        return chin_lips_distance, angle, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips


    def draw_landmarks(self, image, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )
        return image


def main():
    classifier = FaceMeshClassifier()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chin_lips_distance, angle, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips = classifier.classify_jaw(frame_rgb)
        results = classifier.face_mesh.process(frame_rgb)
        frame_with_landmarks = classifier.draw_landmarks(frame, results)

        # Відобразити отримані точки на зображенні та відстані між ними
        if results.multi_face_landmarks:
            cv2.circle(frame_with_landmarks, tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_landmarks, tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_landmarks, tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), 5, (255, 0, 0), -1)
            cv2.circle(frame_with_landmarks, tuple((int(forehead_point[0] * frame.shape[1]), int(forehead_point[1] * frame.shape[0]))), 5, (255, 255, 0), -1)
            cv2.circle(frame_with_landmarks, tuple((int(lips[0] * frame.shape[1]), int(lips[1] * frame.shape[0]))), 5, (255, 255, 255), -1)

            # обов’язково нанести маркери та лінію ширини щелепи
            cv2.line(frame_with_landmarks, tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), (255, 0, 255), 2) # лінія між лівою та правою скронями
            cv2.line(frame_with_landmarks, tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), tuple((int(forehead_point[0] * frame.shape[1]), int(forehead_point[1] * frame.shape[0]))), (0, 255, 255), 2) # лінія від підборіддя до лоба
            cv2.line(frame_with_landmarks, tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), (0, 255, 0), 2) # лінія від лівої скроні до підборіддя
            cv2.line(frame_with_landmarks, tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), (0, 255, 0), 2) # лінія від правої скроні до підборіддя


            with open("result.log", "r") as file:
                accuracy_value = []
                lines = file.readlines()
                accuracy_line = next((line for line in lines if "Total Accuracy" in line), None)
                if accuracy_line:
                    accuracy = accuracy_line.split(": ")[1].strip().replace("%", "")
                    accuracy_value.append(accuracy)


            cv2.putText(frame_with_landmarks, f"Jaw Type: {jaw_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_landmarks, f"Chin to Lips Distance: {chin_lips_distance:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_landmarks, f"Angle: {abs(angle):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_landmarks, f"Accuracy: {accuracy_value[-1]}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(frame_with_landmarks, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f'Face Mesh Models', frame_with_landmarks)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Получить изображения из дерриктории test_photos/ и ее поддерикториями и применить к ним классификатор
def classify_images_from_directory(directory):
    import os
    accuracy = 0
    errors = 0
    classifier = FaceMeshClassifier()
    for root, dirs, files in os.walk(directory):
        for index, file in enumerate(files):
            if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    chin_lips_distance, angle, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips = classifier.classify_jaw(image)
                    logger.info(f"\nIteration: {index}"
                                f"\n\tImage: {file}"
                                f"\n\tJaw Type: {jaw_type}"
                                f"\n\tChin to Lips Distance: {chin_lips_distance:.4f}\n"
                                f"\tAngle: {abs(angle):.2f} degrees\n")

                    if jaw_type in file:
                        accuracy += 1
                    else:
                        errors += 1
                else:
                    logger.error(f"Could not read image: {image_path}")

    # Выводим общую статистику по классификации в процентном соотношении
    if accuracy + errors > 0:
        total_accuracy = (accuracy / (accuracy + errors)) * 100
        logger.info(f"Total Accuracy: {total_accuracy:.2f}%\n")
    else:
        logger.info("No images processed.\n")


if __name__ == "__main__":
    classify_images_from_directory("test_photos")
    main()
