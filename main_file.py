import cv2
import mediapipe as mp
import numpy as np
from log_settings import logger
from math import atan2, degrees





class FaceMeshClassifier:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3)
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

        # Отримати координати нижньої частини губи
        lips = np.array([landmarks[0].x, landmarks[0].y])

        # Отримати координати ключових точок щелепи
        jaw_left_line = np.array([[landmarks[i].x, landmarks[i].y] for i in range(152, 168)])
        jaw_right_line = np.array([[landmarks[i].x, landmarks[i].y] for i in range(168, 184)])

        # Визначити тип щелепи на основі кута між лініями лівої та правої скроні
        jaw_line = np.concatenate((jaw_left_line, jaw_right_line), axis=0) # об’єднати ліву та праву лінію щелепи
        forehead_point = np.array([landmarks[10].x, landmarks[10].y]) # точка на лобі, яка використовується для визначення кута
        chin_lips_distance = np.linalg.norm(jaw_point - lips) # відстань між підборіддям і губами

        # Розрахувати відношення відстані між підборіддям і губами до відстані між лівою та правою скронями
        chin_lips_distance_percent = np.linalg.norm(jaw_point - lips) * 100 # відстань між підборіддям і губами в пікселях
        temple_distance_percent = np.linalg.norm(left_temple - right_temple) * 100 # відстань між лівою та правою скронями в пікселях
        percent = chin_lips_distance_percent / temple_distance_percent * 100 # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями

        # Визначити тип щелепи на основі кута
        if percent < 30:
            jaw_type = "wide"
        elif 30 <= percent < 34.7:
            jaw_type = "medium"
        elif percent >= 34.7:
            jaw_type = "narrow"
        else:
            jaw_type = "unknown"

        return chin_lips_distance, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips, jaw_line, percent


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
        chin_lips_distance, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips, jaw_line, percent = classifier.classify_jaw(frame_rgb)
        results = classifier.face_mesh.process(frame_rgb)
        frame_with_landmarks = classifier.draw_landmarks(frame, results)

        # Відобразити отримані точки на зображенні та відстані між ними для усіх знайдених облич
        if chin_lips_distance is None:
            logger.error("No face detected")
            continue
        logger.info(f"\nResult:\n\tChin-Lips Distance: {chin_lips_distance:.2f} pixels\n\tJaw Type: {jaw_type}]\n\tPercent: {percent:.2f}%\n")

        if results.multi_face_landmarks:
            cv2.circle(frame_with_landmarks, tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), 5, (0, 255, 0), -1) # ліва скроня
            cv2.circle(frame_with_landmarks, tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), 5, (0, 255, 0), -1) # права скроня
            cv2.circle(frame_with_landmarks, tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), 5, (0, 255, 0), -1) # підборіддя
            cv2.circle(frame_with_landmarks, tuple((int(forehead_point[0] * frame.shape[1]), int(forehead_point[1] * frame.shape[0]))), 5, (0, 255, 0), -1) # лоб
            cv2.circle(frame_with_landmarks, tuple((int(lips[0] * frame.shape[1]), int(lips[1] * frame.shape[0]))), 5, (0, 255, 0), -1) # губи
            cv2.line(frame_with_landmarks, tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), (255, 0, 0), 2) # лінія між скронями
            cv2.line(frame_with_landmarks, tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), tuple((int(lips[0] * frame.shape[1]), int(lips[1] * frame.shape[0]))), (255, 0, 0), 2) # лінія між підборіддям і губами
            cv2.putText(frame_with_landmarks, f"Chin-Lips Distance: {chin_lips_distance:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # відстань між підборіддям і губами
            cv2.putText(frame_with_landmarks, f"Jaw Type: {jaw_type}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # тип щелепи
            cv2.putText(frame_with_landmarks, f"Percent: {percent:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями

            # Зчитування точності з файлу result.log
            with open("result.log", "r") as file:
                accuracy_value = []
                lines = file.readlines()
                accuracy_line = next((line for line in lines if "Total Accuracy" in line), None)
                if accuracy_line:
                    accuracy = accuracy_line.split(": ")[1].strip().replace("%", "")
                    accuracy_value.append(accuracy)


            cv2.putText(frame_with_landmarks, f"Accuracy: {accuracy_value[-1]}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(frame_with_landmarks, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f'Face Mesh Models', frame_with_landmarks)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Проітеруватись по всім зображенням з розширеннями [.jpg; .jpeg; .png; .webp] в директорії та класифікувати їх
def classify_images_from_directory(directory):
    import os
    import glob
    classifier = FaceMeshClassifier()
    accuracy = 0
    errors = 0
    for image_path in (glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
                       + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.webp"))):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                continue

            chin_lips_distance, jaw_type, left_temple, right_temple, jaw_point, forehead_point, lips, jaw_line, percent = classifier.classify_jaw(image)

            if chin_lips_distance is None:
                logger.error(f"No face detected in image: {image_path}")
                errors += 1
                continue

            # Записуємо результат в лог
            logger.info(f"\nImage: {image_path}\n\tChin-Lips Distance: {chin_lips_distance:.2f}\n\tJaw Type: {jaw_type}\n\tPercent: {percent:.2f}%\n")
            accuracy += 1

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            errors += 1


    # Выводим общую статистику по классификации в процентном соотношении
    if accuracy + errors > 0:
        total_accuracy = (accuracy / (accuracy + errors)) * 100
        logger.info(f"Total Accuracy: {total_accuracy:.2f}%\n")
        logger.info(f"Valid classification: {accuracy}; Errors: {errors}\n")
    else:
        logger.info("No images processed.\n")


if __name__ == "__main__":
    classify_images_from_directory("test_photos")
    main()
