import cv2
import mediapipe as mp
import numpy as np
from log_settings import logger
import os
import glob





class FaceMeshClassifier:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils

    def classify_jaw(self, image):
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return "No face detected"

        # Отримати ключові точки обличчя
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

        # Визначити градус кута між лініями котрі проходять через ліву скроню і підборіддям
        left_temple_chin_vector = jaw_point - left_temple # вектор від лівої скроні до підборіддя
        right_temple_chin_vector = jaw_point - right_temple # вектор від правої скроні до підборіддя
        left_angle = np.arctan2(left_temple_chin_vector[1], left_temple_chin_vector[0]) * 180 / np.pi # кут між лівою скронею і підборіддям
        right_angle = np.arctan2(right_temple_chin_vector[1], right_temple_chin_vector[0]) * 180 / np.pi # кут між правою скронею і підборіддям

        # Розрахувати відношення відстані між підборіддям і губами до відстані між лівою та правою скронями
        chin_lips_distance_percent = np.linalg.norm(jaw_point - lips) * 100 # відстань між підборіддям і губами в пікселях
        temple_distance_percent = np.linalg.norm(left_temple - right_temple) * 100 # відстань між лівою та правою скронями в пікселях
        percent = chin_lips_distance_percent / temple_distance_percent * 100 # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями

        different_angel_left = abs(90 - left_angle) # різниця між 90 градусами і кутом між лівою скронею і підборіддям
        different_angel_right = abs(90 - right_angle) # різниця між 90 градусами і кутом між правою скронею і підборіддям
        abs_different_angle = sum([different_angel_left, different_angel_right]) # сума різниці між 90 градусами і кутами між лівою та правою скронями і підборіддям


        # Визначити тип щелепи на основі кута
        if percent < 30 or abs_different_angle >= 83.5:
            jaw_type = "wide"
        elif 30 <= percent < 34.7 or 80 <= abs_different_angle < 83.5:
            jaw_type = "medium"
        elif percent >= 34.7 or abs_different_angle < 80:
            jaw_type = "narrow"
        else:
            jaw_type = "unknown"


        return (jaw_type, # тип щелепи
                chin_lips_distance, # відстань між підборіддям і губами
                chin_lips_distance_percent, # відстань між підборіддям і губами в пікселях
                left_temple, # координати лівої скроні
                right_temple, # координати правої скроні
                temple_distance_percent, # відстань між лівою та правою скронями в пікселях
                jaw_point, # координати підборіддя
                forehead_point, # координати лобу
                lips, # координати губ
                percent, # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями
                left_angle, # кут між лівою скронею і підборіддям
                right_angle, # кут між правою скронею і підборіддям
                left_temple_chin_vector, # вектор від лівої скроні до підборіддя
                right_temple_chin_vector,  # вектор від правої скроні до підборіддя
                different_angel_left, # різниця між 90 градусами і кутом між лівою скронею і підборіддям
                different_angel_right,  # різниця між 90 градусами і кутом між правою скронею і підборіддям
                abs_different_angle) # сума різниці між 90 градусами і кутами між лівою та правою скронями і підборіддям


    def draw_landmarks(self, image, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1)
                )
        return image


def main():
    classifier = FaceMeshClassifier()
    cap = cv2.VideoCapture(0)

    # Задать размер окна
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (jaw_type, chin_lips_distance,
         chin_lips_distance_percent, left_temple, right_temple, temple_distance_percent,
         jaw_point, forehead_point, lips, percent,
         left_angle, right_angle, left_temple_chin_vector, right_temple_chin_vector,
        different_angel_left, different_angel_right, abs_different_angle) = classifier.classify_jaw(frame_rgb)

        results = classifier.face_mesh.process(frame_rgb)
        frame_with_landmarks = classifier.draw_landmarks(frame, results)

        # Відобразити отримані точки на зображенні та відстані між ними для усіх знайдених облич
        if chin_lips_distance is None:
            logger.error("No face detected")
            continue
        logger.info(f"Result:"
                    f"\n\tChin-Lips Distance: {chin_lips_distance:.2f}"
                    f"\n\tChin-Lips Distance Percent: {chin_lips_distance_percent:.2f} pixels"
                    f"\n\tTemple Distance Percent: {temple_distance_percent:.2f} pixels"
                    f"\n\tPercent: {percent:.2f}%"
                    f"\n\tLeft Angle: {left_angle:.2f} degrees"
                    f"\n\tRight Angle: {right_angle:.2f} degrees"
                    f"\n\tDifferent Angle Left: {different_angel_left:.2f} degrees"
                    f"\n\tDifferent Angle Right: {different_angel_right:.2f} degrees"
                    f"\n\tAbs Different Angle: {abs_different_angle:.2f} degrees\n")

        if results.multi_face_landmarks:
            cv2.circle(frame_with_landmarks,
                       tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))), 5, (173, 255, 47), -1) # точка лівої скроні
            cv2.circle(frame_with_landmarks,
                       tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), 5, (173, 255, 47), -1) # точка правої скроні
            cv2.circle(frame_with_landmarks,
                       tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), 5, (173, 255, 47), -1) # точка підборіддя
            cv2.circle(frame_with_landmarks,
                       tuple((int(forehead_point[0] * frame.shape[1]), int(forehead_point[1] * frame.shape[0]))), 5, (173, 255, 47), -1) # точка лоба
            cv2.circle(frame_with_landmarks,
                       tuple((int(lips[0] * frame.shape[1]), int(lips[1] * frame.shape[0]))), 5, (173, 255, 47), -1) # точка губ

            cv2.line(frame_with_landmarks,
                     tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))),
                     tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))), (238, 130, 238), 2) # лінія між скронями
            cv2.line(frame_with_landmarks,
                     tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))),
                     tuple((int(lips[0] * frame.shape[1]), int(lips[1] * frame.shape[0]))), (238, 130, 238), 2) # лінія між підборіддям і губами

            # Відобразити вектор між лівою скронею і підборіддям
            cv2.arrowedLine(frame_with_landmarks,
                            tuple((int(left_temple[0] * frame.shape[1]), int(left_temple[1] * frame.shape[0]))),
                            tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), (220, 20, 60), 2)
            # Відобразити вектор між правою скронею і підборіддям
            cv2.arrowedLine(frame_with_landmarks,
                            tuple((int(right_temple[0] * frame.shape[1]), int(right_temple[1] * frame.shape[0]))),
                            tuple((int(jaw_point[0] * frame.shape[1]), int(jaw_point[1] * frame.shape[0]))), (220, 20, 60), 2)

            # тип щелепи
            cv2.putText(
                frame_with_landmarks,  f"Jaw Type: {jaw_type}",  (5, 20), cv2.FONT_HERSHEY_COMPLEX,  0.7,  (6, 8, 10),  2)
            # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями
            cv2.putText(
                frame_with_landmarks, f"Percentage of distances: {percent:.2f}%", (5, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)
            cv2.putText(
                frame_with_landmarks, f"Abs(different angle): {abs_different_angle:.2f} degrees", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)

            # Зчитування точності моделі (скор) з файлу result.log
            with open("result.log", "r") as file:
                accuracy_value = []
                lines = file.readlines()
                accuracy_line = next((line for line in lines if "Total Accuracy" in line), None)
                if accuracy_line:
                    accuracy = accuracy_line.split(": ")[1].strip().replace("%", "")
                    accuracy_value.append(accuracy)

            cv2.putText(frame_with_landmarks, f"Accuracy: {accuracy_value[-1]}%", (2, 95), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)

        else:
            cv2.putText(frame_with_landmarks, "No face detected", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 20, 60), 2)

        cv2.imshow(f'Face Mesh Result', frame_with_landmarks)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def classify_images_from_directory_train(directory):
    classifier = FaceMeshClassifier()
    detected_faces = 0
    errors_detected_faces = 0

    for image_path in (glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
                       + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.webp"))):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}\n")
                continue

            (jaw_type, chin_lips_distance, chin_lips_distance_percent,
            left_temple, right_temple, temple_distance_percent,
            jaw_point, forehead_point, lips, percent, left_angle,
            right_angle, left_temple_chin_vector, right_temple_chin_vector,
            different_angel_left, different_angel_right, abs_different_angle) = classifier.classify_jaw(image)

            if chin_lips_distance is None:
                logger.error(f"No face detected in image: {image_path}\n")
                detected_faces += 1
                continue

            # Записуємо результат в лог
            logger.info(f"Image: {image_path}"
                        f"\n\tJaw Type: {jaw_type}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance:.2f}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance_percent:.2f} pixels"
                        f"\n\tTemple Distance: {temple_distance_percent:.2f} pixels"
                        f"\n\tPercent: {percent:.2f}%"
                        f"\n\tLeft Angle: {left_angle:.2f} degrees"
                        f"\n\tRight Angle: {right_angle:.2f} degrees"
                        f"\n\tDifferent Angle Left: {different_angel_left:.2f} degrees"
                        f"\n\tDifferent Angle Right: {different_angel_right:.2f} degrees"
                        f"\n\tAbs Different Angle: {abs_different_angle:.2f} degrees\n")

            if jaw_type in image_path:
                detected_faces += 1
            else:
                errors_detected_faces += 1

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}\n")
            detected_faces += 1


    # Выводим общую статистику по классификации в процентном соотношении
    if detected_faces + errors_detected_faces > 0:
        total_accuracy = (detected_faces / (detected_faces + errors_detected_faces)) * 100
        logger.info(f"Total Accuracy: {total_accuracy:.2f}%\n")
        logger.info(f"Valid classification: {detected_faces}; Errors: {errors_detected_faces}\n")
    else:
        logger.info("No images processed.\n")


def classify_images_from_directory(directory):
    classifier = FaceMeshClassifier()

    for image_path in (glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
                       + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.webp"))):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}\n")
                continue

            (jaw_type, chin_lips_distance, chin_lips_distance_percent,
            left_temple, right_temple, temple_distance_percent,
            jaw_point, forehead_point, lips, percent, left_angle,
            right_angle, left_temple_chin_vector, right_temple_chin_vector,
            different_angel_left, different_angel_right, abs_different_angle) = classifier.classify_jaw(image)

            if chin_lips_distance is None:
                logger.error(f"No face detected in image: {image_path}\n")
                continue

            # Записуємо результат в лог
            logger.info(f"Image: {image_path}"
                        f"\n\tJaw Type: {jaw_type}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance:.2f}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance_percent:.2f} pixels"
                        f"\n\tTemple Distance: {temple_distance_percent:.2f} pixels"
                        f"\n\tPercent: {percent:.2f}%"
                        f"\n\tLeft Angle: {left_angle:.2f} degrees"
                        f"\n\tRight Angle: {right_angle:.2f} degrees"
                        f"\n\tDifferent Angle Left: {different_angel_left:.2f} degrees"
                        f"\n\tDifferent Angle Right: {different_angel_right:.2f} degrees"
                        f"\n\tAbs Different Angle: {abs_different_angle:.2f} degrees\n")

            results = classifier.face_mesh.process(image) # обробка зображення для отримання результатів
            frame_with_landmarks = classifier.draw_landmarks(image, results) # відображення ключових точок на зображенні
            cv2.imshow(f'TYPE JAW:  {jaw_type}', frame_with_landmarks)


            # Відобразити отримані точки на зображенні та відстані між ними для усіх знайдених облич
            if results.multi_face_landmarks:
                # точка лівої скроні
                cv2.circle(
                    frame_with_landmarks, tuple((int(left_temple[0] * image.shape[1]), int(left_temple[1] * image.shape[0]))), 5, (173, 255, 47), -1)
                # точка правої скроні
                cv2.circle(
                    frame_with_landmarks, tuple((int(right_temple[0] * image.shape[1]), int(right_temple[1] * image.shape[0]))), 5, (173, 255, 47), -1)
                # точка підборіддя
                cv2.circle(
                    frame_with_landmarks, tuple((int(jaw_point[0] * image.shape[1]), int(jaw_point[1] * image.shape[0]))), 5, (173, 255, 47), -1)
                # точка лоба
                cv2.circle(
                    frame_with_landmarks, tuple((int(forehead_point[0] * image.shape[1]), int(forehead_point[1] * image.shape[0]))), 5, (173, 255, 47), -1)
                # точка губ
                cv2.circle(
                    frame_with_landmarks, tuple((int(lips[0] * image.shape[1]), int(lips[1] * image.shape[0]))), 5, (173, 255, 47), -1)


                # Выдобразити лінію ктора проходить через ліву і праву скроні
                cv2.line(
                    frame_with_landmarks,
                     tuple((int(left_temple[0] * image.shape[1]), int(left_temple[1] * image.shape[0]))),
                     tuple((int(right_temple[0] * image.shape[1]), int(right_temple[1] * image.shape[0]))), (238, 130, 238), 2)

                # Выдобразити лінію ктора проходить через підборіддя і губи
                cv2.line(
                    frame_with_landmarks,
                    tuple((int(jaw_point[0] * image.shape[1]), int(jaw_point[1] * image.shape[0]))),
                    tuple((int(lips[0] * image.shape[1]), int(lips[1] * image.shape[0]))), (238, 130, 238), 2)

                # Відобразити вектор між лівою скронею і підборіддям
                cv2.arrowedLine(
                    frame_with_landmarks,
                    tuple((int(left_temple[0] * image.shape[1]), int(left_temple[1] * image.shape[0]))),
                    tuple((int(jaw_point[0] * image.shape[1]), int(jaw_point[1] * image.shape[0]))), (220, 20, 60), 2)

                # Відобразити вектор між правою скронею і підборіддям
                cv2.arrowedLine(
                    frame_with_landmarks,
                    tuple((int(right_temple[0] * image.shape[1]), int(right_temple[1] * image.shape[0]))),
                    tuple((int(jaw_point[0] * image.shape[1]), int(jaw_point[1] * image.shape[0]))), (220, 20, 60), 2)


                # Відобразити сумму різниці між 90 градусами і кутами між лівою та правою скронями і підборіддям
                cv2.putText(
                    frame_with_landmarks, f"Abs Difference Angle: {round(abs_different_angle, 2)}", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)
                # Відобразити тип щелепи
                cv2.putText(
                    frame_with_landmarks, f"Jaw Type: {jaw_type}", (5, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)


            # Зберегти зображення з отриманими ключовими точками та відстанями в директорії "output_images"
            output_directory = "output_images"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_image_path = os.path.join(output_directory, os.path.basename(image_path))
            cv2.imwrite(output_image_path, frame_with_landmarks)
            logger.info(f"Processed image saved to: {output_image_path}\n")

            # Визначити розмір вікна для відображення зображення
            cv2.namedWindow(f'TYPE JAW:  {jaw_type}', cv2.WINDOW_NORMAL) #
            cv2.resizeWindow(f'TYPE JAW:  {jaw_type}', 1200, 720)  # Задати розмір вікна
            # Відобразити зображення з ключовими точками та відстанями
            cv2.imshow(f'TYPE JAW:  {jaw_type}', frame_with_landmarks)

            # Зачекати 2 секунду перед закриттям вікна
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}\n")


if __name__ == "__main__":
    classify_images_from_directory_train("test_photos")
    classify_images_from_directory("Unknown")
    main()
