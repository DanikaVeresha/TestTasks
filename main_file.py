import cv2
import mediapipe as mp
import numpy as np
from log_settings import logger
import os
import glob
import math





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

        # Отримати координату точки на лобі
        forehead_point = np.array([landmarks[10].x, landmarks[10].y])


        # Визначити тип щелепи на основі кута між лініями лівої та правої скроні
        chin_lips_distance = np.linalg.norm(jaw_point - lips) # відстань між підборіддям і губами
        chin_lips_distance_percent = np.linalg.norm(jaw_point - lips) * 100  # відстань між підборіддям і губами в пікселях

        # Відстань між лівою та правою скронями в пікселях
        temple_distance = np.linalg.norm(left_temple - right_temple) # відстань між лівою та правою скронями
        temple_distance_percent = np.linalg.norm(left_temple - right_temple) * 100  # відстань між лівою та правою скронями в пікселях

        # Розрахувати відношення відстані між підборіддям і губами до відстані між лівою та правою скронями
        percent_chip_temple = chin_lips_distance_percent / temple_distance_percent * 100 # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями

        # Отримати скалярний добуток векторів
        left_temple_chin_vector = jaw_point - left_temple
        right_temple_chin_vector = jaw_point - right_temple

        # Визначити кут між лівою скронею і підборіддям
        left_angle = np.arctan2(left_temple_chin_vector[1], left_temple_chin_vector[0]) * 180 / np.pi  # кут між лівою скронею і підборіддям(береться умовна вісь Y від точки лівої скроні до підборіддя, то це буде кут відносно горизонтальної осі тобто відносно осі X)

        # Визначити кут між правою скронею і підборіддям
        right_angle = np.arctan2(right_temple_chin_vector[1], right_temple_chin_vector[0]) * 180 / np.pi  # кут між правою скронею і підборіддям(береться умовна вісь Y від точки правої скроні до підборіддя, то це буде кут відносно горизонтальної осі)


        # Визначити тип щелепи на основі кута
        if percent_chip_temple < 30:
            jaw_type = "wide"
        elif 30 <= percent_chip_temple < 34.7:
            jaw_type = "medium"
        elif percent_chip_temple >= 34.7:
            jaw_type = "narrow"
        else:
            jaw_type = "unknown"

        if left_angle < 47:
            jaw_type_angel = "wide"
        elif left_angle > 50:
            jaw_type_angel = "narrow"
        else:
            jaw_type_angel = "medium"


        return (jaw_type, left_temple, right_temple, jaw_point, lips, forehead_point,
                chin_lips_distance, chin_lips_distance_percent, temple_distance, temple_distance_percent,
                percent_chip_temple, left_angle, right_angle, jaw_type_angel)


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

        jaw_type, left_temple, right_temple, jaw_point, lips, forehead_point, \
        chin_lips_distance, chin_lips_distance_percent, temple_distance, temple_distance_percent, \
        percent_chip_temple, left_angle, right_angle, jaw_type_angel = classifier.classify_jaw(frame_rgb)

        results = classifier.face_mesh.process(frame_rgb)
        frame_with_landmarks = classifier.draw_landmarks(frame, results)

        # Відобразити отримані точки на зображенні та відстані між ними для усіх знайдених облич
        if chin_lips_distance is None:
            logger.error("No face detected")
            continue
        logger.info(f"Result:"
                    f"\n\tJaw Type: {jaw_type}"
                    f"\n\tLeft Temple: {left_temple}"
                    f"\n\tRight Temple: {right_temple}"
                    f"\n\tJaw Point: {jaw_point}"
                    f"\n\tLips: {lips}"
                    f"\n\tForehead Point: {forehead_point}"
                    f"\n\tChin-Lips Distance: {chin_lips_distance:.2f} pixels"
                    f"\n\tChin-Lips Distance Percent: {chin_lips_distance_percent:.2f} pixels"
                    f"\n\tTemple Distance: {temple_distance:.2f} pixels"
                    f"\n\tTemple Distance Percent: {temple_distance_percent:.2f} pixels"
                    f"\n\tPercent: {percent_chip_temple:.2f}%"
                    f"\n\tLeft Angle: {left_angle:.2f} degrees"
                    f"\n\tRight Angle: {right_angle:.2f} degrees"
                    f"\n\tJaw Type (Angel): {jaw_type_angel}\n")


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
                frame_with_landmarks,  f"Jaw Type: % {jaw_type} | ^ {jaw_type_angel} ",  (5, 20), cv2.FONT_HERSHEY_COMPLEX,  0.7,  (6, 8, 10),  2)
            # відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями
            cv2.putText(
                frame_with_landmarks, f"Percentage of distances: {percent_chip_temple:.2f}%", (5, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2)
            cv2.putText(
                frame_with_landmarks, f"Left Angle: {left_angle:.2f}%", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2) # відображення кута між лівою скронею і підборіддям
            cv2.putText(
                frame_with_landmarks, f"Right Angle: {right_angle:.2f}%", (5, 95), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2) # відображення кута між правою скронею і підборіддям


            # Зчитування точності моделі (скор) з файлу result.log
            with open("result.log", "r") as file:
                accuracy_value = []
                accuracy_value_angle = []
                lines = file.readlines()
                accuracy_line = next((line for line in lines if "Total Accuracy" in line), None)
                accuracy_line_angle = next((line for line in lines if "Total Accuracy Angle" in line), None)
                if accuracy_line:
                    accuracy = accuracy_line.split(": ")[1].strip().replace("%", "")
                    accuracy_value.append(accuracy)
                if accuracy_line_angle:
                    accuracy_angle = accuracy_line_angle.split(": ")[1].strip().replace("%", "")
                    accuracy_value_angle.append(accuracy_angle)

            cv2.putText(
                frame_with_landmarks, f"Accuracy: {accuracy_value[-1]}%", (5, 120), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2) # відображення точності моделі
            cv2.putText(
                frame_with_landmarks, f"Accuracy Angle: {accuracy_value_angle[-1]}%", (5, 145), cv2.FONT_HERSHEY_COMPLEX, 0.7, (6, 8, 10), 2) # відображення точності моделі для кута

        else:
            cv2.putText(
                frame_with_landmarks, "No face detected", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 20, 60), 2)

        cv2.imshow(f'Face Mesh Result', frame_with_landmarks)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def classify_images_from_directory_train(directory):
    classifier = FaceMeshClassifier()

    detected_faces = 0
    errors_detected_faces = 0

    valid_detected_faces_angle = 0
    errors_detected_faces_angle = 0

    for image_path in (glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
                       + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.webp"))):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}\n")
                continue

            jaw_type, left_temple, right_temple, jaw_point, lips, forehead_point, \
            chin_lips_distance, chin_lips_distance_percent, temple_distance, temple_distance_percent, \
            percent_chip_temple, left_angle, right_angle, jaw_type_angel = classifier.classify_jaw(image)

            if chin_lips_distance is None:
                logger.error(f"No face detected in image: {image_path}\n")
                detected_faces += 1
                errors_detected_faces += 1
                continue

            # Записуємо результат в лог
            logger.info(f"Image: {image_path}"
                        f"\n\tJaw Type: {jaw_type}"
                        f"\n\tLeft Temple: {left_temple}"
                        f"\n\tRight Temple: {right_temple}"
                        f"\n\tJaw Point: {jaw_point}"
                        f"\n\tLips: {lips}"
                        f"\n\tForehead Point: {forehead_point}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance:.2f} pixels"
                        f"\n\tChin-Lips Distance Percent: {chin_lips_distance_percent:.2f} pixels"
                        f"\n\tTemple Distance: {temple_distance:.2f} pixels"
                        f"\n\tTemple Distance Percent: {temple_distance_percent:.2f} pixels"
                        f"\n\tPercent: {percent_chip_temple:.2f}%"
                        f"\n\tLeft Angle: {left_angle:.2f} degrees"
                        f"\n\tRight Angle: {right_angle:.2f} degrees"
                        f"\n\tJaw Type (Angle): {jaw_type_angel}\n")

            if jaw_type in image_path:
                detected_faces += 1
            else:
                errors_detected_faces += 1

            if jaw_type_angel in image_path:
                valid_detected_faces_angle += 1
            else:
                errors_detected_faces_angle += 1

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

    if valid_detected_faces_angle + errors_detected_faces_angle > 0:
        total_accuracy_angle = (valid_detected_faces_angle / (valid_detected_faces_angle + errors_detected_faces_angle)) * 100
        logger.info(f"Total Accuracy Angle: {total_accuracy_angle:.2f}%\n")
        logger.info(f"Valid classification angle: {valid_detected_faces_angle}; Errors angle: {errors_detected_faces_angle}\n")
    else:
        logger.info("No images processed for angle classification.\n")


def classify_images_from_directory(directory):
    classifier = FaceMeshClassifier()

    for image_path in (glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
                       + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.webp"))):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}\n")
                continue

            jaw_type, left_temple, right_temple, jaw_point, lips, forehead_point, \
            chin_lips_distance, chin_lips_distance_percent, temple_distance, temple_distance_percent, \
            percent_chip_temple, left_angle, right_angle, jaw_type_angel = classifier.classify_jaw(image)

            if chin_lips_distance is None:
                logger.error(f"No face detected in image: {image_path}\n")
                continue

            # Записуємо результат в лог
            logger.info(f"Image: {image_path}"
                        f"\n\tJaw Type: {jaw_type}"
                        f"\n\tLeft Temple: {left_temple}"
                        f"\n\tRight Temple: {right_temple}"
                        f"\n\tJaw Point: {jaw_point}"
                        f"\n\tLips: {lips}"
                        f"\n\tForehead Point: {forehead_point}"
                        f"\n\tChin-Lips Distance: {chin_lips_distance:.2f} pixels"
                        f"\n\tChin-Lips Distance Percent: {chin_lips_distance_percent:.2f} pixels"
                        f"\n\tTemple Distance: {temple_distance:.2f} pixels"
                        f"\n\tTemple Distance Percent: {temple_distance_percent:.2f} pixels"
                        f"\n\tPercent: {percent_chip_temple:.2f}%"
                        f"\n\tLeft Angle: {left_angle:.2f} degrees"
                        f"\n\tRight Angle: {right_angle:.2f} degrees"
                        f"\n\tJaw Type (Angle): {jaw_type_angel}\n")

            results = classifier.face_mesh.process(image) # обробка зображення для отримання результатів
            frame_with_landmarks = classifier.draw_landmarks(image, results) # відображення ключових точок на зображенні
            cv2.imshow(f'TYPE JAW: % {jaw_type} | ^ {jaw_type_angel}', frame_with_landmarks)


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


                # Відобразити тип щелепи
                cv2.putText(
                    frame_with_landmarks, f"Jaw Type: % {jaw_type} | ^ {jaw_type_angel}", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 99, 71), 1)

                # Відобразити відсоток відстані між підборіддям і губами до відстані між лівою та правою скронями
                cv2.putText(
                    frame_with_landmarks, f"Percentage of distances: {percent_chip_temple:.2f}%", (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 99, 71), 1)

                # Відобразити кут між лівою скронею і підборіддям
                cv2.putText(
                    frame_with_landmarks, f"Left Angle: {left_angle:.2f} degrees", (5, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 99, 71), 1)

                # Відобразити кут між правою скронею і підборіддям
                cv2.putText(
                    frame_with_landmarks, f"Right Angle: {right_angle:.2f} degrees", (5, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 99, 71), 1)


            # Зберегти зображення з отриманими ключовими точками та відстанями в директорії "output_images"
            output_directory = "output_images"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_image_path = os.path.join(output_directory, os.path.basename(image_path))
            cv2.imwrite(output_image_path, frame_with_landmarks)
            logger.info(f"Processed image saved to: {output_image_path}\n")

            # Відобразити зображення з ключовими точками та відстанями
            cv2.imshow(f'TYPE JAW: % {jaw_type} | ^ {jaw_type_angel}', frame_with_landmarks)

            # Зачекати 2 секунду перед закриттям вікна
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}\n")


if __name__ == "__main__":
    classify_images_from_directory_train("test_photos")
    classify_images_from_directory("Unknown")
    classify_images_from_directory("New_Unknown")
    main()
