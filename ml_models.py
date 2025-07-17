# import os
#
# import cv2
# import mediapipe as mp
# import numpy as np
# from log_settings import logger
# import pandas as pd
#
#
#
#
# # Создать модель для обнаружения лиц и получения ключевых точек таких как глаза, нос, рот, виски, челюсти, подбородка, скул и т.д.
#
# class FaceMeshModel:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True, # Статический режим для обработки изображений
#             max_num_faces=5, # Максимальное количество лиц для обнаружения
#             refine_landmarks=True, # Улучшение точности ключевых точек
#             min_detection_confidence=0.5, # Минимальная уверенность в обнаружении лица
#             min_tracking_confidence=0.5 # Минимальная уверенность в отслеживании лица
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#
#
#     def detect_face(self, image):
#         try:
#             results = self.face_mesh.process(image)
#             if results.multi_face_landmarks:
#                 return results.multi_face_landmarks[0]
#             else:
#                 return None
#         except Exception as e:
#             logger.error(f"Error in detect_face: {e}")
#             return None
#
#
#     # Метод для отрисовки ключевых точек на изображении
#     def draw_landmarks(self, image, landmarks):
#         if landmarks is not None:
#             self.mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=landmarks,
#                 connections=self.mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), # Отрисовка ключевых точек зеленым цветом
#                 connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1) # Отрисовка соединений между ключевыми точками красным цветом
#             )
#         return image
#
#
#     # Функция для обработки изображения и получения ключевых точек лица
#     def process_image(image_path):
#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 logger.error(f"Image not found at {image_path}")
#                 return None
#
#             # Преобразование изображения в RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             face_mesh_model = FaceMeshModel()
#             landmarks = face_mesh_model.detect_face(image_rgb) # Обнаружение ключевых точек лица
#
#             if landmarks is not None:
#                 image_with_landmarks = face_mesh_model.draw_landmarks(image.copy(), landmarks) # Отрисовка ключевых точек на изображении
#                 return image_with_landmarks
#             else:
#                 logger.warning("No face detected in the image.")
#                 return None
#         except Exception as e:
#             logger.error(f"Error in process_image: {e}")
#             return None
#
#
#     # Метод для получения координат ключевых точек лица
#     def get_landmark_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         coordinates = []
#         for landmark in landmarks.landmark: # Проходим по всем ключевым точкам
#             coordinates.append((landmark.x, landmark.y, landmark.z)) # координаты (x, y, z) ключевых точек
#         return np.array(coordinates)
#
#
#     # Получить координаты ключевых точек левого виска
#     def get_left_temporal_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         left_temporal_index = 234  # Индекс ключевой точки левого виска
#         left_temporal = landmarks.landmark[left_temporal_index]
#         return (left_temporal.x, left_temporal.y, left_temporal.z)  # Возвращаем координаты (x, y, z) левого виска
#
#
#     # Получить координаты ключевых точек правого виска
#     def get_right_temporal_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         right_temporal_index = 454  # Индекс ключевой точки правого виска
#         right_temporal = landmarks.landmark[right_temporal_index]
#         return (right_temporal.x, right_temporal.y, right_temporal.z)  # Возвращаем координаты (x, y, z) правого виска
#
#
#     # Получить координаты ключевых точек подбородка
#     def get_chin_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         chin_index = 152  # Индекс ключевой точки подбородка
#         chin = landmarks.landmark[chin_index]
#         return (chin.x, chin.y, chin.z)  # Возвращаем координаты (x, y, z) подбородка
#
#
#     # Получить координаты ключевых точек рта
#     def get_mouth_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78] # Индексы ключевых точек губ, которые формируют контур рта
#         mouth_coordinates = []
#         for index in mouth_indices:
#             mouth = landmarks.landmark[index]
#             mouth_coordinates.append((mouth.x, mouth.y, mouth.z))
#         return np.array(mouth_coordinates)
#
#
#     # Получить координаты ключевых точек кончика носа
#     def get_nose_tip_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         nose_tip_index = 1  # Индекс ключевой точки кончика носа
#         nose_tip = landmarks.landmark[nose_tip_index]
#         return (nose_tip.x, nose_tip.y, nose_tip.z)
#
#
#     # Получить координаты ключевых точек глаз
#     def get_eye_coordinates(self, landmarks):
#         if landmarks is None:
#             return None
#
#         left_eye_indices = [33, 160, 158, 133, 153, 144, 145, 163]  # Индексы ключевых точек левого глаза
#         right_eye_indices = [362, 385, 387, 263, 373, 380, 381, 398]  # Индексы ключевых точек правого глаза
#
#         left_eye_coordinates = []
#         for index in left_eye_indices:
#             left_eye = landmarks.landmark[index]
#             left_eye_coordinates.append((left_eye.x, left_eye.y, left_eye.z))
#
#         right_eye_coordinates = []
#         for index in right_eye_indices:
#             right_eye = landmarks.landmark[index]
#             right_eye_coordinates.append((right_eye.x, right_eye.y, right_eye.z))
#
#         return np.array(left_eye_coordinates), np.array(right_eye_coordinates)  # Возвращаем координаты глаз
#
#
#     # Пример использования класса FaceMeshModel с использованием фронтальной камеры компьютера и отрисовкой ключевых точек лица в реальном времени
#     def process_camera_feed(self):
#         cap = cv2.VideoCapture(0)  # Открытие камеры
#
#         # Задать размер окна видео
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Ширина окна видео
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Высота окна видео
#
#
#         if not cap.isOpened():
#             logger.error("Could not open video device")
#             return
#
#         with self.mp_face_mesh.FaceMesh(
#             static_image_mode=False,  # Режим реального времени
#             max_num_faces=5,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         ) as face_mesh:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     logger.error("Failed to capture image")
#                     break
#
#                 # Преобразование изображения в RGB
#                 image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_mesh.process(image_rgb)
#
#                 if results.multi_face_landmarks:
#                     for landmarks in results.multi_face_landmarks:
#                         frame = self.draw_landmarks(frame, landmarks)
#
#                 # Отображение ключевых точек на изображении
#                 cv2.putText(frame, f'Faces detected: {len(results.multi_face_landmarks)}', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 8, 10), 2)
#
#                 # Отобразить точки висков на изображении
#                 left_temporal = self.get_left_temporal_coordinates(landmarks)
#                 right_temporal = self.get_right_temporal_coordinates(landmarks)
#
#                 if left_temporal is not None:
#                     cv2.circle(frame, (int(left_temporal[0] * frame.shape[1]), int(left_temporal[1] * frame.shape[0])), 5, (0, 255, 0), -1)
#
#                 if right_temporal is not None:
#                     cv2.circle(frame, (int(right_temporal[0] * frame.shape[1]), int(right_temporal[1] * frame.shape[0])), 5, (0, 255, 0), -1)
#
#                 # Отобразить точки подбородка на изображении
#                 chin = self.get_chin_coordinates(landmarks)
#
#                 if chin is not None:
#                     cv2.circle(frame, (int(chin[0] * frame.shape[1]), int(chin[1] * frame.shape[0])), 5, (0, 255, 0), -1)
#
#
#                 # Отобразить точки рта на изображении
#                 mouth_coordinates = self.get_mouth_coordinates(landmarks)
#
#                 if mouth_coordinates is not None:
#                     for mouth in mouth_coordinates:
#                         cv2.circle(frame, (int(mouth[0] * frame.shape[1]), int(mouth[1] * frame.shape[0])), 5, (0, 255, 0), -1)
#
#                 # Отобразить точки кончика носа на изображении
#                 nose_tip = self.get_nose_tip_coordinates(landmarks)
#
#                 if nose_tip is not None:
#                     cv2.circle(frame, (int(nose_tip[0] * frame.shape[1]), int(nose_tip[1] * frame.shape[0])), 5, (0, 255, 0), -1)
#
#                 # Отобразить точки глаз на изображении
#                 left_eye_coordinates, right_eye_coordinates = self.get_eye_coordinates(landmarks)
#
#                 if left_eye_coordinates is not None:
#                     for left_eye in left_eye_coordinates:
#                         cv2.circle(frame, (int(left_eye[0] * frame.shape[1]), int(left_eye[1] * frame.shape[0])), 5, (195, 0, 255), -1)
#
#                 if right_eye_coordinates is not None:
#                     for right_eye in right_eye_coordinates:
#                         cv2.circle(frame, (int(right_eye[0] * frame.shape[1]), int(right_eye[1] * frame.shape[0])), 5, (195, 0, 255), -1)
#
#
#                 # Отображение изображения с ключевыми точками
#                 cv2.imshow('Face Mesh', frame)
#                 if cv2.waitKey(5) & 0xFF == 27:
#                     break
#
#         cap.release()
#         cv2.destroyAllWindows()
#
#
#     # Применить клас FaceMeshModel ко всем изображениям из дерриктории и поддерикторий test_photos отобразив на изображениях все ключевые точки
#     # весков, подбородка, рта, кончика носа, глаз и т.д. и вернуть DataFrame с координатами ключевых точек
#     def process_directory_images(self, directory_path):
#         try:
#             data = []  # Список для хранения данных
#
#             for root, _, files in os.walk(directory_path):
#                 for file in files:
#                     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):  # Проверка на изображения
#                         image_path = os.path.join(root, file)
#                         logger.info(f"Processing image: {image_path}")
#
#                         image = cv2.imread(image_path)
#                         if image is None:
#                             logger.error(f"Image not found at {image_path}")
#                             continue
#
#                         # Преобразование изображения в RGB
#                         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#                         landmarks = self.detect_face(image_rgb)  # Обнаружение ключевых точек лица
#
#                         if landmarks is not None:
#                             # Получение координат ключевых точек виска, подбородка, рта, кончика носа и глаз и составить из них датафрейм
#                             left_temporal = self.get_left_temporal_coordinates(landmarks)
#                             right_temporal = self.get_right_temporal_coordinates(landmarks)
#                             chin = self.get_chin_coordinates(landmarks)
#                             mouth_coordinates = self.get_mouth_coordinates(landmarks)
#                             nose_tip = self.get_nose_tip_coordinates(landmarks)
#                             left_eye_coordinates, right_eye_coordinates = self.get_eye_coordinates(landmarks)
#                             landmark_coordinates = self.get_landmark_coordinates(landmarks)
#                             # Добавление данных в список
#                             data.append({
#                                 'left_temporal': left_temporal,
#                                 'right_temporal': right_temporal,
#                                 'chin': chin,
#                                 'mouth_coordinates': mouth_coordinates.tolist(),
#                                 'nose_tip': nose_tip,
#                                 'left_eye_coordinates': left_eye_coordinates.tolist(),
#                                 'right_eye_coordinates': right_eye_coordinates.tolist(),
#                                 'landmark_coordinates': landmark_coordinates.tolist()
#                             })
#                         else:
#                             logger.warning(f"No face detected in image: {image_path}")
#
#
#             # Создание DataFrame из списка данных
#             df = pd.DataFrame(data)
#
#             # Вывести df.info() для получения информации о DataFrame
#             df.info()
#
#             # Обекты landmark_coordinates, mouth_coordinates, left_eye_coordinates и right_eye_coordinates преобразовать в строки
#             df['landmark_coordinates'] = df['landmark_coordinates'].apply(lambda x: ', '.join([f"({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})" for coord in x]))
#             df['mouth_coordinates'] = df['mouth_coordinates'].apply(lambda x: ', '.join([f"({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})" for coord in x]))
#             df['left_eye_coordinates'] = df['left_eye_coordinates'].apply(lambda x: ', '.join([f"({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})" for coord in x]))
#             df['right_eye_coordinates'] = df['right_eye_coordinates'].apply(lambda x: ', '.join([f"({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})" for coord in x]))
#             # Преобразовать координаты висков, подбородка и кончика носа в строки
#             df['left_temporal'] = df['left_temporal'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})" if x is not None else None)
#             df['right_temporal'] = df['right_temporal'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})" if x is not None else None)
#             df['chin'] = df['chin'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})" if x is not None else None)
#             df['nose_tip'] = df['nose_tip'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})" if x is not None else None)
#             # Вывести df.head() для получения первых 5 строк DataFrame
#             logger.info(df.head())  # Вывод первых 5 строк DataFrame
#
#
#
#             return df
#
#         except Exception as e:
#             logger.error(f"Error in process_directory_images: {e}")
#             return pd.DataFrame()
#
#
#
#
#
#
# # Пример использования класса FaceMeshModel
# if __name__ == "__main__":
#     face_mesh_model = FaceMeshModel()
#
#     # Пример обработки видео с камеры
#     face_mesh_model.process_camera_feed()
#
#     # Пример обработки изображений из директории
#     directory_path = 'test_photos'  # Путь к директории с изображениями
#     df = face_mesh_model.process_directory_images(directory_path)
#     if not df.empty:
#         print("DataFrame with landmark coordinates:")
#     else:
#         print("No images processed or no landmarks detected.")
#
