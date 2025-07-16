import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,  # Уровень логирования
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Формат логов
    datefmt='%Y-%m-%d %H:%M:%S',  # Формат даты и времени
    handlers=[
        logging.FileHandler(f"result.log"),  # Запись логов в файл
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)

logger = logging.getLogger(__name__)
