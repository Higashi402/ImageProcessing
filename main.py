import cv2
import numpy as np
from tkinter import filedialog, Tk
import os


def median_filter(image_array, kernel_size=3):
    #Получаем размеры изображения
    height, width = image_array.shape[:2]
    #Определяем границы окрестности
    pad_size = kernel_size // 2
    #Добавляем отступы, чтобы избежать выхода за границы
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    #Создаём пустой массив для результата
    result = np.zeros((height, width, 3), dtype=np.uint8)

    #Применяем фильтр для каждого пикселя
    for i in range(height):
        for j in range(width):
            #Извлекаем окно окрестности
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            #Вычисляем медиану для каждого канала цвета
            for k in range(3):  # Каналы R, G и B
                result[i, j, k] = np.median(region[:, :, k])
            print("Current pixel: " + i.__str__() + " : " + j.__str__())

    return result


def mean_filter(image_array, kernel_size=3):
    # Получаем размеры изображения
    height, width = image_array.shape[:2]
    # Определяем границы окрестности
    pad_size = kernel_size // 2
    # Добавляем отступы, чтобы избежать выхода за границы
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    # Создаём пустой массив для результата
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Применяем фильтр для каждого пикселя
    for i in range(height):
        for j in range(width):
            # Извлекаем окно окрестности
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Вычисляем среднее значение для каждого канала цвета
            for k in range(3):  # Каналы R, G и B
                result[i, j, k] = np.mean(region[:, :, k])
            print("Current pixel: " + i.__str__() + " : " + j.__str__())

    return result


def gaussian_kernel(size, sigma=1):
    # Создаём пустое ядро Гаусса
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # Заполняем ядро значениями по формуле Гаусса
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Нормализуем ядро, чтобы сумма его элементов была 1
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(image_array, kernel_size=5, sigma=1):
    # Генерируем ядро Гаусса
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    height, width = image_array.shape[:2]
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Применяем фильтр Гаусса к каждому пикселю
    for i in range(height):
        for j in range(width):
            for k in range(3):  # Для каждого цветового канала
                region = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                result[i, j, k] = np.sum(region * kernel)
            print("Current pixel: " + i.__str__() + " : " + j.__str__())
    return result

def sobel_edge_detection(image_array):
    # Преобразуем изображение в оттенки серого
    height, width = image_array.shape[:2]
    gray_image = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Ядра Собеля
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)

    # Пустое изображение для результата
    result = np.zeros((height, width), dtype=np.uint8)

    # Применяем ядра Собеля
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = gray_image[i - 1:i + 2, j - 1:j + 2]
            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)
            gradient_magnitude = min(255, int(np.sqrt(gx**2 + gy**2)))
            result[i, j] = gradient_magnitude
            print("Current pixel: " + i.__str__() + " : " + j.__str__())
    return result


def apply_contrast(image_array, alpha):
    # Получаем размеры изображения
    height, width, channels = image_array.shape

    # Создаем пустое изображение для хранения результата
    contrasted_image = np.zeros_like(image_array, dtype=np.uint8)

    # Перебираем каждый пиксель изображения
    for y in range(height):
        for x in range(width):
            # Извлекаем пиксель и преобразуем его в int для корректной работы
            pixel = image_array[y, x].astype(int)

            # Применяем формулу контрастирования для каждого канала (BGR)
            blue = int(alpha * (pixel[0] - 128) + 128)
            green = int(alpha * (pixel[1] - 128) + 128)
            red = int(alpha * (pixel[2] - 128) + 128)

            # Ограничиваем значения пикселей в диапазоне [0, 255] с помощью np.clip
            blue = np.clip(blue, 0, 255)
            green = np.clip(green, 0, 255)
            red = np.clip(red, 0, 255)

            # Записываем контрастированное значение в новый массив
            contrasted_image[y, x] = [blue, green, red]

    return contrasted_image.astype(np.uint8)


def open_image_and_apply_filter():
    # Открываем диалог для выбора файла
    root = Tk()
    root.withdraw()  # Скрываем главное окно Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not file_path:
        print("Файл не выбран.")
        return

    #Загружаем изображение
    image = cv2.imread(file_path)

    # Применяем фильтр Гаусса
    #blurred_image = apply_gaussian_filter(image, kernel_size=5, sigma=1)

    # Применяем оператор Собеля для выделения контуров
    #filtered_image_array = sobel_edge_detection(blurred_image)

    filtered_image_array = apply_contrast(image, 1.5)

    #Сохраняем изображение
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_filtered{ext}"
    cv2.imwrite(output_path, filtered_image_array)
    print(f"Фильтрованное изображение сохранено по пути: {output_path}")

if __name__ == '__main__':
    open_image_and_apply_filter()