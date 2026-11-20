"""
Алгоритмы перевода цветных фотографий в чёрно-белый формат.

Категории:
  1. Простые математические методы
  2. Перцептивные методы (физиология зрения)
  3. Продвинутые и художественные методы
"""

import numpy as np
import cv2
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Простые математические методы
# ---------------------------------------------------------------------------

def average_method(image: np.ndarray) -> np.ndarray:
    """
    Усреднение (Average): Y = (R + G + B) / 3

    Самый примитивный метод. Не учитывает особенности человеческого зрения.
    Результат часто выглядит плоским и невыразительным.
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return (r + g + b) / 3.0


def lightness_method(image: np.ndarray) -> np.ndarray:
    """
    Метод Lightness (Светлота): Y = (max(R,G,B) + min(R,G,B)) / 2

    Вычисляет среднее между самым ярким и самым тёмным каналом.
    Используется при переводе через цветовое пространство HSL.
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    return (max_val + min_val) / 2.0


def value_method(image: np.ndarray) -> np.ndarray:
    """
    Метод Value / Brightness (Значение / Яркость): Y = max(R, G, B)

    Берёт значение самого яркого канала. Используется в модели HSV.
    Снимок часто получается пересвеченным.
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return np.maximum(np.maximum(r, g), b)


def single_channel(image: np.ndarray, channel: str = 'G') -> np.ndarray:
    """
    Один канал (Single Channel).

    Зелёный канал ('G') наиболее близок к восприятию яркости глазом
    и содержит меньше всего цифрового шума.

    channel: 'R', 'G' или 'B'
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    ch_map = {'R': r, 'G': g, 'B': b}
    return ch_map.get(channel.upper(), g).copy()


# ---------------------------------------------------------------------------
# 2. Перцептивные методы
# ---------------------------------------------------------------------------

def rec601(image: np.ndarray) -> np.ndarray:
    """
    Rec. 601 (стандарт SDTV): Y = 0.299R + 0.587G + 0.114B

    Классическая формула, используемая по умолчанию во многих
    базовых функциях (OpenCV, Photoshop Grayscale).
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return 0.299 * r + 0.587 * g + 0.114 * b


def rec709(image: np.ndarray) -> np.ndarray:
    """
    Rec. 709 (стандарт HDTV и sRGB): Y = 0.2126R + 0.7152G + 0.0722B

    Более точная формула для современных мониторов.
    Сильнее повышает вклад зелёного канала.
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rec2100(image: np.ndarray) -> np.ndarray:
    """
    Rec. 2100 (стандарт HDR): Y = 0.2627R + 0.6780G + 0.0593B

    Используется для HDR-контента (Nikon Z6 III поддерживает HDR N-RAW).
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return 0.2627 * r + 0.6780 * g + 0.0593 * b


# ---------------------------------------------------------------------------
# 3. Продвинутые и художественные методы
# ---------------------------------------------------------------------------

def channel_mixer(image: np.ndarray,
                  wr: float = 0.33,
                  wg: float = 0.33,
                  wb: float = 0.34) -> np.ndarray:
    """
    Микширование каналов (Channel Mixer): Y = wr*R + wg*G + wb*B

    Позволяет имитировать цветные фотографические светофильтры:
      - Красный фильтр (wr=0.9, wg=0.1, wb=0.0): светлое небо → тёмное, кожа светлее
      - Жёлтый фильтр (wr=0.5, wg=0.4, wb=0.1): умеренное затемнение синего
      - Зелёный фильтр (wr=0.1, wg=0.7, wb=0.2): светлая листва, тёмная кожа
      - Синий фильтр  (wr=0.1, wg=0.1, wb=0.8): светлое небо, тёмные тёплые тона
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    return wr * r + wg * g + wb * b


def lab_lightness(image: np.ndarray) -> np.ndarray:
    """
    Канал L из пространства CIE Lab.

    Перцептивно точная яркость, полностью отделённая от цвета.
    Является «чистым» представлением того, что видит глаз.
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    L, _a, _b = cv2.split(lab)
    # В OpenCV канал L масштабирован 0–255 (соответствует 0–100 в CIE)
    return L.astype(np.float32)


def adaptive_decolorize(image: np.ndarray) -> np.ndarray:
    """
    Адаптивное преобразование с сохранением контраста.

    Упрощённая реализация принципов алгоритма Grundland & Dodgson (Decolorize, 2007).
    Анализирует локальный цветовой контраст: если рядом два пикселя разного цвета,
    но одинаковой яркости, алгоритм сдвигает их яркости для сохранения видимых границ.

    Шаги:
      1. Вычисляем базовую яркость (Rec.709).
      2. Оцениваем «цветовой градиент» (разница в цветах соседних пикселей).
      3. Там, где цветной градиент значительно превышает градиент в оттенках серого,
         искусственно повышаем контраст.
    """
    img_f = image.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = cv2.split(img_f)

    # Базовая яркость Rec.709
    lum = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch

    # Градиенты по каждому каналу (абсолютный)
    def _grad(ch):
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    color_grad = np.sqrt(_grad(r_ch)**2 + _grad(g_ch)**2 + _grad(b_ch)**2)
    lum_grad   = _grad(lum)

    # «Потерянный» цветовой контраст
    lost = np.clip(color_grad - lum_grad, 0, None)
    lost_smooth = ndimage.gaussian_filter(lost, sigma=1.5)

    # Направление коррекции: «доминирование» красного или синего канала
    hue_sign = np.sign(r_ch - b_ch + 0.5 * (g_ch - b_ch))

    max_lost = np.percentile(lost_smooth, 99)
    if max_lost > 1e-6:
        lost_norm = lost_smooth / max_lost
    else:
        lost_norm = lost_smooth

    correction = lost_norm * hue_sign * 0.12  # Максимум 12% коррекции
    adjusted = np.clip(lum + correction, 0.0, 1.0)

    return adjusted * 255.0


# ---------------------------------------------------------------------------
# Универсальная функция применения алгоритма
# ---------------------------------------------------------------------------

# Словарь: ключ → (функция, label, описание)
BW_METHODS: dict[str, tuple] = {
    # --- Простые ---
    'average':    (average_method,    'Усреднение (Average)',
                   'Y = (R + G + B) / 3\nСамый простой метод. Плоский результат.'),
    'lightness':  (lightness_method,  'Светлота (Lightness)',
                   'Y = (max + min) / 2\nИспользуется в модели HSL.'),
    'value':      (value_method,      'Яркость (Value/HSV)',
                   'Y = max(R, G, B)\nМодель HSV. Часто даёт пересвет.'),
    'channel_r':  (lambda img: single_channel(img, 'R'), 'Красный канал (R)',
                   'Используется только красный канал.\nЭффект красного светофильтра.'),
    'channel_g':  (lambda img: single_channel(img, 'G'), 'Зелёный канал (G)',
                   'Используется только зелёный канал.\nМинимум шума, близко к яркости.'),
    'channel_b':  (lambda img: single_channel(img, 'B'), 'Синий канал (B)',
                   'Используется только синий канал.\nЭффект синего светофильтра.'),
    # --- Перцептивные ---
    'rec601':     (rec601,  'Rec. 601 (SDTV)',
                   'Y = 0.299R + 0.587G + 0.114B\nКлассика: OpenCV, Photoshop.'),
    'rec709':     (rec709,  'Rec. 709 (HDTV / sRGB)',
                   'Y = 0.2126R + 0.7152G + 0.0722B\nСовременные мониторы.'),
    'rec2100':    (rec2100, 'Rec. 2100 (HDR)',
                   'Y = 0.2627R + 0.6780G + 0.0593B\nДля HDR-контента.'),
    # --- Продвинутые ---
    'channel_mixer': (None, 'Микширование каналов',
                      'Y = wr·R + wg·G + wb·B\nИмитация светофильтров.'),
    'lab':        (lab_lightness, 'CIE Lab (канал L)',
                   'Перцептивная яркость из Lab.\nЧистое разделение цвета и яркости.'),
    'adaptive':   (adaptive_decolorize, 'Адаптивный (Decolorize)',
                   'Сохраняет локальный контраст цветов.\nПредотвращает слияние близких цветов.'),
}


def apply_bw(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Применяет алгоритм Ч/Б конвертации к изображению.

    Args:
        image: BGR uint8 numpy-массив.
        method: ключ из BW_METHODS или 'channel_mixer'.
        **kwargs: для 'channel_mixer' передаём wr, wg, wb.

    Returns:
        BGR uint8 numpy-массив (серый, но 3-канальный для совместимости).
    """
    if method == 'channel_mixer':
        wr = float(kwargs.get('wr', 0.33))
        wg = float(kwargs.get('wg', 0.33))
        wb = float(kwargs.get('wb', 0.34))
        gray_f = channel_mixer(image, wr, wg, wb)
    elif method in BW_METHODS and BW_METHODS[method][0] is not None:
        gray_f = BW_METHODS[method][0](image)
    else:
        gray_f = rec709(image)

    gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def get_method_list() -> list[tuple[str, str]]:
    """Возвращает список (ключ, отображаемое имя) для UI."""
    return [(k, v[1]) for k, v in BW_METHODS.items()]


def get_method_description(method: str) -> str:
    """Возвращает описание метода для подсказки."""
    if method in BW_METHODS:
        return BW_METHODS[method][2]
    return ''


# --- Пресеты светофильтров для Channel Mixer ---
FILTER_PRESETS = {
    'Нейтральный (Rec.709)': (0.213, 0.715, 0.072),
    'Красный фильтр':        (0.80, 0.15, 0.05),
    'Оранжевый фильтр':      (0.60, 0.33, 0.07),
    'Жёлтый фильтр':         (0.45, 0.42, 0.13),
    'Зелёный фильтр':        (0.08, 0.72, 0.20),
    'Синий фильтр':          (0.06, 0.09, 0.85),
    'Инфракрасный (ИК)':     (0.72, 0.28, 0.00),
    'Портрет (кожа)':        (0.50, 0.40, 0.10),
}
