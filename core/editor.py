"""
Функции редактирования чёрно-белых фотографий.

Инструменты:
  Тон:     Яркость/Контраст, Уровни, Кривые, Эквализация гистограммы (CLAHE)
  Детали:  Нерезкое маскирование (USM), Усиление локального контраста
  Шум:     Gaussian, Медианный, NLMeans
  Артист.: Виньетка, Плёночная зернистость
  Обрезка: Поворот, Кадрирование
"""

import numpy as np
import cv2
from scipy import ndimage


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Гарантирует 3-канальный BGR uint8."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Преобразует в одноканальный float32 0–255."""
    if len(image.shape) == 3:
        b, g, r = cv2.split(image.astype(np.float32))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return image.astype(np.float32)


def build_lut(adjust_fn) -> np.ndarray:
    """Строит таблицу замены LUT[0..255] по функции adjust_fn(x)."""
    x = np.arange(256, dtype=np.float32)
    return np.clip(adjust_fn(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Яркость и контраст
# ---------------------------------------------------------------------------

def adjust_brightness_contrast(image: np.ndarray,
                                brightness: int = 0,
                                contrast: int = 0) -> np.ndarray:
    """
    Яркость (−150..+150) и контраст (−100..+100).

    Контраст реализован как масштабирование вокруг средней точки (128):
        new = alpha * (old − 128) + 128 + brightness
    где alpha = (100 + contrast) / 100.
    """
    alpha = (100.0 + contrast) / 100.0
    beta = brightness - (alpha - 1.0) * 128.0

    result = cv2.convertScaleAbs(image.astype(np.float32), alpha=alpha, beta=beta)
    return result


# ---------------------------------------------------------------------------
# Уровни (Levels)
# ---------------------------------------------------------------------------

def adjust_levels(image: np.ndarray,
                  black_in: int = 0,
                  white_in: int = 255,
                  gamma: float = 1.0,
                  black_out: int = 0,
                  white_out: int = 255) -> np.ndarray:
    """
    Уровни: точки входного чёрного/белого, гамма, выходной диапазон.

    black_in, white_in: точки чёрного и белого на входе (0–255).
    gamma: гамма-коррекция (0.1–10.0; 1.0 = без изменений).
    black_out, white_out: диапазон вывода (0–255).
    """
    range_in = max(white_in - black_in, 1)
    range_out = white_out - black_out

    def fn(x):
        v = np.clip((x - black_in) / range_in, 0.0, 1.0)
        if gamma != 1.0:
            v = np.power(v, 1.0 / max(gamma, 0.01))
        return black_out + v * range_out

    lut = build_lut(fn)
    return cv2.LUT(_ensure_bgr(image), lut)


# ---------------------------------------------------------------------------
# Кривые (Tone Curve)
# ---------------------------------------------------------------------------

def adjust_curves(image: np.ndarray,
                  points: list[tuple[int, int]]) -> np.ndarray:
    """
    Кривые тонов через интерполяцию по опорным точкам.

    points: список (input_value, output_value) 0–255.
            Например [(0,0),(64,50),(128,140),(255,255)] — S-кривая.
    """
    if not points or len(points) < 2:
        return image

    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)

    # Добавляем концы если отсутствуют
    if xs[0] > 0:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, 0.0)
    if xs[-1] < 255:
        xs = np.append(xs, 255.0)
        ys = np.append(ys, 255.0)

    xi = np.arange(256, dtype=np.float64)
    yi = np.interp(xi, xs, ys)
    lut = np.clip(yi, 0, 255).astype(np.uint8)

    return cv2.LUT(_ensure_bgr(image), lut)


# ---------------------------------------------------------------------------
# Гистограмма
# ---------------------------------------------------------------------------

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Классическая эквализация гистограммы (глобальная)."""
    gray = cv2.cvtColor(_ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def clahe(image: np.ndarray,
          clip_limit: float = 2.0,
          tile_size: int = 8) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Улучшает локальный контраст без глобальной пересветки.
    clip_limit: ограничение усиления (1.0–8.0).
    tile_size:  размер тайла для адаптивного расчёта (4–16).
    """
    clahe_obj = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size))
    )
    gray = cv2.cvtColor(_ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    result = clahe_obj.apply(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Резкость
# ---------------------------------------------------------------------------

def unsharp_mask(image: np.ndarray,
                 radius: float = 1.5,
                 amount: float = 1.2,
                 threshold: int = 3) -> np.ndarray:
    """
    Нерезкое маскирование (Unsharp Mask / USM).

    Формула: result = original + amount * (original − blurred)
    при условии |original − blurred| > threshold.

    radius:    радиус размытия (0.5–10.0).
    amount:    сила заострения (0.1–5.0).
    threshold: порог (0–50); пиксели с малой разницей не затрагиваются.
    """
    img = _ensure_bgr(image).astype(np.float32)
    sigma = max(radius, 0.1)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    diff = img - blurred

    if threshold > 0:
        mask = (np.abs(diff) >= threshold).astype(np.float32)
        result = img + amount * diff * mask
    else:
        result = img + amount * diff

    return np.clip(result, 0, 255).astype(np.uint8)


def local_contrast(image: np.ndarray,
                   radius: int = 60,
                   strength: float = 0.5) -> np.ndarray:
    """
    Усиление локального контраста (мягкий Clarity / Mid-contrast).

    Усиливает детали без изменения глобальной экспозиции.
    radius:   радиус подложки (10–200).
    strength: сила эффекта (0.1–2.0).
    """
    img = _ensure_bgr(image).astype(np.float32)
    base = cv2.GaussianBlur(img, (0, 0), float(radius))
    local = img - base
    result = img + strength * local
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Шумоподавление
# ---------------------------------------------------------------------------

def denoise_gaussian(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Шумоподавление гауссовым фильтром."""
    return cv2.GaussianBlur(_ensure_bgr(image), (0, 0), float(max(strength, 0.1)))


def denoise_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Шумоподавление медианным фильтром. Хорошо убирает соль-и-перец."""
    ks = int(kernel_size)
    if ks % 2 == 0:
        ks += 1
    ks = max(3, ks)
    return cv2.medianBlur(_ensure_bgr(image), ks)


def denoise_nlmeans(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Нелокальные средние (Non-Local Means).

    Лучший результат для сохранения деталей, но медленнее.
    strength: сила шумоподавления (5–30).
    """
    img = _ensure_bgr(image)
    h = int(np.clip(strength, 1, 50))
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def denoise_bilateral(image: np.ndarray,
                      diameter: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75) -> np.ndarray:
    """
    Билатеральный фильтр.

    Сохраняет края лучше, чем Gaussian, обрабатывая только пиксели
    с похожим цветом.
    diameter:     диаметр окрестности пикселя.
    sigma_color:  дисперсия по цвету (больше → больше цветов смешивается).
    sigma_space:  дисперсия по пространству.
    """
    img = _ensure_bgr(image)
    return cv2.bilateralFilter(img, int(diameter), float(sigma_color), float(sigma_space))


# ---------------------------------------------------------------------------
# Художественные эффекты
# ---------------------------------------------------------------------------

def add_vignette(image: np.ndarray,
                 strength: float = 0.6,
                 radius: float = 0.65,
                 feather: float = 0.3) -> np.ndarray:
    """
    Виньетка.

    strength: интенсивность затемнения по краям (0–1).
    radius:   нормализованный радиус начала виньетки (0–1).
    feather:  мягкость (ширина переходной зоны, 0.1–0.8).
    """
    img = _ensure_bgr(image).astype(np.float32)
    h, w = img.shape[:2]

    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    xn = (X - cx) / (w / 2.0)
    yn = (Y - cy) / (h / 2.0)
    dist = np.sqrt(xn * xn + yn * yn)

    # Плавный переход за пределами radius
    zone = max(feather * 0.5, 0.01)
    t = np.clip((dist - radius) / zone, 0.0, 1.0)
    t_smooth = t * t * (3.0 - 2.0 * t)  # smoothstep

    factor = (1.0 - t_smooth * strength)[:, :, np.newaxis]
    result = img * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def add_film_grain(image: np.ndarray,
                   intensity: float = 15.0,
                   scale: float = 1.0) -> np.ndarray:
    """
    Плёночная зернистость (Film Grain).

    intensity: RMS амплитуда зерна (1–50).
    scale:     масштаб зерна: 1 — однопиксельное, 2 — крупное.
    """
    img = _ensure_bgr(image).astype(np.float32)
    h, w = img.shape[:2]

    if scale <= 1.0:
        grain = np.random.normal(0.0, float(intensity), (h, w)).astype(np.float32)
    else:
        sh, sw = max(1, int(h / scale)), max(1, int(w / scale))
        grain_small = np.random.normal(0.0, float(intensity), (sh, sw)).astype(np.float32)
        grain = cv2.resize(grain_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Монохромное зерно — одинаковое для всех каналов
    grain3 = grain[:, :, np.newaxis]
    result = img + grain3
    return np.clip(result, 0, 255).astype(np.uint8)


def split_toning(image: np.ndarray,
                 shadow_rgb: tuple = (30, 20, 40),
                 highlight_rgb: tuple = (255, 240, 200),
                 shadow_strength: float = 0.3,
                 highlight_strength: float = 0.3,
                 balance: float = 0.0) -> np.ndarray:
    """
    Раздельное тонирование (Split Toning).

    Тени окрашиваются в один цвет, света — в другой.
    Используется поверх Ч/Б для придания атмосферы.

    shadow_rgb/highlight_rgb: цвета тонирования (R, G, B).
    shadow_strength:   сила тонирования теней (0–1).
    highlight_strength: сила тонирования светов (0–1).
    balance:            баланс теней/светов (−1..+1; 0 = симметрично).
    """
    img = _ensure_bgr(image).astype(np.float32)
    b, g, r = cv2.split(img)
    lum = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255) / 255.0

    # Смещение точки разделения
    mid = 0.5 + balance * 0.25
    shadow_w    = np.clip(1.0 - lum / max(mid, 0.01), 0.0, 1.0)
    highlight_w = np.clip((lum - mid) / max(1.0 - mid, 0.01), 0.0, 1.0)

    sc = np.array(shadow_rgb,    dtype=np.float32)  # RGB
    hc = np.array(highlight_rgb, dtype=np.float32)

    # Применяем тонирование
    result = np.zeros_like(img)
    # Преобразуем из RGB → BGR: sc[2]=B, sc[1]=G, sc[0]=R
    s_bgr = [sc[2], sc[1], sc[0]]
    h_bgr = [hc[2], hc[1], hc[0]]

    channels = cv2.split(img)
    for ci in range(3):
        c = channels[ci]
        toned = c + shadow_w * (s_bgr[ci] - c) * shadow_strength + \
                    highlight_w * (h_bgr[ci] - c) * highlight_strength
        result[:, :, ci] = toned

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Геометрические операции
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    """
    Поворот изображения.

    angle:  угол в градусах (против часовой стрелки).
    expand: если True — изображение расширяется чтобы вместить весь поворот.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    if expand:
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        return cv2.warpAffine(image, M, (new_w, new_h),
                              borderMode=cv2.BORDER_REFLECT_101)
    else:
        return cv2.warpAffine(image, M, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)


def flip_image(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Отражение (flip).
    direction: 'horizontal', 'vertical', 'both'.
    """
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    else:
        return cv2.flip(image, -1)


def crop_image(image: np.ndarray,
               x: int, y: int, w: int, h: int) -> np.ndarray:
    """Кадрирование изображения."""
    ih, iw = image.shape[:2]
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return image[y:y+h, x:x+w].copy()


def resize_image(image: np.ndarray,
                 width: int = None,
                 height: int = None,
                 keep_aspect: bool = True) -> np.ndarray:
    """
    Масштабирование изображения.
    Если задан только один размер — пропорционально масштабируется второй.
    """
    h, w = image.shape[:2]
    if width is None and height is None:
        return image

    if keep_aspect:
        if width is None:
            scale = height / h
            width = int(w * scale)
        elif height is None:
            scale = width / w
            height = int(h * scale)
        else:
            scale = min(width / w, height / h)
            width = int(w * scale)
            height = int(h * scale)

    interpolation = cv2.INTER_LANCZOS4 if (width * height > w * h) else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


# ---------------------------------------------------------------------------
# Реестр инструментов для UI
# ---------------------------------------------------------------------------

EDIT_TOOLS: dict[str, dict] = {
    # Тон
    'brightness_contrast': {
        'label':   'Яркость / Контраст',
        'group':   'Тон',
        'fn':      adjust_brightness_contrast,
        'params':  {'brightness': 0, 'contrast': 0},
    },
    'levels': {
        'label':   'Уровни',
        'group':   'Тон',
        'fn':      adjust_levels,
        'params':  {'black_in': 0, 'white_in': 255, 'gamma': 1.0,
                    'black_out': 0, 'white_out': 255},
    },
    'curves': {
        'label':   'Кривые',
        'group':   'Тон',
        'fn':      adjust_curves,
        'params':  {'points': [(0, 0), (255, 255)]},
    },
    'histogram_eq': {
        'label':   'Эквализация гистограммы',
        'group':   'Тон',
        'fn':      histogram_equalization,
        'params':  {},
    },
    'clahe': {
        'label':   'CLAHE (адаптивная эквализация)',
        'group':   'Тон',
        'fn':      clahe,
        'params':  {'clip_limit': 2.0, 'tile_size': 8},
    },
    # Детали
    'unsharp_mask': {
        'label':   'Нерезкое маскирование (USM)',
        'group':   'Детали',
        'fn':      unsharp_mask,
        'params':  {'radius': 1.5, 'amount': 1.2, 'threshold': 3},
    },
    'local_contrast': {
        'label':   'Локальный контраст',
        'group':   'Детали',
        'fn':      local_contrast,
        'params':  {'radius': 60, 'strength': 0.5},
    },
    # Шум
    'denoise_nlmeans': {
        'label':   'NLMeans (нелокальные средние)',
        'group':   'Шум',
        'fn':      denoise_nlmeans,
        'params':  {'strength': 10},
    },
    'denoise_bilateral': {
        'label':   'Билатеральный фильтр',
        'group':   'Шум',
        'fn':      denoise_bilateral,
        'params':  {'diameter': 9, 'sigma_color': 75, 'sigma_space': 75},
    },
    'denoise_gaussian': {
        'label':   'Гауссово размытие',
        'group':   'Шум',
        'fn':      denoise_gaussian,
        'params':  {'strength': 1.5},
    },
    'denoise_median': {
        'label':   'Медианный фильтр',
        'group':   'Шум',
        'fn':      denoise_median,
        'params':  {'kernel_size': 3},
    },
    # Художественные
    'vignette': {
        'label':   'Виньетка',
        'group':   'Художественные',
        'fn':      add_vignette,
        'params':  {'strength': 0.6, 'radius': 0.65, 'feather': 0.3},
    },
    'film_grain': {
        'label':   'Плёночная зернистость',
        'group':   'Художественные',
        'fn':      add_film_grain,
        'params':  {'intensity': 15.0, 'scale': 1.0},
    },
    'split_toning': {
        'label':   'Раздельное тонирование',
        'group':   'Художественные',
        'fn':      split_toning,
        'params':  {'shadow_rgb': (30, 20, 40), 'highlight_rgb': (255, 240, 200),
                    'shadow_strength': 0.3, 'highlight_strength': 0.3, 'balance': 0.0},
    },
    # Геометрия
    'rotate': {
        'label':   'Поворот',
        'group':   'Геометрия',
        'fn':      rotate_image,
        'params':  {'angle': 0.0, 'expand': True},
    },
    'flip': {
        'label':   'Отражение',
        'group':   'Геометрия',
        'fn':      flip_image,
        'params':  {'direction': 'horizontal'},
    },
}
