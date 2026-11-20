"""
Алгоритмы тонирования в сепию.

Методы:
  1. Матричное преобразование (классический алгоритм Microsoft/Web)
  2. Сдвиг в HSL/HSV
  3. Сдвиг в YUV/YCbCr
  4. Карта градиента (Gradient Map)
  5. Дуплекс (Duotone)
  6. Сдвиг кривых RGB (Curves)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. Матричное преобразование
# ---------------------------------------------------------------------------

def sepia_matrix(image: np.ndarray) -> np.ndarray:
    """
    Классический алгоритм сепии (Microsoft / CSS-фильтры / Web).

    Применяет матрицу линейного преобразования к исходным R, G, B:
        Rn = 0.393R + 0.769G + 0.189B
        Gn = 0.349R + 0.686G + 0.168B
        Bn = 0.272R + 0.534G + 0.131B

    Одновременно обесцвечивает и окрашивает пиксель.
    Значения > 255 обрезаются.
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    rn = 0.393 * r + 0.769 * g + 0.189 * b
    gn = 0.349 * r + 0.686 * g + 0.168 * b
    bn = 0.272 * r + 0.534 * g + 0.131 * b

    result = cv2.merge([
        np.clip(bn, 0, 255),
        np.clip(gn, 0, 255),
        np.clip(rn, 0, 255),
    ])
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Сдвиг в HSL / HSV
# ---------------------------------------------------------------------------

def sepia_hsl(image: np.ndarray,
              hue: int = 35,
              saturation: float = 0.28,
              bw_method: str = 'rec709') -> np.ndarray:
    """
    Сепия через сдвиг в пространстве HLS.

    Алгоритм:
      1. Конвертируем в Ч/Б.
      2. В пространстве HLS задаём фиксированный тон (Hue ≈ 30–45°,
         оранжево-коричневый спектр) и умеренную насыщенность.

    hue: цветовой тон в градусах (10–60 для сепии).
    saturation: насыщенность 0.0–1.0 (рекомендуется 0.2–0.4).
    """
    gray_bgr = _to_gray_bgr(image, bw_method)

    hls = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_ch, l_ch, s_ch = cv2.split(hls)

    # OpenCV использует Hue в диапазоне 0–180
    h_ch[:] = float(hue) / 2.0

    # Насыщенность пропорциональна яркости (тёмные области остаются ненасыщенными)
    s_ch = l_ch * saturation
    s_ch = np.clip(s_ch, 0, 255)

    hls_new = cv2.merge([h_ch, l_ch, s_ch]).astype(np.uint8)
    return cv2.cvtColor(hls_new, cv2.COLOR_HLS2BGR)


# ---------------------------------------------------------------------------
# 3. Сдвиг в YUV / YCbCr
# ---------------------------------------------------------------------------

def sepia_yuv(image: np.ndarray,
              u_offset: int = -18,
              v_offset: int = 28,
              bw_method: str = 'rec601') -> np.ndarray:
    """
    Сепия через YCbCr: яркость Y остаётся нетронутой,
    хроматическим компонентам Cb и Cr задаются константы.

    u_offset: смещение Cb (аналог U, −128..+127).
    v_offset: смещение Cr (аналог V, −128..+127).

    Типичные значения для сепии: Cb=110, Cr=156 (т.е. offset от 128).
    """
    gray_bgr = _to_gray_bgr(image, bw_method)

    yuv = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    y_ch, cr_ch, cb_ch = cv2.split(yuv)

    cr_ch[:] = np.clip(128 + v_offset, 0, 255)  # Cr ≈ V
    cb_ch[:] = np.clip(128 + u_offset, 0, 255)  # Cb ≈ U

    yuv_new = cv2.merge([y_ch, cr_ch, cb_ch])
    yuv_new = np.clip(yuv_new, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yuv_new, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# 4. Карта градиента (Gradient Map)
# ---------------------------------------------------------------------------

def sepia_gradient_map(image: np.ndarray,
                       shadow_rgb: tuple = (40, 25, 15),
                       midtone_rgb: tuple = (120, 80, 45),
                       highlight_rgb: tuple = (240, 220, 195),
                       bw_method: str = 'rec709') -> np.ndarray:
    """
    Карта градиента — самый «плёночный» способ тонирования.

    Тёмным пикселям (теням) назначается глубокий тёмно-коричневый цвет,
    средним тонам — тёплый охристый или золотистый,
    светлым пикселям (бликам) — кремовый или почти белый.

    Алгоритм интерполирует цвета между этими точками на основе яркости.

    shadow_rgb, midtone_rgb, highlight_rgb: цвета в формате (R, G, B).
    """
    gray_bgr = _to_gray_bgr(image, bw_method)
    b, g, r = cv2.split(gray_bgr.astype(np.float32))
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0  # 0..1

    h, w = lum.shape
    result = np.zeros((h, w, 3), dtype=np.float32)  # RGB

    sc = np.array(shadow_rgb,    dtype=np.float32)
    mc = np.array(midtone_rgb,   dtype=np.float32)
    hc = np.array(highlight_rgb, dtype=np.float32)

    # Тени → средние тона (lum: 0..0.5)
    mask_sm = lum <= 0.5
    t_sm = lum[mask_sm] / 0.5
    for ci, (c1, c2) in enumerate(zip(sc, mc)):
        result[:, :, ci][mask_sm] = c1 + (c2 - c1) * t_sm

    # Средние тона → светА (lum: 0.5..1.0)
    mask_mh = lum > 0.5
    t_mh = (lum[mask_mh] - 0.5) / 0.5
    for ci, (c1, c2) in enumerate(zip(mc, hc)):
        result[:, :, ci][mask_mh] = c1 + (c2 - c1) * t_mh

    # RGB → BGR
    result_bgr = result[:, :, ::-1]
    return np.clip(result_bgr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Дуплекс (Duotone)
# ---------------------------------------------------------------------------

def sepia_duotone(image: np.ndarray,
                  color1_rgb: tuple = (20, 10, 5),
                  color2_rgb: tuple = (200, 155, 95),
                  bw_method: str = 'rec709') -> np.ndarray:
    """
    Дуплекс (Duotone) — полиграфический алгоритм.

    Изображение строится на двух «красках»:
      color1_rgb — цвет теней.
      color2_rgb — цвет светов.

    Для каждой краски своя кривая распределения интенсивности;
    здесь используется линейная интерполяция.
    """
    gray_bgr = _to_gray_bgr(image, bw_method)
    b, g, r = cv2.split(gray_bgr.astype(np.float32))
    lum = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255) / 255.0

    h, w = lum.shape
    result = np.zeros((h, w, 3), dtype=np.float32)  # RGB

    c1 = np.array(color1_rgb, dtype=np.float32)
    c2 = np.array(color2_rgb, dtype=np.float32)

    for ci in range(3):
        result[:, :, ci] = c1[ci] * (1.0 - lum) + c2[ci] * lum

    # RGB → BGR
    result_bgr = result[:, :, ::-1]
    return np.clip(result_bgr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 6. Сдвиг кривых RGB
# ---------------------------------------------------------------------------

def sepia_curves(image: np.ndarray,
                 red_boost: float = 20.0,
                 blue_shadow_drop: float = -30.0,
                 blue_mid_drop: float = -15.0,
                 bw_method: str = 'rec709') -> np.ndarray:
    """
    Сепия через кривые RGB.

    После перевода в Ч/Б:
      • Поднимаем красный канал в средних тонах → тепло.
      • Опускаем синий канал в тенях и средних тонах →
        уменьшение синего добавляет жёлто-коричневую теплоту
        (синий и жёлтый — комплементарные цвета).

    red_boost: максимальный подъём красного (0–50).
    blue_shadow_drop: снижение синего в тенях (отрицательное, −50..0).
    blue_mid_drop: снижение синего в средних тонах (−30..0).
    """
    gray_bgr = _to_gray_bgr(image, bw_method)
    b, g, r = cv2.split(gray_bgr.astype(np.float32))
    gray = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255)
    gray_u8 = gray.astype(np.uint8)

    x = np.arange(256, dtype=np.float32)

    # --- Красный канал: Gaussian-подъём вокруг средних тонов ---
    boost_r = red_boost * np.exp(-((x - 128.0) ** 2) / (2.0 * 60.0 ** 2))
    red_lut = np.clip(x + boost_r, 0, 255).astype(np.uint8)

    # --- Синий канал: снижение в тенях + в средних тонах ---
    drop_shadow = blue_shadow_drop * np.clip(1.0 - x / 110.0, 0, 1)
    drop_mid = blue_mid_drop * np.exp(-((x - 100.0) ** 2) / (2.0 * 50.0 ** 2))
    blue_lut = np.clip(x + drop_shadow + drop_mid, 0, 255).astype(np.uint8)

    r_new = cv2.LUT(gray_u8, red_lut)
    g_new = gray_u8.copy()
    b_new = cv2.LUT(gray_u8, blue_lut)

    return cv2.merge([b_new, g_new, r_new])


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _to_gray_bgr(image: np.ndarray, method: str = 'rec709') -> np.ndarray:
    """
    Конвертирует изображение в серый BGR.
    Используется внутри алгоритмов сепии перед тонированием.
    """
    from core.bw_algorithms import rec601, rec709, rec2100

    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    if method == 'rec601':
        gray_f = 0.299 * r + 0.587 * g + 0.114 * b
    elif method == 'rec2100':
        gray_f = 0.2627 * r + 0.6780 * g + 0.0593 * b
    else:  # rec709 по умолчанию
        gray_f = 0.2126 * r + 0.7152 * g + 0.0722 * b

    gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Реестр методов для UI
# ---------------------------------------------------------------------------

SEPIA_METHODS: dict[str, tuple] = {
    'matrix':        (sepia_matrix,       'Матричное (Microsoft/Web)',
                      'Прямое матричное преобразование RGB.\nОдновременно обесцвечивает и окрашивает.'),
    'hsl':           (sepia_hsl,          'Сдвиг HSL',
                      'Конвертация в Ч/Б, затем задание тона\nи насыщенности в пространстве HLS.'),
    'yuv':           (sepia_yuv,          'Сдвиг YUV (YCbCr)',
                      'Яркость Y нетронута.\nCb и Cr фиксируются как константы.'),
    'gradient_map':  (sepia_gradient_map, 'Карта градиента',
                      'Тени — тёмно-коричневый.\nСредние тона — охра.\nСвета — кремовый.\nСамый "плёночный" результат.'),
    'duotone':       (sepia_duotone,      'Дуплекс (Duotone)',
                      'Два цвета: один для теней,\nдругой для светов.\nПолиграфический эффект.'),
    'curves':        (sepia_curves,       'Сдвиг кривых RGB',
                      'Подъём красного в средних тонах.\nСнижение синего в тенях.\nТонкая настройка тепла.'),
}


def apply_sepia(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Применяет метод тонирования в сепию.

    Args:
        image: BGR uint8 numpy-массив.
        method: ключ из SEPIA_METHODS.
        **kwargs: специфичные параметры метода.

    Returns:
        BGR uint8 numpy-массив.
    """
    if method not in SEPIA_METHODS:
        method = 'matrix'

    fn = SEPIA_METHODS[method][0]

    # Передаём kwargs только те параметры, которые принимает функция
    import inspect
    sig = inspect.signature(fn)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}

    return fn(image, **filtered)


def get_sepia_method_list() -> list[tuple[str, str]]:
    """Возвращает список (ключ, отображаемое имя) для UI."""
    return [(k, v[1]) for k, v in SEPIA_METHODS.items()]


def get_sepia_description(method: str) -> str:
    if method in SEPIA_METHODS:
        return SEPIA_METHODS[method][2]
    return ''
