"""
Инфракрасная фотография: обработка и конвертация.

Категории:
  1. Конвертация ложноцветных ИК-снимков в Ч/Б
  2. Классические ИК-эффекты (симуляция)
  3. Художественные ИК-эффекты
  4. Тонирование ИК-снимков

Контекст:
  Nikon Z6 III — сенсор со стандартным горячим зеркалом (hot mirror).
  Для настоящей ИК-съёмки требуется:
    а) Внешний ИК-фильтр (R72, 850nm и т.д.) + длинная выдержка
    б) Аппаратная конвертация камеры (снятие горячего зеркала)

  Алгоритмы ниже работают в обоих случаях, а также позволяют
  симулировать ИК-эффект на обычных цветных снимках.
"""

import numpy as np
import cv2
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Конвертация ложноцветных ИК → Ч/Б
# ---------------------------------------------------------------------------

def ir_channel_swap(image: np.ndarray) -> np.ndarray:
    """
    Обмен каналов R ↔ B (Channel Swap).

    Классическая первая операция для конвертированных ИК-камер.
    Превращает характерное малиново-розовое «ложноцветное» изображение
    в более привычный вид: синее небо, красноватая листва.
    """
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])  # R и B меняются местами


def ir_to_bw_r_channel(image: np.ndarray) -> np.ndarray:
    """
    Прямое использование красного канала как ИК-данных.

    Для снимков с конвертированными камерами: красный канал
    несёт большую часть ИК-информации.
    """
    b, g, r = cv2.split(image)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)


def ir_to_bw_false_color(image: np.ndarray) -> np.ndarray:
    """
    Конвертация ложноцветного ИК в Ч/Б с оптимальным миксом каналов.

    После R↔B swap листва имеет высокий R-канал.
    Этот микс даёт яркую листву и тёмное небо.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    # R-канал несёт ИК (листва), G — промежуточный, B — видимый синий (небо)
    gray = 0.60 * r + 0.30 * g + 0.10 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ir_channel_mixer(image: np.ndarray,
                     wr: float = 0.60,
                     wg: float = 0.30,
                     wb: float = 0.10) -> np.ndarray:
    """
    Настраиваемый микс каналов для ИК.

    Позволяет точно настроить отображение под конкретную камеру и фильтр.
    Рекомендуемые значения:
      - 720nm фильтр: wr=0.70, wg=0.25, wb=0.05
      - 850nm фильтр: wr=0.85, wg=0.15, wb=0.00
      - 590nm фильтр: wr=0.50, wg=0.40, wb=0.10
    """
    b, g, r = cv2.split(image.astype(np.float32))
    gray = wr * r + wg * g + wb * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 2. Классические ИК-эффекты
# ---------------------------------------------------------------------------

def ir_wood_effect(image: np.ndarray,
                   brightness: float = 1.1) -> np.ndarray:
    """
    Эффект Вуда (Wood Effect).

    Имитирует съёмку пейзажа на ИК-плёнку:
    — Листва белая или очень светлая (высокий NIR).
    — Небо и вода тёмные (низкий NIR).
    — Характерный «молочный» вид облаков.

    brightness: коэффициент яркости (0.8–1.5).
    """
    b, g, r = cv2.split(image.astype(np.float32))

    # ИК-взвешивание: акцент на красный канал (ИК), слабый вклад синего
    gray = 0.75 * r + 0.25 * g
    gray = np.clip(gray * brightness, 0, 255)

    # Гамма < 1 → подсветка светов (пересвет листвы)
    gray_norm = gray / 255.0
    gray_gamma = np.power(np.clip(gray_norm, 0, 1), 0.65)

    result = np.clip(gray_gamma * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def ir_simulate_720nm(image: np.ndarray,
                       contrast: float = 1.2) -> np.ndarray:
    """
    Симуляция ИК-фильтра 720nm (Hoya R72 / Cokin P007).

    Самый популярный ИК-фильтр для цифровых камер.
    Пропускает длины волн > 720нм: листва очень светлая,
    небо тёмное, вода тёмная.

    contrast: усиление контраста (1.0–2.0).
    """
    b, g, r = cv2.split(image.astype(np.float32))

    # Характеристика пропускания ~720nm
    gray = 0.78 * r + 0.22 * g + 0.00 * b
    gray_norm = np.clip(gray / 255.0, 0, 1)

    # S-кривая для характерного ИК-контраста
    gray_contr = 0.5 + (gray_norm - 0.5) * contrast
    gray_contr = np.clip(gray_contr, 0, 1)

    # Подъём светов (листва → белый)
    gray_boosted = np.where(
        gray_contr > 0.5,
        0.5 + np.power((gray_contr - 0.5) * 2, 0.7) * 0.5,
        gray_contr
    )

    result = np.clip(gray_boosted * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def ir_simulate_850nm(image: np.ndarray) -> np.ndarray:
    """
    Симуляция ИК-фильтра 850nm.

    Более жёсткий фильтр: проходит только дальний ИК.
    Листва белее, небо почти чёрное. Используется для dramatic-пейзажей.
    """
    b, g, r = cv2.split(image.astype(np.float32))

    # Только красный канал, практически без зелёного
    gray = 0.92 * r + 0.08 * g
    gray = np.clip(gray, 0, 255)

    # Экстремальный подъём светов
    gray_norm = gray / 255.0
    gray_boosted = np.where(
        gray_norm > 0.3,
        np.power(gray_norm, 0.5),
        gray_norm * 0.6
    )

    # Сильное затемнение теней
    gray_final = np.clip(gray_boosted * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_final, cv2.COLOR_GRAY2BGR)


def ir_simulate_590nm(image: np.ndarray) -> np.ndarray:
    """
    Симуляция ИК-фильтра 590nm (мягкий «золотой» ИК).

    Пропускает видимый красный + ближний ИК. Менее экстремальный вид:
    листва золотистая, небо серо-синее, кожа светлая.
    """
    b, g, r = cv2.split(image.astype(np.float32))

    # Смешанный вклад: видимый диапазон + ИК
    gray = 0.50 * r + 0.40 * g + 0.10 * b
    gray_norm = np.clip(gray / 255.0, 0, 1)

    # Мягкая гамма — умеренный ИК-эффект
    gray_final = np.power(gray_norm, 0.75) * 255
    result = np.clip(gray_final, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 3. Художественные ИК-эффекты
# ---------------------------------------------------------------------------

def ir_high_contrast(image: np.ndarray,
                      black_point: float = 0.15,
                      white_point: float = 0.95) -> np.ndarray:
    """
    Высококонтрастная Ч/Б ИК-обработка.

    Экстремальный контраст между светлой листвой и тёмным небом.
    black_point: точка чёрного (0–0.5).
    white_point: точка белого (0.5–1.0).
    """
    b, g, r = cv2.split(image.astype(np.float32))
    gray = 0.75 * r + 0.25 * g
    gray_norm = np.clip(gray / 255.0, 0, 1)

    # Растяжка диапазона
    stretched = np.clip(
        (gray_norm - black_point) / (white_point - black_point),
        0, 1
    )

    # S-кривая для усиления контраста в средних тонах
    s = stretched
    high_contrast = s * s * (3.0 - 2.0 * s)  # smoothstep

    result = np.clip(high_contrast * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def ir_dreamy_glow(image: np.ndarray,
                    glow_strength: float = 0.35,
                    glow_radius: float = 25.0,
                    threshold: float = 0.6) -> np.ndarray:
    """
    Мечтательный ореол / Halation (ИК-свечение).

    Характерный эффект плёнки Kodak HIE и других ИК-плёнок:
    яркие области (листва) создают мягкий ореол вокруг себя.

    glow_strength: сила свечения (0.1–0.8).
    glow_radius:   радиус гало в пикселях (5–60).
    threshold:     порог яркости для применения гало (0.4–0.9).
    """
    img_bw = ir_wood_effect(image)
    img_f = img_bw.astype(np.float32)

    # Маска ярких областей
    b, g, r = cv2.split(img_f)
    lum_norm = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255) / 255.0
    bright_mask = np.clip((lum_norm - threshold) / (1.0 - threshold), 0, 1)

    # Размытый вариант ярких областей
    glow_layer = cv2.GaussianBlur(img_f, (0, 0), glow_radius)

    # Смешиваем: только там, где есть яркие области
    blend = bright_mask[:, :, np.newaxis] * glow_strength
    result = img_f * (1.0 - blend) + glow_layer * blend

    return np.clip(result, 0, 255).astype(np.uint8)


def ir_bleach_bypass(image: np.ndarray,
                      strength: float = 0.6) -> np.ndarray:
    """
    Эффект Bleach Bypass (пропуск отбелки).

    Кинематографический эффект, имитирующий обработку плёнки
    без стадии отбелки: высокий контраст, пониженная насыщенность,
    зернистые тени.

    strength: сила эффекта (0–1).
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Смешение оригинала (Screen режим) с серым
    # Screen: 1 - (1-A)(1-B)
    gray3 = gray[:, :, np.newaxis]
    img3 = img

    # Нормализуем
    a = img3 / 255.0
    b_layer = gray3 / 255.0

    screen = 1.0 - (1.0 - a) * (1.0 - b_layer)
    blended = a * (1.0 - strength) + screen * strength

    return np.clip(blended * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Тонирование ИК
# ---------------------------------------------------------------------------

def ir_warm_tone(image: np.ndarray,
                  warmth: float = 0.15) -> np.ndarray:
    """
    Тёплое тонирование ИК (имитация плёнки Kodak HIE).

    Слегка желтоватые/золотистые тона. Листва ослепительно белая.
    warmth: сила тёплого оттенка (0.05–0.4).
    """
    bw = ir_wood_effect(image)
    b, g, r = cv2.split(bw.astype(np.float32))
    gray = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255)

    new_r = np.clip(gray * (1.0 + warmth * 0.3), 0, 255)
    new_g = np.clip(gray * (1.0 + warmth * 0.1), 0, 255)
    new_b = np.clip(gray * (1.0 - warmth * 0.2), 0, 255)

    return cv2.merge([new_b, new_g, new_r]).astype(np.uint8)


def ir_cool_tone(image: np.ndarray,
                  coolness: float = 0.15) -> np.ndarray:
    """
    Холодное (лунное) тонирование ИК.

    Синеватые тени, белая листва. Создаёт лунно-ночное ощущение.
    coolness: сила холодного оттенка (0.05–0.4).
    """
    bw = ir_simulate_720nm(image)
    b, g, r = cv2.split(bw.astype(np.float32))
    gray = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255)

    new_r = np.clip(gray * (1.0 - coolness * 0.1), 0, 255)
    new_g = np.clip(gray * (1.0 + coolness * 0.05), 0, 255)
    new_b = np.clip(gray * (1.0 + coolness * 0.25), 0, 255)

    return cv2.merge([new_b, new_g, new_r]).astype(np.uint8)


def ir_cyanotype(image: np.ndarray) -> np.ndarray:
    """
    Цианотипия (Cyanotype) — печать синим.

    Классический фотографический процесс XIX века.
    Тени — насыщенный синий, света — белый.
    """
    bw = ir_simulate_720nm(image)
    b, g, r = cv2.split(bw.astype(np.float32))
    gray = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255) / 255.0

    new_r = np.clip(gray * 50, 0, 255)
    new_g = np.clip(gray * 110, 0, 255)
    new_b = np.clip(gray * 230, 0, 255)

    return cv2.merge([new_b, new_g, new_r]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Реестр методов для UI
# ---------------------------------------------------------------------------

IR_METHODS: dict[str, tuple] = {
    # Конвертация
    'channel_swap':     (ir_channel_swap,       'R↔B Обмен каналов',
                         'Меняет красный и синий каналы.\nПервый шаг обработки ИК-камеры.'),
    'to_bw_r':          (ir_to_bw_r_channel,    'Красный канал → Ч/Б',
                         'Только красный канал в Ч/Б.\nДля сконвертированных ИК-камер.'),
    'to_bw_false':      (ir_to_bw_false_color,  'Ложноцветный → Ч/Б',
                         'Оптимальный микс каналов\nдля ложноцветного ИК-снимка.'),
    'channel_mixer':    (ir_channel_mixer,       'Ручной микс каналов',
                         'Настраиваемый микс R/G/B\nдля разных фильтров.'),
    # Симуляция фильтров
    'simulate_590nm':   (ir_simulate_590nm,     'Фильтр 590nm (мягкий)',
                         'Мягкий «золотой» ИК.\nЛиства золотистая, небо серое.'),
    'simulate_720nm':   (ir_simulate_720nm,     'Фильтр 720nm (R72)',
                         'Самый популярный ИК-фильтр.\nЛиства белая, небо тёмное.'),
    'simulate_850nm':   (ir_simulate_850nm,     'Фильтр 850nm (жёсткий)',
                         'Дальний ИК. Экстремальный\nконтраст. Небо почти чёрное.'),
    'wood_effect':      (ir_wood_effect,         'Эффект Вуда',
                         'Классическая плёночная ИК.\nЛиства ослепительно белая.'),
    # Художественные
    'high_contrast':    (ir_high_contrast,       'Высокий контраст',
                         'Драматический контраст.\nЧёрные тени, белая листва.'),
    'dreamy_glow':      (ir_dreamy_glow,          'Мечтательный ореол',
                         'Свечение вокруг ярких областей.\nИмитация Kodak HIE.'),
    'bleach_bypass':    (ir_bleach_bypass,        'Bleach Bypass',
                         'Кинематографический эффект.\nВысокий контраст, зернистость.'),
    # Тонирование
    'warm_tone':        (ir_warm_tone,           'Тёплое тонирование',
                         'Золотисто-жёлтый оттенок.\nИмитация плёнки Kodak HIE.'),
    'cool_tone':        (ir_cool_tone,           'Лунное тонирование',
                         'Холодный синеватый оттенок.\nЛунно-ночное настроение.'),
    'cyanotype':        (ir_cyanotype,           'Цианотипия',
                         'Классический синий процесс.\nТени синие, света белые.'),
}


def apply_ir(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Применяет ИК-алгоритм к изображению.

    Args:
        image: BGR uint8 numpy-массив.
        method: ключ из IR_METHODS.
        **kwargs: специфичные параметры метода.
    """
    if method not in IR_METHODS:
        method = 'wood_effect'

    fn = IR_METHODS[method][0]

    import inspect
    sig = inspect.signature(fn)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}

    return fn(image, **filtered)


def get_ir_method_list() -> list[tuple[str, str]]:
    """Возвращает список (ключ, отображаемое имя) для UI."""
    return [(k, v[1]) for k, v in IR_METHODS.items()]


def get_ir_description(method: str) -> str:
    if method in IR_METHODS:
        return IR_METHODS[method][2]
    return ''
