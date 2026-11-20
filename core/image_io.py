"""
Модуль загрузки и сохранения изображений.
Поддерживает RAW (Nikon NEF/NRW), JPEG, PNG, TIFF, HEIF и другие форматы.
Оптимизировано для Nikon Z6 III.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

try:
    import rawpy
    import rawpy.enhance
    RAW_AVAILABLE = True
except ImportError:
    RAW_AVAILABLE = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- Поддерживаемые расширения ---

RAW_EXTENSIONS = {
    '.nef', '.nrw',          # Nikon (Z6 III, Z8, Z9, D серия)
    '.cr2', '.cr3',          # Canon
    '.arw', '.srf', '.sr2',  # Sony
    '.orf',                  # Olympus / OM System
    '.rw2',                  # Panasonic
    '.dng',                  # Adobe DNG (универсальный)
    '.raf',                  # Fujifilm
    '.pef',                  # Pentax
    '.srw',                  # Samsung
    '.x3f',                  # Sigma (Foveon)
    '.3fr',                  # Hasselblad
    '.mef',                  # Mamiya
    '.mrw',                  # Minolta / Konica-Minolta
    '.erf',                  # Epson
    '.rwl',                  # Leica
}

STANDARD_EXTENSIONS = {
    '.jpg', '.jpeg',
    '.png',
    '.tiff', '.tif',
    '.bmp',
    '.ppm', '.pgm', '.pbm',
    '.webp',
}

HEIF_EXTENSIONS = {'.heif', '.heic'}

ALL_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS | HEIF_EXTENSIONS


def get_supported_extensions() -> list[str]:
    """Возвращает список поддерживаемых расширений."""
    exts = list(STANDARD_EXTENSIONS)
    if RAW_AVAILABLE:
        exts.extend(list(RAW_EXTENSIONS))
    if HEIF_AVAILABLE:
        exts.extend(list(HEIF_EXTENSIONS))
    return sorted(exts)


def is_raw_file(filepath: str) -> bool:
    """Проверяет, является ли файл RAW."""
    return Path(filepath).suffix.lower() in RAW_EXTENSIONS


def get_file_filter() -> str:
    """Возвращает строку фильтра для диалога открытия файлов Qt."""
    std_exts = ' '.join(f'*{e}' for e in sorted(STANDARD_EXTENSIONS))
    all_exts_parts = [std_exts]

    filters = []

    if RAW_AVAILABLE:
        raw_exts = ' '.join(f'*{e}' for e in sorted(RAW_EXTENSIONS))
        all_exts_parts.append(raw_exts)
        filters.append(f'RAW файлы (Nikon NEF/NRW и др.) ({raw_exts})')

    if HEIF_AVAILABLE:
        heif_exts = ' '.join(f'*{e}' for e in sorted(HEIF_EXTENSIONS))
        all_exts_parts.append(heif_exts)
        filters.append(f'HEIF/HEIC файлы ({heif_exts})')

    all_exts = ' '.join(all_exts_parts)
    filters.insert(0, f'Все поддерживаемые форматы ({all_exts})')
    filters.extend([
        'JPEG файлы (*.jpg *.jpeg)',
        'PNG файлы (*.png)',
        'TIFF файлы (*.tiff *.tif)',
        'Все файлы (*.*)',
    ])
    return ';;'.join(filters)


def get_save_filter() -> str:
    """Строка фильтра для диалога сохранения файлов."""
    return (
        'JPEG файлы (*.jpg);;'
        'PNG файлы (*.png);;'
        'TIFF файлы (*.tiff);;'
        'BMP файлы (*.bmp)'
    )


# ---------------------------------------------------------------------------
# Загрузка изображений
# ---------------------------------------------------------------------------

def load_image(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Загружает изображение из файла.

    Returns:
        (bgr_image, metadata) — BGR numpy uint8 массив и словарь метаданных.

    Raises:
        FileNotFoundError: файл не найден.
        ValueError: формат не поддерживается или файл повреждён.
        ImportError: нужная библиотека не установлена.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f'Файл не найден: {filepath}')

    ext = filepath.suffix.lower()
    metadata: dict = {
        'filename': filepath.name,
        'format': ext.upper().lstrip('.'),
        'path': str(filepath),
        'is_raw': ext in RAW_EXTENSIONS,
        'size': (0, 0),
        'channels': 3,
        'bit_depth': 8,
    }

    if ext in RAW_EXTENSIONS:
        return _load_raw(filepath, metadata)
    elif ext in HEIF_EXTENSIONS:
        return _load_heif(filepath, metadata)
    else:
        return _load_standard(filepath, metadata)


def _load_raw(filepath: Path, metadata: dict) -> tuple[np.ndarray, dict]:
    """Загружает RAW-файл через rawpy (LibRaw)."""
    if not RAW_AVAILABLE:
        raise ImportError(
            'Библиотека rawpy не установлена.\n'
            'Установите командой: pip install rawpy'
        )

    with rawpy.imread(str(filepath)) as raw:
        # --- Метаданные камеры ---
        try:
            metadata['camera_make'] = raw.metadata.make.decode('utf-8', errors='ignore').strip()
        except Exception:
            metadata['camera_make'] = ''
        try:
            metadata['camera_model'] = raw.metadata.model.decode('utf-8', errors='ignore').strip()
        except Exception:
            metadata['camera_model'] = ''
        try:
            metadata['iso'] = raw.metadata.iso
        except Exception:
            metadata['iso'] = None
        try:
            metadata['shutter'] = raw.metadata.shutter
        except Exception:
            metadata['shutter'] = None
        try:
            metadata['aperture'] = raw.metadata.aperture
        except Exception:
            metadata['aperture'] = None
        try:
            metadata['focal_len'] = raw.metadata.focal_len
        except Exception:
            metadata['focal_len'] = None

        metadata['raw_colors'] = int(raw.num_colors)

        # --- Обработка RAW ---
        # Параметры оптимизированы для Nikon Z6 III (14-бит NEF)
        try:
            rgb = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                half_size=False,
                no_auto_bright=False,
                output_bps=8,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Light,
                bright=1.0,
                user_flip=0,          # Не переворачивать — EXIF обработаем сами
            )
        except Exception:
            # Запасной вариант без расширенных параметров
            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )
            except Exception:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Поворот по EXIF
    bgr = _apply_exif_rotation(bgr, str(filepath))

    metadata['size'] = (bgr.shape[1], bgr.shape[0])
    metadata['channels'] = bgr.shape[2]
    metadata['bit_depth'] = 8  # Мы всегда конвертируем в 8 бит для отображения

    return bgr, metadata


def _load_heif(filepath: Path, metadata: dict) -> tuple[np.ndarray, dict]:
    """Загружает HEIF/HEIC файл."""
    if not HEIF_AVAILABLE:
        raise ImportError(
            'Библиотека pillow-heif не установлена.\n'
            'Установите командой: pip install pillow-heif'
        )
    if not PIL_AVAILABLE:
        raise ImportError('Pillow не установлен. Установите: pip install Pillow')

    pil_img = PILImage.open(str(filepath))
    pil_img = pil_img.convert('RGB')
    rgb = np.array(pil_img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    metadata['size'] = (bgr.shape[1], bgr.shape[0])
    metadata['channels'] = 3
    metadata['bit_depth'] = 8

    return bgr, metadata


def _load_standard(filepath: Path, metadata: dict) -> tuple[np.ndarray, dict]:
    """Загружает стандартный формат (JPEG, PNG, TIFF, BMP)."""
    bgr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    if bgr is None:
        # Запасной вариант через PIL
        if PIL_AVAILABLE:
            pil_img = PILImage.open(str(filepath))
            if pil_img.mode == 'RGBA':
                bg = PILImage.new('RGB', pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[3])
                pil_img = bg
            elif pil_img.mode not in ('RGB', 'L'):
                pil_img = pil_img.convert('RGB')
            rgb = np.array(pil_img, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if pil_img.mode == 'RGB' else \
                  cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f'Не удалось открыть файл: {filepath}')
    else:
        # Нормализация глубины цвета
        if bgr.dtype == np.uint16:
            bgr = (bgr >> 8).astype(np.uint8)
            metadata['bit_depth'] = 16
        elif bgr.dtype == np.float32 or bgr.dtype == np.float64:
            bgr = np.clip(bgr * 255, 0, 255).astype(np.uint8)

        # Нормализация количества каналов
        if len(bgr.shape) == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            # BGRA → BGR на белом фоне
            alpha = bgr[:, :, 3:4].astype(np.float32) / 255.0
            rgb_part = bgr[:, :, :3].astype(np.float32)
            white = np.full_like(rgb_part, 255.0)
            bgr = np.clip(rgb_part * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)

    # Поворот по EXIF
    bgr = _apply_exif_rotation(bgr, str(filepath))

    metadata['size'] = (bgr.shape[1], bgr.shape[0])
    metadata['channels'] = bgr.shape[2] if len(bgr.shape) > 2 else 1

    return bgr, metadata


def _apply_exif_rotation(image: np.ndarray, filepath: str) -> np.ndarray:
    """Применяет поворот по EXIF-ориентации."""
    if not PIL_AVAILABLE:
        return image
    try:
        pil_img = PILImage.open(filepath)
        exif = pil_img._getexif()
        if exif:
            orientation = exif.get(274)  # 274 = Orientation tag
            if orientation == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif orientation == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image


# ---------------------------------------------------------------------------
# Сохранение изображений
# ---------------------------------------------------------------------------

def save_image(image: np.ndarray, filepath: str, quality: int = 95) -> bool:
    """
    Сохраняет изображение в файл.

    Args:
        image: BGR numpy uint8 массив.
        filepath: путь для сохранения.
        quality: качество JPEG (1–100).

    Returns:
        True при успехе, False при ошибке.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    try:
        if ext in {'.jpg', '.jpeg'}:
            params = [cv2.IMWRITE_JPEG_QUALITY, min(100, max(1, quality))]
            success = cv2.imwrite(str(filepath), image, params)
        elif ext == '.png':
            compression = max(0, min(9, 9 - quality // 11))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            success = cv2.imwrite(str(filepath), image, params)
        elif ext in {'.tiff', '.tif'}:
            # Сохраняем без потерь
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
            success = cv2.imwrite(str(filepath), image, params)
        else:
            success = cv2.imwrite(str(filepath), image)

        return bool(success)

    except Exception as e:
        print(f'Ошибка сохранения файла: {e}')
        return False


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_image_info(image: np.ndarray, metadata: dict) -> str:
    """Возвращает строку с информацией об изображении для статусной строки."""
    if image is None:
        return ''
    h, w = image.shape[:2]
    fmt = metadata.get('format', '?')
    cam = ''
    model = metadata.get('camera_model', '')
    make = metadata.get('camera_make', '')
    if model:
        cam = f' | {make} {model}'.strip()
    iso = metadata.get('iso')
    shutter = metadata.get('shutter')
    aperture = metadata.get('aperture')
    exp = ''
    if iso:
        exp += f' ISO {int(iso)}'
    if shutter and shutter > 0:
        if shutter >= 1:
            exp += f' {int(shutter)}s'
        else:
            exp += f' 1/{int(round(1.0/shutter))}s'
    if aperture:
        exp += f' f/{aperture:.1f}'
    return f'{w}×{h} | {fmt}{cam}{exp}'


def raw_available() -> bool:
    return RAW_AVAILABLE


def heif_available() -> bool:
    return HEIF_AVAILABLE
