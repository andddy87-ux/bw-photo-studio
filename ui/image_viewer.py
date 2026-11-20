"""
Виджет для отображения изображений с поддержкой:
  - Режим сравнения Before/After с перетаскиваемым разделителем
  - Масштабирование колесом мыши
  - Перемещение (pan) правой кнопкой мыши
  - Гистограмма изображения
"""

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsTextItem, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPen, QFont, QPainter,
    QPainterPath, QLinearGradient, QBrush
)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Конвертирует BGR numpy uint8 → QPixmap."""
    if image is None or image.size == 0:
        return QPixmap()
    img = np.ascontiguousarray(image)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # copy() чтобы не держать ссылку на буфер


# ---------------------------------------------------------------------------
# Виджет просмотра
# ---------------------------------------------------------------------------

class ImageViewer(QGraphicsView):
    """
    Масштабируемый просмотрщик с режимом сравнения Before/After.

    Сигналы:
        zoom_changed(float) — коэффициент масштабирования.
        cursor_moved(int, int) — координаты курсора в пикселях исходного изображения.
    """

    zoom_changed  = pyqtSignal(float)
    cursor_moved  = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Сцена ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # --- Настройки вида ---
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(28, 28, 28))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # --- Элементы сцены ---
        self._orig_item = QGraphicsPixmapItem()
        self._proc_item = QGraphicsPixmapItem()
        self._scene.addItem(self._orig_item)
        self._scene.addItem(self._proc_item)

        # Разделительная линия Before/After
        pen = QPen(QColor(255, 210, 50), 2, Qt.SolidLine)
        self._divider = QGraphicsLineItem()
        self._divider.setPen(pen)
        self._divider.setZValue(10)
        self._scene.addItem(self._divider)

        # Метки
        self._lbl_orig = self._make_label('ДО')
        self._lbl_proc = self._make_label('ПОСЛЕ')

        # --- Состояние ---
        self._orig_img: np.ndarray | None = None
        self._proc_img: np.ndarray | None = None
        self._split = 0.5          # Позиция разделителя 0..1
        self._drag_divider = False
        self._panning = False
        self._pan_start = QPointF()
        self._show_split = True
        self._zoom = 1.0

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def set_original(self, image: np.ndarray):
        self._orig_img = image
        self._refresh()

    def set_processed(self, image: np.ndarray | None):
        self._proc_img = image
        self._refresh()

    def set_split_mode(self, enabled: bool):
        self._show_split = enabled
        self._refresh()

    def fit_view(self):
        """Вписать изображение в окно."""
        if self._orig_img is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()
            self.zoom_changed.emit(self._zoom)

    def zoom_actual(self):
        """100% масштаб."""
        self.resetTransform()
        self._zoom = 1.0
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self):
        self._do_scale(1.25)

    def zoom_out(self):
        self._do_scale(1.0 / 1.25)

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _make_label(self, text: str) -> QGraphicsTextItem:
        item = QGraphicsTextItem(text)
        item.setDefaultTextColor(QColor(255, 230, 80))
        font = QFont('Segoe UI', 10, QFont.Bold)
        item.setFont(font)
        item.setZValue(11)
        self._scene.addItem(item)
        return item

    def _refresh(self):
        """Обновляет отображение изображений."""
        if self._orig_img is None:
            self._scene.setSceneRect(QRectF(0, 0, 400, 300))
            return

        h, w = self._orig_img.shape[:2]
        split_x = int(w * self._split)
        has_proc = self._proc_img is not None and self._show_split

        if has_proc:
            # --- Режим сравнения ---
            orig_half = self._orig_img[:, :split_x]
            proc_half = self._proc_img[:, split_x:]

            self._orig_item.setPixmap(ndarray_to_qpixmap(orig_half))
            self._orig_item.setPos(0, 0)
            self._proc_item.setPixmap(ndarray_to_qpixmap(proc_half))
            self._proc_item.setPos(split_x, 0)

            self._divider.setLine(split_x, 0, split_x, h)
            self._divider.setVisible(True)

            # Позиционируем метки
            ox = max(4, split_x // 2 - 20)
            self._lbl_orig.setPos(ox, 8)
            self._lbl_orig.setVisible(True)
            px = split_x + max(4, (w - split_x) // 2 - 30)
            self._lbl_proc.setPos(px, 8)
            self._lbl_proc.setVisible(True)
        else:
            # --- Один кадр ---
            img = self._proc_img if (self._proc_img is not None) else self._orig_img
            self._orig_item.setPixmap(ndarray_to_qpixmap(img))
            self._orig_item.setPos(0, 0)
            self._proc_item.setPixmap(QPixmap())
            self._divider.setVisible(False)
            self._lbl_orig.setVisible(False)
            self._lbl_proc.setVisible(False)

        self._scene.setSceneRect(0, 0, w, h)

    def _do_scale(self, factor: float):
        self.scale(factor, factor)
        self._zoom = self.transform().m11()
        self.zoom_changed.emit(self._zoom)

    def _near_divider(self, scene_x: float) -> bool:
        if self._orig_img is None:
            return False
        w = self._orig_img.shape[1]
        return abs(scene_x - w * self._split) < 10 / self._zoom

    # ------------------------------------------------------------------
    # События мыши
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self._do_scale(factor)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            if self._show_split and self._proc_img is not None and self._near_divider(sp.x()):
                self._drag_divider = True
                self.setCursor(Qt.SplitHCursor)
                return
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.pos())

        if self._drag_divider and self._orig_img is not None:
            w = self._orig_img.shape[1]
            self._split = float(np.clip(sp.x() / max(w, 1), 0.02, 0.98))
            self._refresh()
            return

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return

        # Обновляем курсор и испускаем позицию
        if self._show_split and self._proc_img is not None and self._near_divider(sp.x()):
            self.setCursor(Qt.SplitHCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self._orig_img is not None:
            h, w = self._orig_img.shape[:2]
            px, py = int(sp.x()), int(sp.y())
            if 0 <= px < w and 0 <= py < h:
                self.cursor_moved.emit(px, py)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_divider = False
        self._panning = False
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Виджет гистограммы
# ---------------------------------------------------------------------------

class HistogramWidget(QWidget):
    """Рисует гистограмму яркости/каналов поверх чёрного фона."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[tuple[np.ndarray, QColor]] = []
        self.setMinimumHeight(90)
        self.setMaximumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip('Гистограмма изображения')

    def set_image(self, image: np.ndarray | None):
        """Вычисляет и отрисовывает гистограмму."""
        self._data = []
        if image is None or image.size == 0:
            self.update()
            return

        img = np.clip(image, 0, 255).astype(np.uint8)

        if len(img.shape) == 3:
            # Яркость (серый)
            b_ch, g_ch, r_ch = cv2.split(img)
            lum = (0.2126 * r_ch.astype(np.float32) +
                   0.7152 * g_ch.astype(np.float32) +
                   0.0722 * b_ch.astype(np.float32)).astype(np.uint8)

            # Проверяем: цветное или серое изображение?
            is_gray = (np.mean(np.abs(r_ch.astype(int) - b_ch.astype(int))) < 2.0)

            if is_gray:
                h = cv2.calcHist([lum], [0], None, [256], [0, 256])
                self._data.append((h.flatten(), QColor(200, 200, 200)))
            else:
                for ch_arr, col in [
                    (b_ch, QColor(80,  120, 220, 180)),
                    (g_ch, QColor(80,  200,  80, 180)),
                    (r_ch, QColor(220,  80,  80, 180)),
                ]:
                    h = cv2.calcHist([ch_arr], [0], None, [256], [0, 256])
                    self._data.append((h.flatten(), col))
        else:
            h = cv2.calcHist([img], [0], None, [256], [0, 256])
            self._data.append((h.flatten(), QColor(200, 200, 200)))

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 4, 4, 4, 18

        # Фон
        painter.fillRect(0, 0, w, h, QColor(18, 18, 18))

        if not self._data:
            painter.setPen(QColor(60, 60, 60))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, 'Нет изображения')
            painter.end()
            return

        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        # Максимум для нормализации
        global_max = max(d[0].max() for d in self._data)
        if global_max == 0:
            painter.end()
            return

        # Рисуем каналы
        painter.setCompositionMode(QPainter.CompositionMode_Screen)
        for hist, color in self._data:
            path = QPainterPath()
            for i in range(256):
                bar_h = int(hist[i] / global_max * chart_h * 0.95)
                x = pad_l + int(i * chart_w / 256)
                y = pad_t + chart_h - bar_h
                if i == 0:
                    path.moveTo(x, pad_t + chart_h)
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)
            path.lineTo(pad_l + chart_w, pad_t + chart_h)
            path.closeSubpath()
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawPath(path)

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Оси
        painter.setPen(QColor(50, 50, 50))
        painter.drawLine(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h)

        # Шкала (0, 64, 128, 192, 255)
        painter.setPen(QColor(80, 80, 80))
        font = QFont('Segoe UI', 7)
        painter.setFont(font)
        for val in [0, 64, 128, 192, 255]:
            x = pad_l + int(val * chart_w / 255)
            painter.drawLine(x, pad_t + chart_h, x, pad_t + chart_h + 3)
            painter.drawText(x - 8, pad_t + chart_h + 4, 20, 14,
                             Qt.AlignCenter, str(val))

        painter.end()
