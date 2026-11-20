"""
Панели инструментов для BW Photo Studio.

Классы:
  BWConversionPanel  — конвертация в Ч/Б
  SepiaPanel         — тонирование в сепию
  EditingPanel       — редактирование Ч/Б фото
  InfraredPanel      — инфракрасная обработка
"""

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QSlider, QComboBox, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup, QScrollArea, QFrame,
    QDoubleSpinBox, QSpinBox, QSizePolicy, QColorDialog,
    QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPalette

from core.bw_algorithms  import (apply_bw, BW_METHODS, get_method_list,
                                   get_method_description, FILTER_PRESETS)
from core.sepia           import (apply_sepia, SEPIA_METHODS, get_sepia_method_list,
                                   get_sepia_description)
from core.editor          import (EDIT_TOOLS, adjust_brightness_contrast, adjust_levels,
                                   adjust_curves, histogram_equalization, clahe,
                                   unsharp_mask, local_contrast,
                                   denoise_nlmeans, denoise_bilateral,
                                   denoise_gaussian, denoise_median,
                                   add_vignette, add_film_grain, split_toning,
                                   rotate_image, flip_image)
from core.infrared        import (apply_ir, IR_METHODS, get_ir_method_list,
                                   get_ir_description)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _make_slider(parent, min_val: int, max_val: int, default: int,
                 step: int = 1) -> tuple[QSlider, QLabel]:
    """Создаёт горизонтальный слайдер и метку со значением."""
    slider = QSlider(Qt.Horizontal, parent)
    slider.setRange(min_val, max_val)
    slider.setValue(default)
    slider.setSingleStep(step)
    value_lbl = QLabel(str(default), parent)
    value_lbl.setObjectName('valueLabel')
    value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    value_lbl.setFixedWidth(44)
    slider.valueChanged.connect(lambda v: value_lbl.setText(str(v)))
    return slider, value_lbl


def _slider_row(parent, label: str, min_val: int, max_val: int,
                default: int) -> tuple[QHBoxLayout, QSlider]:
    """Строка: «Метка  [====slider====]  value»"""
    row = QHBoxLayout()
    lbl = QLabel(label, parent)
    lbl.setFixedWidth(110)
    slider, val_lbl = _make_slider(parent, min_val, max_val, default)
    row.addWidget(lbl)
    row.addWidget(slider)
    row.addWidget(val_lbl)
    return row, slider


def _color_button(parent, color_rgb: tuple) -> QPushButton:
    btn = QPushButton(parent)
    btn.setObjectName('colorBtn')
    btn.setFixedSize(50, 22)
    _set_btn_color(btn, color_rgb)
    return btn


def _set_btn_color(btn: QPushButton, rgb: tuple):
    r, g, b = rgb
    btn.setStyleSheet(
        f'QPushButton#colorBtn {{ background-color: rgb({r},{g},{b}); '
        f'border: 2px solid #666; border-radius: 3px; }}'
        f'QPushButton#colorBtn:hover {{ border-color: #aaa; }}'
    )
    btn._color = rgb


def _pick_color(btn: QPushButton, parent_widget: QWidget):
    """Открывает диалог выбора цвета и обновляет кнопку."""
    r, g, b = getattr(btn, '_color', (128, 128, 128))
    initial = QColor(r, g, b)
    color = QColorDialog.getColor(initial, parent_widget, 'Выберите цвет')
    if color.isValid():
        rgb = (color.red(), color.green(), color.blue())
        _set_btn_color(btn, rgb)
        return rgb
    return None


def _apply_primary_btn(text: str, parent=None) -> QPushButton:
    btn = QPushButton(text, parent)
    btn.setObjectName('primaryBtn')
    return btn


def _reset_btn(parent=None) -> QPushButton:
    btn = QPushButton('Сброс', parent)
    btn.setObjectName('resetBtn')
    return btn


def _section_title(text: str, parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setObjectName('sectionTitle')
    return lbl


def _wrap_scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QFrame.NoFrame)
    return area


# ---------------------------------------------------------------------------
# Базовый класс панели
# ---------------------------------------------------------------------------

class _BasePanel(QWidget):
    """Базовый класс панели инструментов. Испускает сигнал result_ready."""

    result_ready = pyqtSignal(np.ndarray)  # Испускает обработанное изображение

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: np.ndarray | None = None

    def set_source(self, image: np.ndarray | None):
        """Устанавливает исходное изображение (оригинал или текущая рабочая копия)."""
        self._source = image

    def _emit_result(self, result: np.ndarray | None):
        if result is not None:
            self.result_ready.emit(result)


# ===========================================================================
# 1. Панель конвертации в Ч/Б
# ===========================================================================

class BWConversionPanel(_BasePanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Заголовок ---
        layout.addWidget(_section_title('Конвертация в Ч/Б'))

        # --- Выбор алгоритма ---
        lbl_alg = QLabel('Алгоритм:')
        self._combo = QComboBox()
        for key, name in get_method_list():
            self._combo.addItem(name, key)
        layout.addWidget(lbl_alg)
        layout.addWidget(self._combo)

        # --- Описание метода ---
        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName('methodDesc')
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(50)
        layout.addWidget(self._desc_lbl)

        layout.addWidget(_hline())

        # --- Динамические настройки ---
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Страница 0: пустая (для методов без настроек)
        self._stack.addWidget(QWidget())

        # Страница 1: Один канал (R/G/B)
        self._channel_page = self._build_channel_page()
        self._stack.addWidget(self._channel_page)

        # Страница 2: Микширование каналов
        self._mixer_page, self._r_sl, self._g_sl, self._b_sl, \
            self._r_lbl, self._g_lbl, self._b_lbl, self._preset_combo = \
            self._build_mixer_page()
        self._stack.addWidget(self._mixer_page)

        layout.addWidget(_hline())

        # --- Кнопки ---
        btn_row = QHBoxLayout()
        self._apply_btn = _apply_primary_btn('Применить')
        self._reset_btn = _reset_btn()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        scroll = _wrap_scroll(content)
        outer.addWidget(scroll)

        # --- Подключение сигналов ---
        self._combo.currentIndexChanged.connect(self._on_method_changed)
        self._apply_btn.clicked.connect(self._apply)
        self._reset_btn.clicked.connect(self._reset)

        # Начальное состояние
        self._combo.setCurrentIndex(6)  # rec601 по умолчанию
        self._on_method_changed()

    def _build_channel_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Использовать канал:'))
        self._ch_group = QButtonGroup(w)
        for i, (val, text) in enumerate([('R', 'Красный (R)'), ('G', 'Зелёный (G)'), ('B', 'Синий (B)')]):
            rb = QRadioButton(text, w)
            rb.setProperty('channel', val)
            if val == 'G':
                rb.setChecked(True)
            self._ch_group.addButton(rb, i)
            layout.addWidget(rb)
        return w

    def _build_mixer_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel('Пресет фильтра:'))
        preset_combo = QComboBox()
        preset_combo.addItem('— Настроить вручную —', None)
        for name, vals in FILTER_PRESETS.items():
            preset_combo.addItem(name, vals)
        layout.addWidget(preset_combo)

        layout.addWidget(QLabel('Веса каналов (сумма = 1.0):'))

        r_row = QHBoxLayout()
        r_lbl = QLabel('R:'); r_lbl.setFixedWidth(20)
        r_sl = QSlider(Qt.Horizontal); r_sl.setRange(0, 100); r_sl.setValue(21)
        r_val = QLabel('0.21'); r_val.setObjectName('valueLabel'); r_val.setFixedWidth(36)
        r_sl.valueChanged.connect(lambda v: (r_val.setText(f'{v/100:.2f}'), self._normalize_mixer('R', v)))
        r_row.addWidget(r_lbl); r_row.addWidget(r_sl); r_row.addWidget(r_val)
        layout.addLayout(r_row)

        g_row = QHBoxLayout()
        g_lbl = QLabel('G:'); g_lbl.setFixedWidth(20)
        g_sl = QSlider(Qt.Horizontal); g_sl.setRange(0, 100); g_sl.setValue(72)
        g_val = QLabel('0.72'); g_val.setObjectName('valueLabel'); g_val.setFixedWidth(36)
        g_sl.valueChanged.connect(lambda v: (g_val.setText(f'{v/100:.2f}'), self._normalize_mixer('G', v)))
        g_row.addWidget(g_lbl); g_row.addWidget(g_sl); g_row.addWidget(g_val)
        layout.addLayout(g_row)

        b_row = QHBoxLayout()
        b_lbl = QLabel('B:'); b_lbl.setFixedWidth(20)
        b_sl = QSlider(Qt.Horizontal); b_sl.setRange(0, 100); b_sl.setValue(7)
        b_val = QLabel('0.07'); b_val.setObjectName('valueLabel'); b_val.setFixedWidth(36)
        b_sl.valueChanged.connect(lambda v: (b_val.setText(f'{v/100:.2f}'), self._normalize_mixer('B', v)))
        b_row.addWidget(b_lbl); b_row.addWidget(b_sl); b_row.addWidget(b_val)
        layout.addLayout(b_row)

        preset_combo.currentIndexChanged.connect(lambda i: self._apply_preset(preset_combo, r_sl, g_sl, b_sl))

        return w, r_sl, g_sl, b_sl, r_val, g_val, b_val, preset_combo

    def _apply_preset(self, combo, r_sl, g_sl, b_sl):
        vals = combo.currentData()
        if vals:
            wr, wg, wb = vals
            # Блокируем обратный вызов нормализации при установке значений
            r_sl.blockSignals(True); g_sl.blockSignals(True); b_sl.blockSignals(True)
            r_sl.setValue(int(round(wr * 100)))
            g_sl.setValue(int(round(wg * 100)))
            b_sl.setValue(int(round(wb * 100)))
            r_sl.blockSignals(False); g_sl.blockSignals(False); b_sl.blockSignals(False)
            self._r_lbl.setText(f'{wr:.2f}')
            self._g_lbl.setText(f'{wg:.2f}')
            self._b_lbl.setText(f'{wb:.2f}')

    def _normalize_mixer(self, changed: str, val: int):
        """Нормализует слайдеры микшера (сумма = 100)."""
        # Не нормализуем автоматически — пользователь сам следит
        pass

    def _on_method_changed(self):
        idx = self._combo.currentIndex()
        key = self._combo.itemData(idx)
        desc = get_method_description(key)
        self._desc_lbl.setText(desc)

        if key in ('channel_r', 'channel_g', 'channel_b'):
            self._stack.setCurrentIndex(1)
            # Выбираем нужную радиокнопку
            ch_map = {'channel_r': 0, 'channel_g': 1, 'channel_b': 2}
            self._ch_group.button(ch_map[key]).setChecked(True)
        elif key == 'channel_mixer':
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(0)

    def _apply(self):
        if self._source is None:
            return
        key = self._combo.currentData()
        kwargs = {}

        if key in ('channel_r', 'channel_g', 'channel_b'):
            ch_map = {0: 'R', 1: 'G', 2: 'B'}
            ch = ch_map.get(self._ch_group.checkedId(), 'G')
            key = f'channel_{ch.lower()}'
        elif key == 'channel_mixer':
            wr = self._r_sl.value() / 100.0
            wg = self._g_sl.value() / 100.0
            wb = self._b_sl.value() / 100.0
            total = wr + wg + wb
            if total > 0:
                wr /= total; wg /= total; wb /= total
            kwargs = {'wr': wr, 'wg': wg, 'wb': wb}

        result = apply_bw(self._source, key, **kwargs)
        self._emit_result(result)

    def _reset(self):
        if self._source is not None:
            self._emit_result(self._source.copy())


# ===========================================================================
# 2. Панель тонирования в сепию
# ===========================================================================

class SepiaPanel(_BasePanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(_section_title('Тонирование в сепию'))

        self._combo = QComboBox()
        for key, name in get_sepia_method_list():
            self._combo.addItem(name, key)
        layout.addWidget(QLabel('Метод:'))
        layout.addWidget(self._combo)

        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName('methodDesc')
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(50)
        layout.addWidget(self._desc_lbl)

        layout.addWidget(_hline())

        # --- Динамические настройки (stack) ---
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # 0: matrix — нет настроек
        self._stack.addWidget(QWidget())

        # 1: HSL
        p1 = QWidget(); pl1 = QVBoxLayout(p1); pl1.setContentsMargins(0,0,0,0)
        hue_row, self._hue_sl = _slider_row(p1, 'Тон (°):', 10, 60, 35)
        sat_row, self._sat_sl = _slider_row(p1, 'Насыщенность %:', 5, 60, 28)
        pl1.addLayout(hue_row); pl1.addLayout(sat_row)
        self._stack.addWidget(p1)

        # 2: YUV
        p2 = QWidget(); pl2 = QVBoxLayout(p2); pl2.setContentsMargins(0,0,0,0)
        u_row, self._u_sl = _slider_row(p2, 'U (Cb) offset:', -60, 60, -18)
        v_row, self._v_sl = _slider_row(p2, 'V (Cr) offset:', -60, 60, 28)
        pl2.addLayout(u_row); pl2.addLayout(v_row)
        self._stack.addWidget(p2)

        # 3: Gradient Map
        p3 = QWidget(); pl3 = QVBoxLayout(p3); pl3.setContentsMargins(0,0,0,0)
        pl3.addWidget(QLabel('Цвет теней:'))
        sh_row = QHBoxLayout()
        self._shadow_btn = _color_button(p3, (40, 25, 15))
        sh_row.addWidget(self._shadow_btn); sh_row.addStretch()
        pl3.addLayout(sh_row)
        pl3.addWidget(QLabel('Цвет средних тонов:'))
        mid_row = QHBoxLayout()
        self._midtone_btn = _color_button(p3, (120, 80, 45))
        mid_row.addWidget(self._midtone_btn); mid_row.addStretch()
        pl3.addLayout(mid_row)
        pl3.addWidget(QLabel('Цвет светов:'))
        hl_row = QHBoxLayout()
        self._highlight_btn = _color_button(p3, (240, 220, 195))
        hl_row.addWidget(self._highlight_btn); hl_row.addStretch()
        pl3.addLayout(hl_row)
        self._shadow_btn.clicked.connect(lambda: _pick_color(self._shadow_btn, self))
        self._midtone_btn.clicked.connect(lambda: _pick_color(self._midtone_btn, self))
        self._highlight_btn.clicked.connect(lambda: _pick_color(self._highlight_btn, self))
        self._stack.addWidget(p3)

        # 4: Duotone
        p4 = QWidget(); pl4 = QVBoxLayout(p4); pl4.setContentsMargins(0,0,0,0)
        pl4.addWidget(QLabel('Цвет теней (тёмный):'))
        c1r = QHBoxLayout(); self._duo_c1_btn = _color_button(p4, (20, 10, 5))
        c1r.addWidget(self._duo_c1_btn); c1r.addStretch(); pl4.addLayout(c1r)
        pl4.addWidget(QLabel('Цвет светов (светлый):'))
        c2r = QHBoxLayout(); self._duo_c2_btn = _color_button(p4, (200, 155, 95))
        c2r.addWidget(self._duo_c2_btn); c2r.addStretch(); pl4.addLayout(c2r)
        self._duo_c1_btn.clicked.connect(lambda: _pick_color(self._duo_c1_btn, self))
        self._duo_c2_btn.clicked.connect(lambda: _pick_color(self._duo_c2_btn, self))
        self._stack.addWidget(p4)

        # 5: Curves
        p5 = QWidget(); pl5 = QVBoxLayout(p5); pl5.setContentsMargins(0,0,0,0)
        rb_row, self._rb_sl = _slider_row(p5, 'Подъём красного:', 0, 50, 20)
        bs_row, self._bs_sl = _slider_row(p5, 'Тени (синий):', -60, 0, -30)
        bm_row, self._bm_sl = _slider_row(p5, 'Средние (синий):', -40, 0, -15)
        pl5.addLayout(rb_row); pl5.addLayout(bs_row); pl5.addLayout(bm_row)
        self._stack.addWidget(p5)

        layout.addWidget(_hline())

        btn_row = QHBoxLayout()
        self._apply_btn = _apply_primary_btn('Применить')
        self._reset_btn = _reset_btn()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        scroll = _wrap_scroll(content)
        outer.addWidget(scroll)

        self._combo.currentIndexChanged.connect(self._on_method_changed)
        self._apply_btn.clicked.connect(self._apply)
        self._reset_btn.clicked.connect(self._reset)
        self._on_method_changed()

    _PAGE = {'matrix': 0, 'hsl': 1, 'yuv': 2,
              'gradient_map': 3, 'duotone': 4, 'curves': 5}

    def _on_method_changed(self):
        key = self._combo.currentData()
        self._desc_lbl.setText(get_sepia_description(key))
        self._stack.setCurrentIndex(self._PAGE.get(key, 0))

    def _apply(self):
        if self._source is None:
            return
        key = self._combo.currentData()
        kwargs = {}
        if key == 'hsl':
            kwargs = {'hue': self._hue_sl.value(),
                      'saturation': self._sat_sl.value() / 100.0}
        elif key == 'yuv':
            kwargs = {'u_offset': self._u_sl.value(),
                      'v_offset': self._v_sl.value()}
        elif key == 'gradient_map':
            kwargs = {'shadow_rgb': self._shadow_btn._color,
                      'midtone_rgb': self._midtone_btn._color,
                      'highlight_rgb': self._highlight_btn._color}
        elif key == 'duotone':
            kwargs = {'color1_rgb': self._duo_c1_btn._color,
                      'color2_rgb': self._duo_c2_btn._color}
        elif key == 'curves':
            kwargs = {'red_boost': float(self._rb_sl.value()),
                      'blue_shadow_drop': float(self._bs_sl.value()),
                      'blue_mid_drop': float(self._bm_sl.value())}
        result = apply_sepia(self._source, key, **kwargs)
        self._emit_result(result)

    def _reset(self):
        if self._source is not None:
            self._emit_result(self._source.copy())


# ===========================================================================
# 3. Панель редактирования Ч/Б
# ===========================================================================

class EditingPanel(_BasePanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Левая часть — список инструментов
        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        main_layout.addWidget(_section_title('Редактирование Ч/Б'))

        # --- Группы инструментов ---
        groups = {
            'Тон':            ['brightness_contrast', 'levels', 'curves',
                               'histogram_eq', 'clahe'],
            'Детали':         ['unsharp_mask', 'local_contrast'],
            'Шум':            ['denoise_nlmeans', 'denoise_bilateral',
                               'denoise_gaussian', 'denoise_median'],
            'Художественные': ['vignette', 'film_grain', 'split_toning'],
            'Геометрия':      ['rotate', 'flip'],
        }

        self._tool_btns: dict[str, QPushButton] = {}
        self._stack = QStackedWidget()

        for group_name, tools in groups.items():
            gb = QGroupBox(group_name)
            gb_layout = QVBoxLayout(gb)
            gb_layout.setContentsMargins(4, 8, 4, 4)
            gb_layout.setSpacing(2)

            for tool_key in tools:
                info = EDIT_TOOLS.get(tool_key)
                if not info:
                    continue
                btn = QPushButton(info['label'])
                btn.setCheckable(True)
                btn.setProperty('tool_key', tool_key)
                btn.clicked.connect(self._on_tool_btn)
                gb_layout.addWidget(btn)
                self._tool_btns[tool_key] = btn

            main_layout.addWidget(gb)

        main_layout.addWidget(_hline())

        # --- Динамические настройки ---
        main_layout.addWidget(QLabel('Настройки:'))
        main_layout.addWidget(self._stack)

        # Страницы настроек
        self._pages: dict[str, int] = {}
        self._add_pages()

        main_layout.addWidget(_hline())

        btn_row = QHBoxLayout()
        self._apply_btn = _apply_primary_btn('Применить')
        self._reset_btn = _reset_btn()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        main_layout.addLayout(btn_row)
        main_layout.addStretch(1)

        scroll = _wrap_scroll(content)
        outer.addWidget(scroll)

        self._apply_btn.clicked.connect(self._apply)
        self._reset_btn.clicked.connect(self._reset)
        self._current_tool: str = ''

    def _add_pages(self):
        def _add(key, widget):
            idx = self._stack.addWidget(widget)
            self._pages[key] = idx

        # --- Яркость / Контраст ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        br_row, self._bright_sl = _slider_row(p, 'Яркость:', -150, 150, 0)
        ct_row, self._contr_sl  = _slider_row(p, 'Контраст:', -100, 100, 0)
        l.addLayout(br_row); l.addLayout(ct_row)
        _add('brightness_contrast', p)

        # --- Уровни ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        bi_row, self._blk_in_sl   = _slider_row(p, 'Вх. чёрная:', 0, 254, 0)
        wi_row, self._wht_in_sl   = _slider_row(p, 'Вх. белая:', 1, 255, 255)
        gm_row, self._gamma_sl    = _slider_row(p, 'Гамма x10:', 1, 100, 10)
        bo_row, self._blk_out_sl  = _slider_row(p, 'Вых. чёрная:', 0, 254, 0)
        wo_row, self._wht_out_sl  = _slider_row(p, 'Вых. белая:', 1, 255, 255)
        for row in [bi_row, wi_row, gm_row, bo_row, wo_row]:
            l.addLayout(row)
        _add('levels', p)

        # --- Кривые (упрощённые: 5 точек) ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        l.addWidget(QLabel('Кривая яркости (вход→выход):'))
        self._curve_sls: list[tuple[QSlider, QLabel]] = []
        curve_labels = ['0 (тени)', '64', '128 (средн.)', '192', '255 (света)']
        for ci, (inp, lbl_text) in enumerate(zip([0, 64, 128, 192, 255], curve_labels)):
            row, sl = _slider_row(p, lbl_text + ':', 0, 255, inp)
            l.addLayout(row)
            self._curve_sls.append((sl, None))
        _add('curves', p)

        # --- Эквализация гистограммы ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        l.addWidget(QLabel('Нет настроек.\nНажмите «Применить».'))
        _add('histogram_eq', p)

        # --- CLAHE ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        cl_row, self._clahe_clip_sl = _slider_row(p, 'Clip Limit x10:', 5, 80, 20)
        ts_row, self._clahe_tile_sl = _slider_row(p, 'Размер тайла:', 2, 16, 8)
        l.addLayout(cl_row); l.addLayout(ts_row)
        _add('clahe', p)

        # --- Нерезкое маскирование ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        ur_row, self._usm_r_sl = _slider_row(p, 'Радиус x10:', 1, 100, 15)
        ua_row, self._usm_a_sl = _slider_row(p, 'Сила x10:', 1, 50, 12)
        ut_row, self._usm_t_sl = _slider_row(p, 'Порог:', 0, 50, 3)
        l.addLayout(ur_row); l.addLayout(ua_row); l.addLayout(ut_row)
        _add('unsharp_mask', p)

        # --- Локальный контраст ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        lc_row, self._lc_r_sl = _slider_row(p, 'Радиус:', 10, 200, 60)
        ls_row, self._lc_s_sl = _slider_row(p, 'Сила x10:', 1, 30, 5)
        l.addLayout(lc_row); l.addLayout(ls_row)
        _add('local_contrast', p)

        # --- NLMeans ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        nl_row, self._nl_sl = _slider_row(p, 'Сила:', 1, 40, 10)
        l.addLayout(nl_row)
        _add('denoise_nlmeans', p)

        # --- Bilateral ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        bd_row, self._bil_d_sl  = _slider_row(p, 'Диаметр:', 3, 20, 9)
        bc_row, self._bil_c_sl  = _slider_row(p, 'Sigma цвет:', 10, 150, 75)
        bs_row, self._bil_sp_sl = _slider_row(p, 'Sigma простр.:', 10, 150, 75)
        l.addLayout(bd_row); l.addLayout(bc_row); l.addLayout(bs_row)
        _add('denoise_bilateral', p)

        # --- Gaussian ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        gs_row, self._gauss_sl = _slider_row(p, 'Сила x10:', 1, 50, 15)
        l.addLayout(gs_row)
        _add('denoise_gaussian', p)

        # --- Median ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        mk_row, self._med_sl = _slider_row(p, 'Ядро (нечётн.):', 3, 15, 3)
        l.addLayout(mk_row)
        _add('denoise_median', p)

        # --- Виньетка ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        vs_row, self._vig_s_sl = _slider_row(p, 'Сила x100:', 1, 100, 60)
        vr_row, self._vig_r_sl = _slider_row(p, 'Радиус x100:', 10, 100, 65)
        vf_row, self._vig_f_sl = _slider_row(p, 'Мягкость x10:', 1, 10, 3)
        l.addLayout(vs_row); l.addLayout(vr_row); l.addLayout(vf_row)
        _add('vignette', p)

        # --- Плёночное зерно ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        gi_row, self._grain_i_sl = _slider_row(p, 'Интенсивность:', 1, 50, 15)
        gsc_row, self._grain_s_sl = _slider_row(p, 'Масштаб x10:', 5, 30, 10)
        l.addLayout(gi_row); l.addLayout(gsc_row)
        _add('film_grain', p)

        # --- Раздельное тонирование ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        l.addWidget(QLabel('Цвет теней:'))
        sh_r = QHBoxLayout(); self._st_shadow_btn = _color_button(p, (30, 20, 40))
        sh_r.addWidget(self._st_shadow_btn); sh_r.addStretch(); l.addLayout(sh_r)
        l.addWidget(QLabel('Цвет светов:'))
        hl_r = QHBoxLayout(); self._st_hi_btn = _color_button(p, (255, 240, 200))
        hl_r.addWidget(self._st_hi_btn); hl_r.addStretch(); l.addLayout(hl_r)
        ss_row, self._st_ss_sl = _slider_row(p, 'Тени x100:', 0, 100, 30)
        sh_row, self._st_hs_sl = _slider_row(p, 'Света x100:', 0, 100, 30)
        sb_row, self._st_b_sl  = _slider_row(p, 'Баланс:', -10, 10, 0)
        l.addLayout(ss_row); l.addLayout(sh_row); l.addLayout(sb_row)
        self._st_shadow_btn.clicked.connect(lambda: _pick_color(self._st_shadow_btn, self))
        self._st_hi_btn.clicked.connect(lambda: _pick_color(self._st_hi_btn, self))
        _add('split_toning', p)

        # --- Поворот ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        an_row, self._rot_sl = _slider_row(p, 'Угол (-180..180):', -180, 180, 0)
        exp_cb = QCheckBox('Расширить холст')
        exp_cb.setChecked(True)
        self._rot_expand = exp_cb
        l.addLayout(an_row); l.addWidget(exp_cb)
        _add('rotate', p)

        # --- Отражение ---
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        self._flip_grp = QButtonGroup(p)
        for i, (val, text) in enumerate([
                ('horizontal', 'По горизонтали (←→)'),
                ('vertical',   'По вертикали (↑↓)'),
                ('both',       'Оба направления')]):
            rb = QRadioButton(text)
            rb.setProperty('flip_dir', val)
            if i == 0:
                rb.setChecked(True)
            self._flip_grp.addButton(rb, i)
            l.addWidget(rb)
        _add('flip', p)

        # Пустая страница по умолчанию
        self._stack.addWidget(QWidget())
        self._stack.setCurrentIndex(self._stack.count() - 1)

    def _on_tool_btn(self):
        btn = self.sender()
        key = btn.property('tool_key')

        # Снимаем выделение со всех кнопок
        for k, b in self._tool_btns.items():
            b.setChecked(k == key)

        self._current_tool = key
        if key in self._pages:
            self._stack.setCurrentIndex(self._pages[key])

    def _apply(self):
        if self._source is None or not self._current_tool:
            return
        key = self._current_tool

        try:
            if key == 'brightness_contrast':
                result = adjust_brightness_contrast(
                    self._source,
                    brightness=self._bright_sl.value(),
                    contrast=self._contr_sl.value())
            elif key == 'levels':
                result = adjust_levels(
                    self._source,
                    black_in=self._blk_in_sl.value(),
                    white_in=self._wht_in_sl.value(),
                    gamma=self._gamma_sl.value() / 10.0,
                    black_out=self._blk_out_sl.value(),
                    white_out=self._wht_out_sl.value())
            elif key == 'curves':
                inputs = [0, 64, 128, 192, 255]
                pts = [(inputs[i], self._curve_sls[i][0].value())
                       for i in range(5)]
                result = adjust_curves(self._source, pts)
            elif key == 'histogram_eq':
                result = histogram_equalization(self._source)
            elif key == 'clahe':
                result = clahe(self._source,
                               clip_limit=self._clahe_clip_sl.value() / 10.0,
                               tile_size=self._clahe_tile_sl.value())
            elif key == 'unsharp_mask':
                result = unsharp_mask(self._source,
                                      radius=self._usm_r_sl.value() / 10.0,
                                      amount=self._usm_a_sl.value() / 10.0,
                                      threshold=self._usm_t_sl.value())
            elif key == 'local_contrast':
                result = local_contrast(self._source,
                                        radius=self._lc_r_sl.value(),
                                        strength=self._lc_s_sl.value() / 10.0)
            elif key == 'denoise_nlmeans':
                result = denoise_nlmeans(self._source, strength=self._nl_sl.value())
            elif key == 'denoise_bilateral':
                result = denoise_bilateral(self._source,
                                           diameter=self._bil_d_sl.value(),
                                           sigma_color=float(self._bil_c_sl.value()),
                                           sigma_space=float(self._bil_sp_sl.value()))
            elif key == 'denoise_gaussian':
                result = denoise_gaussian(self._source, strength=self._gauss_sl.value() / 10.0)
            elif key == 'denoise_median':
                result = denoise_median(self._source, kernel_size=self._med_sl.value())
            elif key == 'vignette':
                result = add_vignette(self._source,
                                      strength=self._vig_s_sl.value() / 100.0,
                                      radius=self._vig_r_sl.value() / 100.0,
                                      feather=self._vig_f_sl.value() / 10.0)
            elif key == 'film_grain':
                result = add_film_grain(self._source,
                                        intensity=float(self._grain_i_sl.value()),
                                        scale=self._grain_s_sl.value() / 10.0)
            elif key == 'split_toning':
                result = split_toning(self._source,
                                      shadow_rgb=self._st_shadow_btn._color,
                                      highlight_rgb=self._st_hi_btn._color,
                                      shadow_strength=self._st_ss_sl.value() / 100.0,
                                      highlight_strength=self._st_hs_sl.value() / 100.0,
                                      balance=self._st_b_sl.value() / 10.0)
            elif key == 'rotate':
                result = rotate_image(self._source,
                                      angle=float(self._rot_sl.value()),
                                      expand=self._rot_expand.isChecked())
            elif key == 'flip':
                btn = self._flip_grp.checkedButton()
                direction = btn.property('flip_dir') if btn else 'horizontal'
                result = flip_image(self._source, direction=direction)
            else:
                return
        except Exception as e:
            print(f'Ошибка применения инструмента {key}: {e}')
            return

        self._emit_result(result)

    def _reset(self):
        if self._source is not None:
            self._emit_result(self._source.copy())


# ===========================================================================
# 4. Панель инфракрасной обработки
# ===========================================================================

class InfraredPanel(_BasePanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(_section_title('Инфракрасная обработка'))

        # Пояснение
        info_lbl = QLabel(
            'Для подлинных ИК-снимков (камеры с ИК-конвертацией\n'
            'или ИК-фильтром) — начните с «R↔B Обмен каналов».\n'
            'Для симуляции ИК на обычных снимках — выберите фильтр.'
        )
        info_lbl.setObjectName('infoLabel')
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        self._combo = QComboBox()

        # Добавляем группами через разделители
        groups = [
            ('── Конвертация ──', []),
            (None, ['channel_swap', 'to_bw_r', 'to_bw_false', 'channel_mixer']),
            ('── Симуляция фильтров ──', []),
            (None, ['simulate_590nm', 'simulate_720nm', 'simulate_850nm', 'wood_effect']),
            ('── Художественные ──', []),
            (None, ['high_contrast', 'dreamy_glow', 'bleach_bypass']),
            ('── Тонирование ──', []),
            (None, ['warm_tone', 'cool_tone', 'cyanotype']),
        ]

        for label, keys in groups:
            if label:
                self._combo.insertSeparator(self._combo.count())
                # Вставляем нередактируемый заголовок
                self._combo.addItem(label, '__sep__')
                model = self._combo.model()
                item = model.item(self._combo.count() - 1)
                item.setEnabled(False)
            else:
                for key in keys:
                    name = IR_METHODS[key][1]
                    self._combo.addItem(name, key)

        layout.addWidget(QLabel('Метод:'))
        layout.addWidget(self._combo)

        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName('methodDesc')
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(50)
        layout.addWidget(self._desc_lbl)

        layout.addWidget(_hline())

        # --- Динамические настройки ---
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._pages: dict[str, int] = {}
        self._add_ir_pages()

        layout.addWidget(_hline())

        btn_row = QHBoxLayout()
        self._apply_btn = _apply_primary_btn('Применить')
        self._reset_btn = _reset_btn()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        scroll = _wrap_scroll(content)
        outer.addWidget(scroll)

        self._combo.currentIndexChanged.connect(self._on_method_changed)
        self._apply_btn.clicked.connect(self._apply)
        self._reset_btn.clicked.connect(self._reset)

        # Выбираем первый активный пункт
        for i in range(self._combo.count()):
            if self._combo.itemData(i) not in (None, '__sep__'):
                self._combo.setCurrentIndex(i)
                break

    def _add_ir_pages(self):
        def _add(key, widget):
            idx = self._stack.addWidget(widget)
            self._pages[key] = idx

        # Без настроек
        for key in ['channel_swap', 'to_bw_r', 'to_bw_false',
                    'simulate_590nm', 'simulate_850nm', 'cyanotype']:
            p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
            l.addWidget(QLabel('Нет настроек.'))
            _add(key, p)

        # channel_mixer
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        l.addWidget(QLabel('Веса каналов (для ИК):'))
        rr, self._ir_r_sl = _slider_row(p, 'R (ИК):', 0, 100, 60)
        gr, self._ir_g_sl = _slider_row(p, 'G:', 0, 100, 30)
        br, self._ir_b_sl = _slider_row(p, 'B:', 0, 100, 10)
        l.addLayout(rr); l.addLayout(gr); l.addLayout(br)
        _add('channel_mixer', p)

        # simulate_720nm
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        cr, self._ir720_c_sl = _slider_row(p, 'Контраст x10:', 8, 25, 12)
        l.addLayout(cr)
        _add('simulate_720nm', p)

        # wood_effect
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        wr, self._wood_b_sl = _slider_row(p, 'Яркость x10:', 7, 20, 11)
        l.addLayout(wr)
        _add('wood_effect', p)

        # high_contrast
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        bpr, self._hc_bp_sl = _slider_row(p, 'Чёрная точка %:', 0, 40, 15)
        wpr, self._hc_wp_sl = _slider_row(p, 'Белая точка %:', 60, 100, 95)
        l.addLayout(bpr); l.addLayout(wpr)
        _add('high_contrast', p)

        # dreamy_glow
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        gsr, self._glow_s_sl = _slider_row(p, 'Сила x100:', 5, 80, 35)
        grr, self._glow_r_sl = _slider_row(p, 'Радиус:', 5, 60, 25)
        gtr, self._glow_t_sl = _slider_row(p, 'Порог %:', 30, 90, 60)
        l.addLayout(gsr); l.addLayout(grr); l.addLayout(gtr)
        _add('dreamy_glow', p)

        # bleach_bypass
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        bsr, self._bb_s_sl = _slider_row(p, 'Сила x10:', 1, 10, 6)
        l.addLayout(bsr)
        _add('bleach_bypass', p)

        # warm_tone
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        wtr, self._warm_sl = _slider_row(p, 'Теплота x100:', 5, 50, 15)
        l.addLayout(wtr)
        _add('warm_tone', p)

        # cool_tone
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0,0,0,0)
        clr, self._cool_sl = _slider_row(p, 'Холодность x100:', 5, 50, 15)
        l.addLayout(clr)
        _add('cool_tone', p)

        # Пустая страница по умолчанию
        self._stack.addWidget(QWidget())
        self._stack.setCurrentIndex(self._stack.count() - 1)

    def _on_method_changed(self):
        key = self._combo.currentData()
        if not key or key == '__sep__':
            return
        self._desc_lbl.setText(get_ir_description(key))
        if key in self._pages:
            self._stack.setCurrentIndex(self._pages[key])

    def _apply(self):
        if self._source is None:
            return
        key = self._combo.currentData()
        if not key or key == '__sep__':
            return
        kwargs = {}

        if key == 'channel_mixer':
            wr = self._ir_r_sl.value() / 100.0
            wg = self._ir_g_sl.value() / 100.0
            wb = self._ir_b_sl.value() / 100.0
            total = wr + wg + wb
            if total > 0:
                wr /= total; wg /= total; wb /= total
            kwargs = {'wr': wr, 'wg': wg, 'wb': wb}
        elif key == 'simulate_720nm':
            kwargs = {'contrast': self._ir720_c_sl.value() / 10.0}
        elif key == 'wood_effect':
            kwargs = {'brightness': self._wood_b_sl.value() / 10.0}
        elif key == 'high_contrast':
            kwargs = {'black_point': self._hc_bp_sl.value() / 100.0,
                      'white_point': self._hc_wp_sl.value() / 100.0}
        elif key == 'dreamy_glow':
            kwargs = {'glow_strength': self._glow_s_sl.value() / 100.0,
                      'glow_radius': float(self._glow_r_sl.value()),
                      'threshold': self._glow_t_sl.value() / 100.0}
        elif key == 'bleach_bypass':
            kwargs = {'strength': self._bb_s_sl.value() / 10.0}
        elif key == 'warm_tone':
            kwargs = {'warmth': self._warm_sl.value() / 100.0}
        elif key == 'cool_tone':
            kwargs = {'coolness': self._cool_sl.value() / 100.0}

        try:
            result = apply_ir(self._source, key, **kwargs)
            self._emit_result(result)
        except Exception as e:
            print(f'Ошибка ИК-обработки: {e}')

    def _reset(self):
        if self._source is not None:
            self._emit_result(self._source.copy())
