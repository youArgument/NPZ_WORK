"""Простой просмотрщик магнитограмм с YOLO-детекцией по выбранному сектору."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Reff"))

from EntryPoint_temp import _prepare_image, search
from schemas import (
    COLOR_MAP_BY_LABEL,
    NAME_MAP_BY_LABEL,
    DataBatch,
    SearchParams,
    TargetData,
    TargetType,
)


DEFAULT_DATA_ROOT = ROOT / "ROW_DATA" / "df"


def _find_default_npz() -> Path | None:
    if not DEFAULT_DATA_ROOT.exists():
        return None
    return next(DEFAULT_DATA_ROOT.rglob("*.npz"), None)


def _load_npz(npz_path: Path) -> tuple[DataBatch, SearchParams]:
    with np.load(npz_path) as npz:
        batch = DataBatch(
            npz["magnetogram"],
            npz["velocity"],
            npz["accelerometer"],
            npz["orientation"],
        )
        odomstep = float(npz.get("odomstep", 0.002))
    return batch, SearchParams(StartDistance=0.0, OdomStep=odomstep)


def _target_to_plot_pixels(points: np.ndarray, params: SearchParams, sensor_count: int) -> np.ndarray:
    out = np.zeros_like(points, dtype=np.float64)
    out[:, 0] = points[:, 0] / params.OdomStep
    out[:, 1] = points[:, 1] * sensor_count / 12.0
    out[:, 2] = points[:, 2] / params.OdomStep
    out[:, 3] = points[:, 3] * sensor_count / 12.0
    return out


class SectorViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Просмотр магнитограмм с YOLO по секторам")
        self.geometry("1300x850")

        self.npz_path: Path | None = None
        self.batch: DataBatch | None = None
        self.params: SearchParams | None = None
        self.sectors: list[tuple[int, int]] = []
        self.current_result: TargetData | None = None

        self.conf_var = tk.DoubleVar(value=0.02)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.sector_size_var = tk.IntVar(value=800)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.rotation_var = tk.IntVar(value=0)
        # Tiling params for YOLO on big images
        self.tile_size_var = tk.IntVar(value=640)
        self.overlap_ratio_var = tk.DoubleVar(value=0.2)
        self.normalize_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Загрузите .npz файл")


        self._build_ui()

        default_npz = _find_default_npz()
        if default_npz is not None:
            self.load_file(default_npz)

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        panel = ttk.Frame(root, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(panel, text="Открыть .npz", command=self.open_file).pack(fill=tk.X)
        ttk.Button(panel, text="Обновить секторы", command=self.rebuild_sectors).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(panel, text="Детекция YOLO", command=self.detect_selected_sector).pack(fill=tk.X, pady=(6, 12))
        
        ttk.Checkbutton(panel, text="Нормализация", variable=self.normalize_var).pack(anchor=tk.W)

        ttk.Label(panel, text="Tile size (плитка, пиксели)").pack(anchor=tk.W, pady=(8, 0))
        ttk.Spinbox(
            panel,
            from_=128,
            to=4096,
            increment=64,
            textvariable=self.tile_size_var,
            width=10,
        ).pack(fill=tk.X, pady=(0, 8))

        self._add_slider(panel, "overlap", self.overlap_ratio_var, 0.0, 0.75)

        self.file_label = ttk.Label(panel, text="Файл не выбран", wraplength=260)
        self.file_label.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(panel, text="Размер сектора, строк").pack(anchor=tk.W)
        self.sector_spin = ttk.Spinbox(
            panel,
            from_=100,
            to=20000,
            increment=100,
            textvariable=self.sector_size_var,
            command=self.rebuild_sectors,
            width=10,
        )
        self.sector_spin.pack(fill=tk.X, pady=(0, 8))

        self._add_slider(panel, "conf", self.conf_var, 0.001, 0.5)
        self._add_slider(panel, "iou", self.iou_var, 0.05, 0.95)
        self._add_slider(panel, "зум сектора", self.zoom_var, 1.0, 6.0)

        ttk.Label(panel, text="Поворот YOLO").pack(anchor=tk.W, pady=(8, 0))
        rotation_box = ttk.Combobox(
            panel,
            textvariable=self.rotation_var,
            values=(0, 90, 180, 270),
            state="readonly",
            width=8,
        )
        rotation_box.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(panel, text="Секторы").pack(anchor=tk.W)
        list_frame = ttk.Frame(panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.sector_list = tk.Listbox(list_frame, width=34, exportselection=False)
        self.sector_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.sector_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sector_list.configure(yscrollcommand=scrollbar.set)
        self.sector_list.bind("<<ListboxSelect>>", lambda _event: self.plot_selected_sector())

        ttk.Label(panel, textvariable=self.status_var, wraplength=260).pack(fill=tk.X, pady=(12, 0))

        plot_area = ttk.Frame(root)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.overview_ax = self.figure.add_subplot(2, 1, 1)
        self.sector_ax = self.figure.add_subplot(2, 1, 2)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area)
        toolbar.update()

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        start: float,
        end: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 8))
        text_var = tk.StringVar()

        def refresh_text(*_args) -> None:
            text_var.set(f"{label}: {variable.get():.3f}")

        variable.trace_add("write", refresh_text)
        refresh_text()

        ttk.Label(row, textvariable=text_var).pack(anchor=tk.W)
        ttk.Scale(row, from_=start, to=end, variable=variable, orient=tk.HORIZONTAL).pack(fill=tk.X)

    def open_file(self) -> None:
        initial_dir = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else ROOT
        selected = filedialog.askopenfilename(
            title="Выберите .npz файл",
            initialdir=str(initial_dir),
            filetypes=(("NPZ files", "*.npz"), ("All files", "*.*")),
        )
        if selected:
            self.load_file(Path(selected))

    def load_file(self, npz_path: Path) -> None:
        try:
            self.batch, self.params = _load_npz(npz_path)
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        self.npz_path = npz_path
        self.current_result = None
        self.file_label.configure(text=str(npz_path))
        self.status_var.set(f"Загружено: shape={self.batch.Data.shape}, OdomStep={self.params.OdomStep}")
        self.rebuild_sectors()

    def rebuild_sectors(self) -> None:
        if self.batch is None:
            return

        size = max(1, int(self.sector_size_var.get()))
        rows = self.batch.Data.shape[0]
        self.sectors = [(start, min(start + size, rows)) for start in range(0, rows, size)]

        self.sector_list.delete(0, tk.END)
        for index, (start, end) in enumerate(self.sectors):
            dist_start = start * self.params.OdomStep if self.params else start
            dist_end = end * self.params.OdomStep if self.params else end
            self.sector_list.insert(
                tk.END,
                f"{index:03d}: строки {start}-{end}  |  {dist_start:.3f}-{dist_end:.3f}",
            )
        if self.sectors:
            self.sector_list.selection_set(0)
            self.sector_list.activate(0)
        self.current_result = None
        self.plot_selected_sector()

    def selected_sector_index(self) -> int | None:
        selection = self.sector_list.curselection()
        if not selection:
            return None
        return int(selection[0])

    def selected_sector(self) -> tuple[int, int] | None:
        index = self.selected_sector_index()
        if index is None or index >= len(self.sectors):
            return None
        return self.sectors[index]

    def detect_selected_sector(self) -> None:
        if self.batch is None or self.params is None:
            return
        sector = self.selected_sector()
        if sector is None:
            return

        start, end = sector
        sector_data = self.batch.Data[start:end, :]
        sector_batch = DataBatch(
            sector_data,
            self.batch.Velocity[start:end],
            self.batch.Accelerometer[start:end],
            self.batch.Orientation[start:end],
        )
        sector_params = SearchParams(
            StartDistance=0.0,
            OdomStep=self.params.OdomStep,
            confidence_threshold=float(self.conf_var.get()),
            iou_threshold=float(self.iou_var.get()),
            yolo_rotation=int(self.rotation_var.get()),
        )

        self.status_var.set("YOLO выполняет детекцию...")
        self.update_idletasks()

        try:
            self.current_result = search(
                sector_batch,
                sector_params,
                normalize=self.normalize_var.get(),
                tile_size=int(self.tile_size_var.get()),
                overlap_ratio=float(self.overlap_ratio_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Ошибка YOLO", str(exc))
            self.current_result = None
            return

        labels = self.current_result.labels if self.current_result else []
        counts: dict[TargetType, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        readable = ", ".join(f"{NAME_MAP_BY_LABEL.get(k, k.name)}: {v}" for k, v in counts.items())
        self.status_var.set(f"Готово. Детекций: {len(labels)}" + (f" ({readable})" if readable else ""))
        self.plot_selected_sector()

    def plot_selected_sector(self) -> None:
        if self.batch is None:
            return
        sector = self.selected_sector()
        if sector is None:
            return

        start, end = sector
        sector_data = self.batch.Data[start:end, :]
        self._plot_overview(start, end)
        self._plot_sector(sector_data)
        self.canvas.draw_idle()

    def _plot_overview(self, start: int, end: int) -> None:
        self.overview_ax.clear()
        image = _prepare_image(self.batch.Data, rotation=0, normalize=self.normalize_var.get())[:, :, 0].T
        self.overview_ax.imshow(image, cmap="gray", aspect="auto")
        for s, e in self.sectors:
            self.overview_ax.axvline(s, color="#2f80ff", linewidth=0.5, alpha=0.35)
        self.overview_ax.axvspan(start, end, color="#ff3333", alpha=0.22)
        self.overview_ax.set_title("Вся магнитограмма и выбранный сектор")
        self.overview_ax.set_ylabel("датчики")

    def _plot_sector(self, sector_data: np.ndarray) -> None:
        self.sector_ax.clear()
        image = _prepare_image(sector_data, rotation=0, normalize=self.normalize_var.get())[:, :, 0].T
        self.sector_ax.imshow(image, cmap="gray", aspect="auto")
        self.sector_ax.set_title("Выбранный сектор")
        self.sector_ax.set_xlabel("дистанция внутри сектора, строки")
        self.sector_ax.set_ylabel("датчики")

        # Визуализация квадратной сетки тайлов для выбранного сектора.
        # Координаты на графике соответствуют (x=distance/строки, y=датчики).
        tile_px = max(1, int(self.tile_size_var.get()))
        overlap = float(self.overlap_ratio_var.get())
        step = max(1, int(tile_px * (1.0 - overlap)))

        rows = sector_data.shape[0]  # по X (distance)
        cols = sector_data.shape[1]  # по Y (sensors)

        # Рисуем квадраты tile_px x tile_px с перекрытием.
        for x0 in range(0, rows, step):
            x1 = min(x0 + tile_px, rows)
            for y0 in range(0, cols, step):
                y1 = min(y0 + tile_px, cols)
                rect = patches.Rectangle(
                    (x0, y0),
                    max(x1 - x0, 1),
                    max(y1 - y0, 1),
                    linewidth=1,
                    edgecolor="#00b894",
                    facecolor="none",
                    alpha=0.35,
                    linestyle="--",
                )
                self.sector_ax.add_patch(rect)

        if self.current_result is not None and self.params is not None:
            self._draw_detections(sector_data.shape[1])

        zoom = max(1.0, float(self.zoom_var.get()))
        if zoom > 1.0:
            width = sector_data.shape[0]
            height = sector_data.shape[1]
            center_x = width / 2
            center_y = height / 2
            half_w = width / (2 * zoom)
            half_h = height / (2 * zoom)
            self.sector_ax.set_xlim(center_x - half_w, center_x + half_w)
            self.sector_ax.set_ylim(center_y + half_h, center_y - half_h)

    def _draw_detections(self, sensor_count: int) -> None:
        if self.current_result is None or self.params is None:
            return
        points = self.current_result.points
        if points.size == 0:
            return
    
        converted = _target_to_plot_pixels(points, self.params, sensor_count)
        
        # Цвет для всех объектов (можно изменить на любой другой)
        anon_color = "yellow"
    
        for bbox, _ in zip(converted, self.current_result.labels):
            x1, y1, x2, y2 = bbox
            if x1 == y1 == x2 == y2 == 0:
                continue
            
            rect = patches.Rectangle(
                (x1, y1),
                max(x2 - x1, 1),
                max(y2 - y1, 1),
                linewidth=2,
                edgecolor=anon_color,
                facecolor=anon_color,
                alpha=0.25,
            )
            self.sector_ax.add_patch(rect)


def main() -> None:
    app = SectorViewer()
    app.mainloop()


if __name__ == "__main__":
    main()
