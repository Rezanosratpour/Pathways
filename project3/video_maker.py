# Integrated: Create video from a (T,H,W) data sequence
# - backend="mpl"   -> MP4 via matplotlib+ffmpeg (best quality)
# - backend="opencv"-> MP4 via OpenCV (fast, no ffmpeg needed)
# - backend="gif"   -> GIF via imageio (no ffmpeg)
#
# Fixes included:
# 1) Auto-create output directory
# 2) Detect if ffmpeg exists; if missing and backend="mpl", fallback to opencv
# 3) Works on Windows paths like: r"D:/projects/.../precip.mp4"
# ============================================================

import os
import shutil
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import cv2
except Exception:
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def ensure_parent_dir(out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


class SequenceVideoMaker:
    """
    Create videos from a 3D array: data[t, y, x].
    Optionally pass lat/lon for geographic extent and timestamps for titles.

    data: np.ndarray shape (T,H,W)
    lat:  1D array length H (optional)
    lon:  1D array length W (optional)
    timestamps: list-like length T (optional), can be strings/datetimes
    """

    def __init__(self, data: np.ndarray, lat=None, lon=None, timestamps=None):
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"Expected data with shape (T,H,W). Got {data.shape}")

        self.data = data
        self.T, self.H, self.W = data.shape

        self.lat = None if lat is None else np.asarray(lat)
        self.lon = None if lon is None else np.asarray(lon)
        self.timestamps = timestamps

        if (self.lat is None) ^ (self.lon is None):
            raise ValueError("Provide both lat and lon, or neither.")

        if self.timestamps is not None and len(self.timestamps) != self.T:
            raise ValueError("timestamps length must equal T.")

    # ------------------- helpers -------------------
    def _compute_vmin_vmax(self, vmin, vmax, robust=False, q=(2, 98)):
        x = self.data
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            raise ValueError("Data contains no finite values.")

        if vmin is None or vmax is None:
            if robust:
                lo, hi = np.percentile(finite, q)
                vmin = lo if vmin is None else vmin
                vmax = hi if vmax is None else vmax
            else:
                vmin = np.nanmin(x) if vmin is None else vmin
                vmax = np.nanmax(x) if vmax is None else vmax

        vmin, vmax = float(vmin), float(vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            raise ValueError(f"Bad vmin/vmax: vmin={vmin}, vmax={vmax}")
        return vmin, vmax

    def _norm_to_uint8(self, vmin, vmax):
        x = np.clip(self.data, vmin, vmax)
        x = (x - vmin) / (vmax - vmin)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return (255 * x).astype(np.uint8)

    def _extent(self):
        if self.lat is None or self.lon is None:
            return None
        return [float(self.lon.min()), float(self.lon.max()), float(self.lat.min()), float(self.lat.max())]

    def _title_for_frame(self, t, title_prefix=""):
        if self.timestamps is None:
            return f"{title_prefix}t={t}"
        return f"{title_prefix}{self.timestamps[t]}"

    # ------------------- public API -------------------
    def export(
        self,
        out_path: str,
        backend: str = "mpl",
        fps: int = 10,
        cmap: str = "Blues",
        vmin=None,
        vmax=None,
        robust: bool = False,
        interval_ms: int | None = None,
        dpi: int = 200,
        bitrate: int = 5000,
        add_colorbar: bool = True,
        title_prefix: str = "",
        opencv_color: bool = True,   # OpenCV: colored frames or grayscale
        opencv_put_timestamp: bool = True,  # burn-in timestamp text if timestamps provided
    ):
        """
        Export animation.

        out_path: mp4 or gif path.
        backend:
          - "mpl":   MP4 via matplotlib+ffmpeg (needs ffmpeg in PATH)
          - "opencv":MP4 via OpenCV (no ffmpeg needed)
          - "gif":   GIF via imageio (no ffmpeg needed)

        If backend="mpl" and ffmpeg is missing -> auto-fallback to OpenCV (if installed).
        """

        backend = backend.lower().strip()
        ensure_parent_dir(out_path)

        if interval_ms is None:
            interval_ms = int(1000 / max(1, fps))

        vmin, vmax = self._compute_vmin_vmax(vmin, vmax, robust=robust)

        if backend == "mpl":
            try:
                self._export_mpl_mp4(
                    out_path=out_path,
                    fps=fps,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interval_ms=interval_ms,
                    dpi=dpi,
                    bitrate=bitrate,
                    add_colorbar=add_colorbar,
                    title_prefix=title_prefix,
                )
            except FileNotFoundError as e:
                # Most common on Windows: ffmpeg not found
                print(f"[WARN] {e}")
                print("[WARN] Falling back to backend='opencv' (MP4).")
                if cv2 is None:
                    raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
                self._export_opencv_mp4(
                    out_path=out_path,
                    fps=fps,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    color=opencv_color,
                    put_timestamp=opencv_put_timestamp,
                )

        elif backend == "opencv":
            if cv2 is None:
                raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
            self._export_opencv_mp4(
                out_path=out_path,
                fps=fps,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                color=opencv_color,
                put_timestamp=opencv_put_timestamp,
            )

        elif backend == "gif":
            if imageio is None:
                raise ImportError("imageio not installed. Install with: pip install imageio")
            self._export_gif(
                out_path=out_path,
                fps=fps,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        else:
            raise ValueError("backend must be one of: 'mpl', 'opencv', 'gif'")

        return out_path

    # ------------------- backends -------------------
    def _export_mpl_mp4(
        self, out_path, fps, cmap, vmin, vmax, interval_ms, dpi, bitrate, add_colorbar, title_prefix
    ):
        ext = os.path.splitext(out_path)[1].lower()
        if ext != ".mp4":
            raise ValueError("matplotlib backend exports MP4. Use out_path ending with .mp4")

        # Critical on Windows: ffmpeg must be available
        if not ffmpeg_exists():
            raise FileNotFoundError("ffmpeg not found in PATH. Install ffmpeg or use backend='opencv'/'gif'.")

        fig, ax = plt.subplots()
        extent = self._extent()

        im = ax.imshow(
            self.data[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower" if extent is not None else "upper",
        )

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(self._title_for_frame(0, title_prefix=title_prefix))
        if extent is not None:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        def update(t):
            im.set_array(self.data[t])
            ax.set_title(self._title_for_frame(t, title_prefix=title_prefix))
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=self.T, interval=interval_ms, blit=False)
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
        ani.save(out_path, writer=writer, dpi=dpi)
        plt.close(fig)

    def _export_opencv_mp4(self, out_path, fps, cmap, vmin, vmax, color=True, put_timestamp=True):
        ext = os.path.splitext(out_path)[1].lower()
        if ext != ".mp4":
            raise ValueError("opencv backend exports MP4. Use out_path ending with .mp4")

        u8 = self._norm_to_uint8(vmin, vmax)  # (T,H,W)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if color:
            # OpenCV colormaps are not identical to matplotlib's, but close.
            # If you need EXACT matplotlib colors + colorbar, use backend="mpl".
            cmap_id = cv2.COLORMAP_OCEAN if cmap.lower() == "blues" else cv2.COLORMAP_VIRIDIS

            first = cv2.applyColorMap(u8[0], cmap_id)
            h, w, _ = first.shape
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)

            for t in range(self.T):
                frame = cv2.applyColorMap(u8[t], cmap_id)

                if put_timestamp and (self.timestamps is not None):
                    txt = str(self.timestamps[t])
                    cv2.putText(
                        frame,
                        txt,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                vw.write(frame)
            vw.release()

        else:
            h, w = u8.shape[1], u8.shape[2]
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)
            for t in range(self.T):
                vw.write(u8[t])
            vw.release()

    def _export_gif(self, out_path, fps, cmap, vmin, vmax):
        ext = os.path.splitext(out_path)[1].lower()
        if ext != ".gif":
            raise ValueError("gif backend exports GIF. Use out_path ending with .gif")

        frames = []
        extent = self._extent()

        fig, ax = plt.subplots()
        im = ax.imshow(
            self.data[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower" if extent is not None else "upper",
        )
        ax.axis("off")

        for t in range(self.T):
            im.set_array(self.data[t])
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(rgb)

        plt.close(fig)
        imageio.mimsave(out_path, frames, fps=fps)


