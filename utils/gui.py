"""
utils/gui.py
============
Tkinter-based GUI that embeds the live OpenCV frame alongside a
statistics panel.  The video capture and detection logic remain in
main.py; this module provides the window, canvas, and stat labels.

Architecture
------------
The GUI runs on the main thread (Tkinter requirement).
A background thread (in main.py) processes frames and pushes results
into a thread-safe queue.  The GUI polls that queue every ~15 ms.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


class DrowsinessGUI:
    """
    Main application window.

    Parameters
    ----------
    title        : str       Window title.
    frame_queue  : queue.Queue
                   Producer (detection thread) puts (frame, stats_dict) tuples here.
                   Consumer (GUI) reads them every poll_interval_ms.
    on_quit      : Callable  Called when the user closes the window.
    on_screenshot: Callable  Called when the user clicks "Save Screenshot".
    poll_interval_ms : int   How often (ms) to pull frames from the queue.
    """

    def __init__(
        self,
        title: str = "AI Driver Drowsiness Detection System",
        frame_queue: Optional[queue.Queue] = None,
        on_quit: Optional[Callable] = None,
        on_screenshot: Optional[Callable] = None,
        poll_interval_ms: int = 15,
    ):
        self._frame_queue      = frame_queue or queue.Queue(maxsize=2)
        self._on_quit          = on_quit or (lambda: None)
        self._on_screenshot    = on_screenshot or (lambda: None)
        self._poll_interval    = poll_interval_ms
        self._running          = False

        # ── Build the Tk window ──────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._handle_quit)
        self.root.configure(bg="#1e1e2e")

        self._build_layout()
        self._bind_keys()

    # ── Layout construction ──────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build all widgets."""
        DARK_BG   = "#1e1e2e"
        PANEL_BG  = "#2a2a3e"
        TEXT_FG   = "#cdd6f4"
        ACCENT    = "#89b4fa"
        RED       = "#f38ba8"
        GREEN     = "#a6e3a1"
        YELLOW    = "#f9e2af"
        MAUVE     = "#cba6f7"

        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#11111b", pady=6)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🚗  AI Driver Drowsiness Detection System",
            font=("Helvetica", 15, "bold"),
            fg=ACCENT,
            bg="#11111b",
        ).pack()

        # ── Body row (video | stats) ─────────────────────────────────────
        body = tk.Frame(self.root, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        # Video canvas (640×480)
        self._canvas = tk.Canvas(
            body, width=640, height=480,
            bg="black", highlightthickness=2,
            highlightbackground=ACCENT,
        )
        self._canvas.pack(side="left", padx=(0, 8))

        # Stats panel
        stats_frame = tk.Frame(body, bg=PANEL_BG, bd=0,
                               width=240, padx=14, pady=14)
        stats_frame.pack(side="left", fill="y")
        stats_frame.pack_propagate(False)

        def section(parent, text):
            tk.Label(parent, text=text, font=("Helvetica", 10, "bold"),
                     fg=ACCENT, bg=PANEL_BG).pack(anchor="w", pady=(10, 2))
            ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=2)

        def stat_row(parent, label: str, var: tk.StringVar,
                     color: str = TEXT_FG) -> tk.Label:
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Courier", 9),
                     fg=TEXT_FG, bg=PANEL_BG, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, textvariable=var,
                           font=("Courier", 9, "bold"),
                           fg=color, bg=PANEL_BG)
            lbl.pack(side="left")
            return lbl

        # ── Real-time metrics ────────────────────────────────────────────
        section(stats_frame, "Real-time Metrics")

        self._ear_var = tk.StringVar(value="—")
        self._mar_var = tk.StringVar(value="—")
        self._fps_var = tk.StringVar(value="—")

        self._ear_lbl = stat_row(stats_frame, "EAR :", self._ear_var, GREEN)
        self._mar_lbl = stat_row(stats_frame, "MAR :", self._mar_var, YELLOW)
        stat_row(stats_frame, "FPS :", self._fps_var, ACCENT)

        # ── Status flags ─────────────────────────────────────────────────
        section(stats_frame, "Detection Status")

        self._face_var  = tk.StringVar(value="—")
        self._eyes_var  = tk.StringVar(value="—")
        self._yawn_var  = tk.StringVar(value="—")
        self._alarm_var = tk.StringVar(value="OFF")
        self._light_var = tk.StringVar(value="OK")

        self._face_lbl  = stat_row(stats_frame, "Face   :", self._face_var)
        self._eyes_lbl  = stat_row(stats_frame, "Eyes   :", self._eyes_var)
        self._yawn_lbl  = stat_row(stats_frame, "Yawn   :", self._yawn_var)
        self._alarm_lbl = stat_row(stats_frame, "ALARM  :", self._alarm_var, RED)
        stat_row(stats_frame, "Light  :", self._light_var)

        # ── Session totals ────────────────────────────────────────────────
        section(stats_frame, "Session Totals")

        self._blink_var = tk.StringVar(value="0")
        self._tyawn_var = tk.StringVar(value="0")
        self._tdrow_var = tk.StringVar(value="0")

        stat_row(stats_frame, "Blinks :", self._blink_var, MAUVE)
        stat_row(stats_frame, "Yawns  :", self._tyawn_var, MAUVE)
        stat_row(stats_frame, "Drowsy :", self._tdrow_var, RED)

        # ── Alarm status banner ───────────────────────────────────────────
        self._alert_banner = tk.Label(
            stats_frame, text="",
            font=("Helvetica", 11, "bold"),
            fg="#1e1e2e", bg=PANEL_BG, pady=4,
        )
        self._alert_banner.pack(fill="x", pady=(14, 0))

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(stats_frame, bg=PANEL_BG)
        btn_frame.pack(fill="x", pady=(16, 0))

        btn_style = dict(
            font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6, bd=0,
        )

        tk.Button(
            btn_frame, text="📸  Save Screenshot",
            bg="#45475a", fg=TEXT_FG,
            command=self._on_screenshot,
            **btn_style,
        ).pack(fill="x", pady=3)

        tk.Button(
            btn_frame, text="❌  Quit (Q)",
            bg=RED, fg="#1e1e2e",
            command=self._handle_quit,
            **btn_style,
        ).pack(fill="x", pady=3)

        # ── Footer ────────────────────────────────────────────────────────
        footer = tk.Frame(self.root, bg="#11111b", pady=4)
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="Press  Q = Quit   S = Screenshot",
            font=("Helvetica", 8),
            fg="#585b70",
            bg="#11111b",
        ).pack()

    def _bind_keys(self) -> None:
        """Register keyboard shortcuts."""
        self.root.bind("<q>", lambda e: self._handle_quit())
        self.root.bind("<Q>", lambda e: self._handle_quit())
        self.root.bind("<s>", lambda e: self._on_screenshot())
        self.root.bind("<S>", lambda e: self._on_screenshot())

    # ── Frame display ────────────────────────────────────────────────────────

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convert BGR OpenCV frame → Tkinter PhotoImage and draw on canvas."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image)
        self._canvas.create_image(0, 0, anchor="nw", image=photo)
        self._canvas.image = photo   # keep reference; prevents GC

    # ── Stat update ──────────────────────────────────────────────────────────

    def _update_stats(self, stats: dict) -> None:
        """Refresh all stat labels from the *stats* dictionary."""
        GREEN = "#a6e3a1"
        RED   = "#f38ba8"
        YELL  = "#f9e2af"

        ear = stats.get("ear", 0.0)
        mar = stats.get("mar", 0.0)

        self._ear_var.set(f"{ear:.3f}")
        self._mar_var.set(f"{mar:.3f}")
        self._fps_var.set(f"{stats.get('fps', 0.0):.1f}")

        self._ear_lbl.configure(fg=RED if ear < 0.25 else GREEN)
        self._mar_lbl.configure(fg=YELL if mar > 0.75 else GREEN)

        face_det = stats.get("face_detected", False)
        self._face_var.set("YES" if face_det else "NO")
        self._face_lbl.configure(fg=GREEN if face_det else RED)

        eyes_cl = stats.get("eyes_closed", False)
        self._eyes_var.set("CLOSED" if eyes_cl else "OPEN")
        self._eyes_lbl.configure(fg=RED if eyes_cl else GREEN)

        yawning = stats.get("yawning", False)
        self._yawn_var.set("YES" if yawning else "NO")
        self._yawn_lbl.configure(fg=YELL if yawning else GREEN)

        alarm = stats.get("alarm_on", False)
        self._alarm_var.set("🔔 ACTIVE" if alarm else "OFF")
        self._alarm_lbl.configure(fg=RED if alarm else GREEN)

        light = stats.get("low_light", False)
        self._light_var.set("⚠ LOW" if light else "OK")

        self._blink_var.set(str(stats.get("blink_count", 0)))
        self._tyawn_var.set(str(stats.get("yawn_count",  0)))
        self._tdrow_var.set(str(stats.get("drowsy_total", 0)))

        # Alert banner
        if alarm:
            self._alert_banner.configure(
                text="⚠  DRIVER DROWSY  ⚠",
                fg="#1e1e2e", bg=RED,
            )
        else:
            self._alert_banner.configure(text="", bg="#2a2a3e")

    # ── Main poll loop ───────────────────────────────────────────────────────

    def _poll(self) -> None:
        """
        Drain up to one item from *frame_queue* and refresh the GUI.
        Reschedules itself every *poll_interval_ms* milliseconds.
        """
        if not self._running:
            return

        try:
            frame, stats = self._frame_queue.get_nowait()
            self._display_frame(frame)
            self._update_stats(stats)
        except queue.Empty:
            pass
        except Exception as exc:
            pass   # don't crash the GUI on a transient decode error

        self.root.after(self._poll_interval, self._poll)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _handle_quit(self) -> None:
        """Stop the poll loop, call the quit callback, destroy the window."""
        self._running = False
        self._on_quit()
        self.root.destroy()

    def start(self) -> None:
        """
        Start the GUI main loop.  Blocks until the window is closed.
        """
        self._running = True
        self.root.after(self._poll_interval, self._poll)
        self.root.mainloop()

    @property
    def frame_queue(self) -> queue.Queue:
        """The queue that the detection thread should push frames into."""
        return self._frame_queue
