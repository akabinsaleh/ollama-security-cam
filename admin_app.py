"""
admin_app.py — Security Camera Admin Dashboard

Run this script to view all detection events, snapshots, and video clips.

    python admin_app.py

Reads events from events.db (created automatically by security_cam.py).
Requires:  pip install opencv-python   (already needed by security_cam.py)
"""

import base64
import os
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import cv2

# Path to the SQLite database written by security_cam.py
DB_FILE      = "events.db"
CAPTURES_DIR = "captures"

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
BG       = "#0d1117"
BG2      = "#161b22"
BG3      = "#21262d"
ACCENT   = "#238636"   # green — used for "Play Video" button
ACCENT2  = "#1f6feb"   # blue  — selection highlight
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
ALERT_FG = "#f85149"   # red   — "no snapshot" warnings
BORDER   = "#30363d"


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AdminApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Security Camera — Admin Dashboard")
        self.root.geometry("1280x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG)

        self._photo: tk.PhotoImage | None = None    # prevent GC
        self._current_video:  str | None  = None
        self._current_folder: str | None  = None

        self._apply_styles()
        self._build_ui()
        self._refresh_events()

        # Auto-refresh the event list every 10 seconds
        self.root.after(10_000, self._auto_refresh)

    # ------------------------------------------------------------------
    # TTK styles
    # ------------------------------------------------------------------

    def _apply_styles(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")

        s.configure(".",
            background=BG, foreground=TEXT,
            fieldbackground=BG2, bordercolor=BORDER,
            troughcolor=BG2, insertcolor=TEXT,
            selectbackground=ACCENT2, selectforeground=TEXT,
            font=("Segoe UI", 10))

        s.configure("Treeview",
            background=BG2, foreground=TEXT,
            fieldbackground=BG2, rowheight=30, borderwidth=0,
            font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
            background=BG3, foreground=TEXT_DIM,
            relief="flat", font=("Segoe UI", 9, "bold"))
        s.map("Treeview",
            background=[("selected", ACCENT2)],
            foreground=[("selected", TEXT)])

        s.configure("TScrollbar",
            background=BG3, troughcolor=BG,
            bordercolor=BORDER, arrowcolor=TEXT_DIM, width=10)

        s.configure("TButton",
            background=BG3, foreground=TEXT, relief="flat",
            borderwidth=1, focusthickness=0, font=("Segoe UI", 10))
        s.map("TButton",
            background=[("active", ACCENT2)])

        s.configure("Green.TButton",
            background=ACCENT, foreground=TEXT, relief="flat",
            borderwidth=0, font=("Segoe UI", 10, "bold"))
        s.map("Green.TButton",
            background=[("active", "#2ea043")])

        s.configure("TSeparator", background=BORDER)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Title bar ────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=BG2, height=52)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="Security Camera  —  Admin Dashboard",
            bg=BG2, fg=TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left", padx=18, pady=12)

        self._status_var = tk.StringVar(value="")
        tk.Label(
            title_bar,
            textvariable=self._status_var,
            bg=BG2, fg=TEXT_DIM,
            font=("Segoe UI", 10),
        ).pack(side="right", padx=18)

        # ── Body ─────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Left panel — event list (fixed 460 px wide)
        left = tk.Frame(body, bg=BG2, width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Right panel — snapshot + details
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: tk.Frame) -> None:
        # Header row
        hdr = tk.Frame(parent, bg=BG3, height=42)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="Detection Events",
            bg=BG3, fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=12, pady=8)

        ttk.Button(
            hdr, text="Refresh", style="TButton",
            command=self._refresh_events,
        ).pack(side="right", padx=8, pady=6)

        # Thin separator
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")

        # Treeview
        cols = ("time", "camera", "persons", "summary")
        self._tree = ttk.Treeview(
            parent, columns=cols, show="headings", selectmode="browse",
        )
        self._tree.heading("time",    text="Date / Time")
        self._tree.heading("camera",  text="Camera")
        self._tree.heading("persons", text="Persons")
        self._tree.heading("summary", text="AI Summary")

        self._tree.column("time",    width=158, anchor="w",      stretch=False)
        self._tree.column("camera",  width=72,  anchor="center", stretch=False)
        self._tree.column("persons", width=62,  anchor="center", stretch=False)
        self._tree.column("summary", width=150, anchor="w",      stretch=True)

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Row colours: human-sized (bright) vs small / unknown (dim)
        self._tree.tag_configure("human", foreground=TEXT)
        self._tree.tag_configure("small", foreground=TEXT_DIM)

    def _build_right(self, parent: tk.Frame) -> None:
        # ── Snapshot area (expands to fill space) ────────────────────
        self._img_frame = tk.Frame(parent, bg="#080c10")
        self._img_frame.pack(fill="both", expand=True)

        self._img_label = tk.Label(
            self._img_frame,
            text="Select an event to view its snapshot",
            bg="#080c10", fg=TEXT_DIM,
            font=("Segoe UI", 12),
        )
        self._img_label.pack(expand=True)

        # ── Info strip at bottom (fixed height) ──────────────────────
        info = tk.Frame(parent, bg=BG2, height=190)
        info.pack(fill="x", side="bottom")
        info.pack_propagate(False)

        tk.Frame(info, bg=BORDER, height=1).pack(fill="x")

        inner = tk.Frame(info, bg=BG2)
        inner.pack(fill="both", expand=True, padx=16, pady=10)

        # ── Left: event details ──────────────────────────────────────
        details = tk.Frame(inner, bg=BG2)
        details.pack(side="left", fill="both", expand=True)

        self._info_ts       = tk.StringVar(value="—")
        self._info_camera   = tk.StringVar(value="—")
        self._info_persons  = tk.StringVar(value="—")
        self._info_analysis = tk.StringVar(value="—")

        rows = [
            ("Date / Time",  self._info_ts),
            ("Camera",       self._info_camera),
            ("Persons",      self._info_persons),
            ("AI Analysis",  self._info_analysis),
        ]
        for r, (label_text, var) in enumerate(rows):
            tk.Label(
                details, text=f"{label_text} :",
                bg=BG2, fg=TEXT_DIM,
                font=("Segoe UI", 9),
                anchor="w",
            ).grid(row=r, column=0, sticky="nw", padx=(0, 10), pady=3)
            tk.Label(
                details, textvariable=var,
                bg=BG2, fg=TEXT,
                font=("Segoe UI", 10),
                wraplength=460, justify="left", anchor="nw",
            ).grid(row=r, column=1, sticky="nw", pady=3)

        # ── Right: action buttons ────────────────────────────────────
        btns = tk.Frame(inner, bg=BG2)
        btns.pack(side="right", padx=(20, 0))

        ttk.Button(
            btns, text="▶   Play Video",
            style="Green.TButton",
            command=self._open_video,
        ).pack(fill="x", pady=(0, 6), ipadx=10, ipady=4)

        ttk.Button(
            btns, text="Open Captures Folder",
            style="TButton",
            command=self._open_folder,
        ).pack(fill="x", ipadx=6, ipady=3)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_events(self) -> list:
        if not os.path.exists(DB_FILE):
            return []
        try:
            with sqlite3.connect(DB_FILE) as conn:
                return conn.execute(
                    "SELECT id, timestamp, camera_index, snapshot_path, "
                    "video_path, analysis, box_count "
                    "FROM events ORDER BY id DESC"
                ).fetchall()
        except sqlite3.Error:
            return []

    def _refresh_events(self) -> None:
        rows = self._load_events()
        self._tree.delete(*self._tree.get_children())

        for row in rows:
            id_, ts, cam_idx, snap, vid, analysis, boxes = row
            boxes = boxes or 0

            # Shorten AI analysis to a one-line preview
            preview = (analysis or "").replace("\n", " ")
            if len(preview) > 42:
                preview = preview[:39] + "…"

            # Only show as "alert" (bright) if box_count > 0 (human-sized blob confirmed)
            tag = "human" if boxes > 0 else "small"

            self._tree.insert(
                "", "end",
                iid=str(id_),
                values=(ts, f"Cam {cam_idx}", boxes, preview),
                tags=(tag,),
            )

        count = len(rows)
        self._status_var.set(f"{count} event{'s' if count != 1 else ''} logged")

    def _auto_refresh(self) -> None:
        self._refresh_events()
        self.root.after(10_000, self._auto_refresh)

    # ------------------------------------------------------------------
    # Selection handler
    # ------------------------------------------------------------------

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        event_id = int(sel[0])

        try:
            with sqlite3.connect(DB_FILE) as conn:
                row = conn.execute(
                    "SELECT timestamp, camera_index, snapshot_path, "
                    "video_path, analysis, box_count "
                    "FROM events WHERE id=?",
                    (event_id,),
                ).fetchone()
        except sqlite3.Error:
            return

        if not row:
            return

        ts, cam_idx, snap, vid, analysis, boxes = row
        self._current_video  = vid
        self._current_folder = os.path.dirname(snap) if snap else CAPTURES_DIR

        # Update info strip
        self._info_ts.set(ts)
        self._info_camera.set(f"Camera {cam_idx}  (index {cam_idx})")
        self._info_persons.set(
            f"{boxes or 0} human-sized blob{'s' if (boxes or 0) != 1 else ''} detected"
        )
        self._info_analysis.set(analysis or "No AI analysis available.")

        # Show snapshot
        self._show_snapshot(snap)

    def _show_snapshot(self, path: str | None) -> None:
        if not path or not os.path.exists(path):
            self._img_label.configure(
                image="",
                text="Snapshot file not found.",
                fg=ALERT_FG,
            )
            self._img_label.image = None
            return

        img = cv2.imread(path)
        if img is None:
            self._img_label.configure(
                image="",
                text="Cannot load image.",
                fg=ALERT_FG,
            )
            return

        # Scale to fit inside the image frame while preserving aspect ratio
        self._img_frame.update_idletasks()
        max_w = max(self._img_frame.winfo_width()  - 4, 400)
        max_h = max(self._img_frame.winfo_height() - 4, 300)
        h, w  = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Convert BGR → RGB, encode as PNG, wrap in PhotoImage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, png   = cv2.imencode(".png", img_rgb)
        b64      = base64.b64encode(png.tobytes()).decode()

        photo = tk.PhotoImage(data=b64)
        self._img_label.configure(image=photo, text="")
        self._img_label.image = photo   # keep reference on widget
        self._photo           = photo   # keep reference on self

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_video(self) -> None:
        if not self._current_video:
            messagebox.showinfo(
                "No Video",
                "No video clip was saved for this event.\n\n"
                "Make sure RECORD_VIDEO = True in config.py.",
            )
            return
        if not os.path.exists(self._current_video):
            messagebox.showerror(
                "File Not Found",
                f"Video file not found:\n{self._current_video}",
            )
            return
        os.startfile(self._current_video)

    def _open_folder(self) -> None:
        folder = self._current_folder or CAPTURES_DIR
        if not os.path.exists(folder):
            folder = CAPTURES_DIR
        if not os.path.exists(folder):
            messagebox.showinfo(
                "No Folder",
                "The captures folder does not exist yet.\n"
                "Run security_cam.py first to start recording.",
            )
            return
        os.startfile(folder)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = AdminApp(root)
    root.mainloop()
