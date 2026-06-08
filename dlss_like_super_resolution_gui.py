#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import queue
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Text, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

CURRENT_VERSION = "v1.1.3"


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = application_dir()
ENGINE_SCRIPT = APP_DIR / "dlss_like_super_resolution.py"
APP_ICON = APP_DIR / "super-resolution.ico"
GUI_EXE = APP_DIR / "Image Super Resolution Tool.exe"
APP_DATA_NAME = "ImageSuperResolutionToolEN"
APP_DATA_ROOT = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or APP_DIR) / APP_DATA_NAME
RUNTIME_ROOT = APP_DATA_ROOT / "runtime"
LOCAL_PYTHON_DIR = RUNTIME_ROOT / "python-3.12.10"
LOCAL_PYTHON = LOCAL_PYTHON_DIR / "python.exe"
LEGACY_LOCAL_PYTHON_DIR = APP_DIR / ".python"
LEGACY_LOCAL_PYTHON = LEGACY_LOCAL_PYTHON_DIR / "python.exe"
VENV_DIR = RUNTIME_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_PYTHONW = VENV_DIR / "Scripts" / "pythonw.exe"
LEGACY_VENV_DIR = APP_DIR / ".venv"
LEGACY_VENV_PYTHON = LEGACY_VENV_DIR / "Scripts" / "python.exe"
LEGACY_VENV_PYTHONW = LEGACY_VENV_DIR / "Scripts" / "pythonw.exe"
REQUIREMENTS = APP_DIR / "requirements.txt"
SETTINGS_DIR = Path(os.environ.get("APPDATA", APP_DIR)) / APP_DATA_NAME
SETTINGS_PATH = SETTINGS_DIR / "settings.json"
LATEST_RELEASE_API = "https://api.github.com/repos/qwertasdfg77/image-super-resolution-tool-en/releases/latest"
RELEASES_URL = "https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest"
PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
PYTHON_INSTALLER_NAME = "python-3.12.10-amd64.exe"
PYTHON_INSTALLER_LOG = RUNTIME_ROOT / "python-installer.log"

MODEL_OPTIONS = {
    "ATD Official Super-Resolution Model": "atd",
    "HAT High-Quality Super-Resolution Model": "hat",
    "Real-ESRGAN General Photo Model": "photo",
    "Real-ESRGAN Anime and Illustration Model": "anime",
}
DEFAULT_MODEL_DISPLAY = "ATD Official Super-Resolution Model"

GPU_USAGE_OPTIONS = {
    "Auto": "auto",
    "Conservative": "conservative",
    "Balanced": "balanced",
    "Performance": "performance",
}

OUTPUT_FORMAT_OPTIONS = {
    "Auto": "auto",
    "PNG": "png",
    "JPEG": "jpeg",
    "WEBP": "webp",
}


def installed_venv_python() -> Path | None:
    for candidate in (VENV_PYTHON, LEGACY_VENV_PYTHON):
        if candidate.exists():
            return candidate
    return None


def runtime_python() -> str:
    installed = installed_venv_python()
    if installed:
        return str(installed)
    return base_python()


def is_windows_store_alias(path: str) -> bool:
    return "\\microsoft\\windowsapps\\" in path.lower()


def python_candidate_paths(include_local: bool = True) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | Path | None) -> None:
        if not value:
            return
        path = Path(value)
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        key = resolved.lower()
        if key in seen or is_windows_store_alias(resolved) or not path.exists():
            return
        seen.add(key)
        candidates.append(resolved)

    if include_local:
        add(LOCAL_PYTHON)
        add(LEGACY_LOCAL_PYTHON)

    if not getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe":
            add(exe.with_name("python.exe"))
        add(exe)

    add(shutil.which("python"))
    add(shutil.which("python3"))

    search_roots: list[Path] = []
    if os.environ.get("LOCALAPPDATA"):
        search_roots.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Python")
    if os.environ.get("ProgramFiles"):
        search_roots.append(Path(os.environ["ProgramFiles"]))
    if os.environ.get("ProgramFiles(x86)"):
        search_roots.append(Path(os.environ["ProgramFiles(x86)"]))
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("Python3*/python.exe"), reverse=True):
            add(candidate)

    return candidates


def base_python() -> str:
    candidates = python_candidate_paths()
    if candidates:
        return candidates[0]
    raise RuntimeError("No usable Python runtime was found. Click Install/Check Runtime first.")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def choice_setting(settings: dict, key: str, default: str, choices: dict[str, str]) -> str:
    value = settings.get(key, default)
    return value if value in choices else default


def int_setting(settings: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(settings.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def version_tuple(tag: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", tag)
    return tuple(int(part) for part in numbers[:3]) or (0,)


def is_newer_version(latest: str, current: str) -> bool:
    return version_tuple(latest) > version_tuple(current)


class SuperResolutionApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(f"Image Super Resolution Tool {CURRENT_VERSION}")
        if APP_ICON.exists():
            try:
                self.root.iconbitmap(str(APP_ICON))
            except Exception:
                pass
        self.root.geometry("1080x760")
        self.root.minsize(940, 680)

        settings = load_settings()
        self.settings_ready = False
        self.input_path = StringVar(value="")
        self.output_path = StringVar(value="")
        self.model_display = StringVar(value=choice_setting(settings, "model_display", DEFAULT_MODEL_DISPLAY, MODEL_OPTIONS))
        self.scale = StringVar(value=str(settings.get("scale", "4")) if str(settings.get("scale", "4")) in {"2", "3", "4"} else "4")
        self.gpu_usage_display = StringVar(value=choice_setting(settings, "gpu_usage_display", "Auto", GPU_USAGE_OPTIONS))
        self.output_format_display = StringVar(value=choice_setting(settings, "output_format_display", "Auto", OUTPUT_FORMAT_OPTIONS))
        self.jpeg_quality = IntVar(value=int_setting(settings, "jpeg_quality", 95, 1, 100))
        self.auto_sharpness = BooleanVar(value=bool(settings.get("auto_sharpness", True)))
        self.auto_denoise = BooleanVar(value=bool(settings.get("auto_denoise", True)))
        self.sharpness = DoubleVar(value=float(settings.get("sharpness", 0.65)))
        self.denoise = DoubleVar(value=float(settings.get("denoise", 0.06)))
        self.tta = BooleanVar(value=bool(settings.get("tta", False)))
        self.quiet = BooleanVar(value=bool(settings.get("quiet", False)))
        self.status = StringVar(value="Select an image or folder")
        self.progress_text = StringVar(value="Progress: 0% | Elapsed 00:00 | Remaining --:--")

        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.sharpness_slider: ttk.Scale | None = None
        self.denoise_slider: ttk.Scale | None = None
        self.run_started_at: float | None = None
        self.last_eta: float | None = None
        self.last_percent = 0.0
        self.last_output_path: Path | None = None
        self.installer_running = False

        self.build_ui()
        self.auto_sharpness.trace_add("write", lambda *_: self.update_slider_state())
        self.auto_denoise.trace_add("write", lambda *_: self.update_slider_state())
        for var in (
            self.model_display,
            self.scale,
            self.gpu_usage_display,
            self.output_format_display,
            self.jpeg_quality,
            self.auto_sharpness,
            self.auto_denoise,
            self.sharpness,
            self.denoise,
            self.tta,
            self.quiet,
        ):
            var.trace_add("write", lambda *_: self.save_current_settings())
        self.settings_ready = True
        if "input_path" in settings or "output_path" in settings:
            self.save_current_settings()
        self.update_slider_state()
        self.root.after(120, self.drain_log_queue)
        self.root.after(1000, self.update_elapsed_clock)
        self.root.after(1800, lambda: self.check_updates(manual=False))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.check_environment(silent=True)

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=18)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        ttk.Label(main, text="Image Super Resolution Tool", font=("Segoe UI", 17, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(main, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=(4, 14))

        paths = ttk.LabelFrame(main, text="Images")
        paths.grid(row=2, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)

        ttk.Button(paths, text="Choose Image", command=self.choose_file).grid(row=0, column=0, padx=10, pady=(12, 6), sticky="ew")
        ttk.Entry(paths, textvariable=self.input_path).grid(row=0, column=1, padx=(0, 10), pady=(12, 6), sticky="ew")
        ttk.Button(paths, text="Choose Folder", command=self.choose_folder).grid(row=0, column=2, padx=(0, 10), pady=(12, 6), sticky="ew")

        ttk.Button(paths, text="Output Location", command=self.choose_output).grid(row=1, column=0, padx=10, pady=(0, 12), sticky="ew")
        ttk.Entry(paths, textvariable=self.output_path).grid(row=1, column=1, columnspan=2, padx=(0, 10), pady=(0, 12), sticky="ew")

        options = ttk.LabelFrame(main, text="Output")
        options.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        for col in range(5):
            options.columnconfigure(col, weight=1)

        ttk.Label(options, text="Model").grid(row=0, column=0, padx=10, pady=(12, 4), sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.model_display,
            values=tuple(MODEL_OPTIONS),
            state="readonly",
            width=22,
        ).grid(row=1, column=0, padx=10, pady=(0, 12), sticky="ew")

        ttk.Label(options, text="Scale").grid(row=0, column=1, padx=10, pady=(12, 4), sticky="w")
        ttk.Combobox(options, textvariable=self.scale, values=("2", "3", "4"), state="readonly", width=10).grid(
            row=1, column=1, padx=10, pady=(0, 12), sticky="ew"
        )

        ttk.Label(options, text="GPU Usage").grid(row=0, column=2, padx=10, pady=(12, 4), sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.gpu_usage_display,
            values=tuple(GPU_USAGE_OPTIONS),
            state="readonly",
            width=10,
        ).grid(row=1, column=2, padx=10, pady=(0, 12), sticky="ew")

        ttk.Checkbutton(options, text="Higher Quality", variable=self.tta).grid(row=1, column=3, padx=10, pady=(0, 12), sticky="w")
        ttk.Checkbutton(options, text="Brief Log", variable=self.quiet).grid(row=1, column=4, padx=10, pady=(0, 12), sticky="w")

        ttk.Label(options, text="Output Format").grid(row=2, column=0, padx=10, pady=(0, 4), sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.output_format_display,
            values=tuple(OUTPUT_FORMAT_OPTIONS),
            state="readonly",
            width=10,
        ).grid(row=3, column=0, padx=10, pady=(0, 12), sticky="ew")

        ttk.Label(options, text="JPEG/WEBP Quality").grid(row=2, column=1, padx=10, pady=(0, 4), sticky="w")
        ttk.Spinbox(options, from_=1, to=100, textvariable=self.jpeg_quality, width=10).grid(
            row=3, column=1, padx=10, pady=(0, 12), sticky="ew"
        )

        post = ttk.LabelFrame(main, text="Auto Sharpen and Denoise")
        post.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        post.columnconfigure(2, weight=1)
        post.columnconfigure(5, weight=1)

        ttk.Checkbutton(post, text="Auto Sharpen", variable=self.auto_sharpness).grid(row=0, column=0, padx=10, pady=12, sticky="w")
        ttk.Label(post, text="Manual Sharpen").grid(row=0, column=1, padx=(4, 10), pady=12, sticky="w")
        self.sharpness_slider = ttk.Scale(post, variable=self.sharpness, from_=0, to=1.5, orient="horizontal")
        self.sharpness_slider.grid(row=0, column=2, padx=(0, 18), pady=12, sticky="ew")

        ttk.Checkbutton(post, text="Auto Denoise", variable=self.auto_denoise).grid(row=0, column=3, padx=10, pady=12, sticky="w")
        ttk.Label(post, text="Manual Denoise").grid(row=0, column=4, padx=(4, 10), pady=12, sticky="w")
        self.denoise_slider = ttk.Scale(post, variable=self.denoise, from_=0, to=0.3, orient="horizontal")
        self.denoise_slider.grid(row=0, column=5, padx=(0, 10), pady=12, sticky="ew")

        log_frame = ttk.LabelFrame(main, text="Status")
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(14, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = Text(log_frame, height=9, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)
        self.log.configure(yscrollcommand=scroll.set)

        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, sticky="ew", pady=(14, 0))
        actions.columnconfigure(4, weight=1)

        ttk.Button(actions, text="Install/Check Runtime", command=self.open_installer).grid(row=0, column=0, padx=(0, 10), sticky="w")
        ttk.Button(actions, text="Detect GPU", command=lambda: self.check_environment(silent=False)).grid(row=0, column=1, padx=(0, 10), sticky="w")
        ttk.Button(actions, text="Check Updates", command=lambda: self.check_updates(manual=True)).grid(row=0, column=2, padx=(0, 10), sticky="w")
        self.open_output_button = ttk.Button(actions, text="Open Output Folder", command=self.open_output_folder, state="disabled")
        self.open_output_button.grid(row=0, column=3, padx=(0, 10), sticky="w")
        self.progress = ttk.Progressbar(actions, mode="determinate", maximum=100, value=0)
        self.progress.grid(row=0, column=4, padx=10, sticky="ew")
        ttk.Label(actions, textvariable=self.progress_text).grid(row=1, column=4, padx=10, pady=(4, 0), sticky="ew")
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_run, state="disabled")
        self.stop_button.grid(row=0, column=5, padx=(10, 0), sticky="e")
        self.start_button = ttk.Button(actions, text="Start Upscaling", command=self.start_run)
        self.start_button.grid(row=0, column=6, padx=(10, 0), sticky="e")

        self.append_log('Workflow: install/check the runtime first, choose images, then click "Start Upscaling".')
        self.append_log("The app automatically detects GPU model and VRAM, then selects tile size, precision, and memory usage.")

    def update_slider_state(self) -> None:
        if self.sharpness_slider:
            self.sharpness_slider.configure(state="disabled" if self.auto_sharpness.get() else "normal")
        if self.denoise_slider:
            self.denoise_slider.configure(state="disabled" if self.auto_denoise.get() else "normal")

    def save_current_settings(self) -> None:
        if not getattr(self, "settings_ready", False):
            return
        try:
            save_settings(
                {
                    "model_display": self.model_display.get(),
                    "scale": self.scale.get(),
                    "gpu_usage_display": self.gpu_usage_display.get(),
                    "output_format_display": self.output_format_display.get(),
                    "jpeg_quality": self.current_jpeg_quality(),
                    "auto_sharpness": self.auto_sharpness.get(),
                    "auto_denoise": self.auto_denoise.get(),
                    "sharpness": self.sharpness.get(),
                    "denoise": self.denoise.get(),
                    "tta": self.tta.get(),
                    "quiet": self.quiet.get(),
                }
            )
        except Exception:
            pass

    def current_jpeg_quality(self) -> int:
        try:
            value = int(self.jpeg_quality.get())
        except Exception:
            return 95
        return max(1, min(100, value))

    def on_close(self) -> None:
        self.save_current_settings()
        self.root.destroy()

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)
            self.output_path.set(str(Path(path).parent / "upscaled"))

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose Image Folder")
        if path:
            self.input_path.set(path)
            self.output_path.set(str(Path(path) / "upscaled"))

    def choose_output(self) -> None:
        path = filedialog.askdirectory(title="Choose Output Folder")
        if path:
            self.output_path.set(path)

    def open_output_folder(self) -> None:
        output = Path(self.output_path.get()) if self.output_path.get() else None
        if self.last_output_path and self.last_output_path.exists():
            folder = self.last_output_path.parent
        elif output and output.suffix:
            folder = output.parent
        elif output:
            folder = output
        else:
            messagebox.showwarning("Missing Output Location", "There is no output folder to open yet.")
            return
        if not folder.exists():
            messagebox.showwarning("Folder Not Found", f"Output folder not found: {folder}")
            return
        os.startfile(folder)

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def parse_engine_path(self, text: str, prefix: str) -> Path | None:
        if not text.startswith(prefix):
            return None
        value = text[len(prefix) :].split(" (", 1)[0].strip()
        return Path(value) if value else None

    def handle_engine_log(self, text: str) -> None:
        output_path = self.parse_engine_path(text, "Output: ")
        if output_path is not None:
            self.last_output_path = output_path
            self.open_output_button.configure(state="normal")
        self.append_log(text)

    def reset_progress(self) -> None:
        self.last_percent = 0.0
        self.last_eta = None
        self.progress.configure(value=0)
        self.progress_text.set("Progress: 0% | Elapsed 00:00 | Remaining --:--")

    def handle_progress_message(self, item: str) -> None:
        parts = item.split("|")
        if len(parts) != 5:
            return
        try:
            current = int(parts[1])
            total = max(1, int(parts[2]))
            elapsed = float(parts[3])
            eta = float(parts[4])
        except ValueError:
            return
        percent = max(0.0, min(100.0, current / total * 100))
        self.last_percent = percent
        self.last_eta = eta
        self.progress.configure(value=percent)
        self.progress_text.set(
            f"Progress: {percent:.0f}% | Elapsed {format_duration(elapsed)} | Remaining {format_duration(eta)}"
        )

    def update_elapsed_clock(self) -> None:
        if self.process is not None and self.run_started_at is not None:
            elapsed = time.monotonic() - self.run_started_at
            self.progress_text.set(
                f"Progress: {self.last_percent:.0f}% | Elapsed {format_duration(elapsed)} | Remaining {format_duration(self.last_eta)}"
            )
        self.root.after(1000, self.update_elapsed_clock)

    def drain_log_queue(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item.startswith("__DONE__:"):
                self.finish_run(int(item.split(":", 1)[1]))
            elif item.startswith("__PROGRESS__|"):
                self.handle_progress_message(item)
            else:
                self.handle_engine_log(item)
        self.root.after(120, self.drain_log_queue)

    def build_command(self) -> list[str]:
        sharpness = "auto" if self.auto_sharpness.get() else f"{self.sharpness.get():.2f}"
        denoise = "auto" if self.auto_denoise.get() else f"{self.denoise.get():.2f}"
        command = [
            runtime_python(),
            str(ENGINE_SCRIPT),
            self.input_path.get(),
            "-o",
            self.output_path.get(),
            "--model",
            MODEL_OPTIONS[self.model_display.get()],
            "--outscale",
            self.scale.get(),
            "--output-format",
            OUTPUT_FORMAT_OPTIONS[self.output_format_display.get()],
            "--gpu-usage",
            GPU_USAGE_OPTIONS[self.gpu_usage_display.get()],
            "--tile",
            "auto",
            "--tile-pad",
            "auto",
            "--precision",
            "auto",
            "--model-dir",
            str(APP_DIR / "models"),
            "--sharpness",
            sharpness,
            "--denoise",
            denoise,
            "--jpeg-quality",
            str(self.current_jpeg_quality()),
        ]
        if self.tta.get():
            command.append("--tta")
        if self.quiet.get():
            command.append("--quiet")
        return command

    def start_run(self) -> None:
        if self.process is not None:
            return
        if not self.input_path.get():
            messagebox.showwarning("Missing Image", "Choose an image or image folder first.")
            return
        if not self.output_path.get():
            messagebox.showwarning("Missing Output Location", "Choose an output folder first.")
            return
        if not ENGINE_SCRIPT.exists():
            messagebox.showerror("Missing File", f"Main script not found: {ENGINE_SCRIPT}")
            return
        try:
            command = self.build_command()
        except Exception as exc:
            messagebox.showwarning("Runtime Required", f"{exc}\n\nClick Install/Check Runtime first.")
            return

        self.save_current_settings()
        self.append_log("")
        self.append_log("Starting...")
        self.status.set("Upscaling with GPU, please wait")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_progress()
        self.run_started_at = time.monotonic()
        self.last_output_path = None

        threading.Thread(target=self.run_process, args=(command,), daemon=True).start()

    def run_process(self, command: list[str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.log_queue.put(line.rstrip())
            code = self.process.wait()
        except Exception as exc:
            self.log_queue.put(f"Startup failed: {exc}")
            code = 1
        self.log_queue.put(f"__DONE__:{code}")

    def finish_run(self, code: int) -> None:
        elapsed = time.monotonic() - self.run_started_at if self.run_started_at is not None else 0.0
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.process = None
        if code == 0:
            self.progress.configure(value=100)
            self.progress_text.set(f"Progress: 100% | Elapsed {format_duration(elapsed)} | Remaining 00:00")
            self.status.set("Done")
            self.append_log("Processing complete.")
            messagebox.showinfo("Done", "Image upscaling is complete.")
        else:
            self.status.set("Not Completed")
            self.progress_text.set(
                f"Progress: {self.last_percent:.0f}% | Elapsed {format_duration(elapsed)} | Remaining --:--"
            )
            self.append_log("Processing did not complete. Check the status log above.")
            messagebox.showerror("Not Completed", "Processing did not complete. Check the status log.")
        self.run_started_at = None

    def stop_run(self) -> None:
        if self.process is None:
            return
        self.append_log("Stopping...")
        self.process.terminate()

    def check_updates(self, manual: bool) -> None:
        def worker() -> None:
            try:
                request = urllib.request.Request(
                    LATEST_RELEASE_API,
                    headers={"User-Agent": f"ImageSuperResolutionTool/{CURRENT_VERSION}"},
                )
                with urllib.request.urlopen(request, timeout=12) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latest_tag = payload.get("tag_name", "")
                latest_url = payload.get("html_url", RELEASES_URL)
                if latest_tag and is_newer_version(latest_tag, CURRENT_VERSION):
                    def notify() -> None:
                        self.status.set(f"New version available: {latest_tag}")
                        self.append_log(f"New version available: {latest_tag}, current version: {CURRENT_VERSION}")
                        if messagebox.askyesno("New Version Available", f"Version {latest_tag} is available. Open the download page?"):
                            webbrowser.open(latest_url)

                    self.root.after(0, notify)
                elif manual:
                    self.root.after(0, lambda: messagebox.showinfo("Check Updates", f"You are already on the latest version: {CURRENT_VERSION}"))
            except Exception as exc:
                if manual:
                    message = str(exc)
                    self.root.after(0, lambda: messagebox.showwarning("Update Check Failed", message))

        threading.Thread(target=worker, daemon=True).start()

    def open_installer(self) -> None:
        if self.installer_running:
            messagebox.showinfo("Install/Check Runtime", "Runtime installation is already running. Wait for the current task to finish.")
            return

        window = Toplevel(self.root)
        window.title("Install/Check Runtime")
        window.geometry("720x460")
        window.minsize(620, 380)
        window.transient(self.root)

        status_text = StringVar(value="Preparing runtime check")
        percent_text = StringVar(value="0%")
        progress_value = DoubleVar(value=0)

        body = ttk.Frame(window, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, textvariable=status_text).pack(anchor="w")
        progress_row = ttk.Frame(body)
        progress_row.pack(fill="x", pady=(10, 8))
        ttk.Progressbar(progress_row, variable=progress_value, maximum=100, mode="determinate").pack(
            side="left", fill="x", expand=True
        )
        ttk.Label(progress_row, textvariable=percent_text, width=8, anchor="e").pack(side="left", padx=(10, 0))

        log = Text(body, height=14, wrap="word")
        log.pack(fill="both", expand=True)
        log.configure(state="disabled")

        button_row = ttk.Frame(body)
        button_row.pack(fill="x", pady=(12, 0))
        close_button = ttk.Button(button_row, text="Close", state="disabled", command=window.destroy)
        close_button.pack(side="right")

        def append_log(text: str) -> None:
            log.configure(state="normal")
            log.insert("end", text.rstrip() + "\n")
            log.see("end")
            log.configure(state="disabled")

        def set_progress(percent: float, status: str | None = None) -> None:
            percent = max(0, min(100, percent))
            progress_value.set(percent)
            percent_text.set(f"{percent:.0f}%")
            if status:
                status_text.set(status)

        def finish(success: bool, message: str) -> None:
            self.installer_running = False
            close_button.configure(state="normal")
            set_progress(100 if success else progress_value.get(), message)
            append_log(message)
            if success:
                self.check_environment(silent=True)

        def format_command(command: list[str]) -> str:
            if os.name == "nt":
                return subprocess.list2cmdline(command)
            return " ".join(command)

        def run_command(command: list[str], start: float, end: float, status: str) -> None:
            self.root.after(0, lambda: set_progress(start, status))
            self.root.after(0, lambda: append_log("> " + format_command(command)))

            stop_animation = threading.Event()

            def animate() -> None:
                current = start
                step = max(0.5, (end - start) / 60)
                while not stop_animation.wait(1.0):
                    current = min(end - 1, current + step)
                    self.root.after(0, lambda value=current: set_progress(value))

            threading.Thread(target=animate, daemon=True).start()
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.strip()
                if clean:
                    self.root.after(0, lambda text=clean: append_log(text))
            code = process.wait()
            stop_animation.set()
            if code != 0:
                raise RuntimeError(f"Command failed with exit code {code}: {format_command(command)}")
            self.root.after(0, lambda: set_progress(end, status))

        def download_python_installer(start: float, end: float, force_download: bool = False) -> Path:
            target = Path(tempfile.gettempdir()) / PYTHON_INSTALLER_NAME
            if force_download and target.exists():
                target.unlink(missing_ok=True)
            if target.exists() and target.stat().st_size > 20_000_000:
                self.root.after(0, lambda: append_log(f"Reusing downloaded Python installer: {target}"))
                self.root.after(0, lambda: set_progress(end, "Python installer is ready"))
                return target
            if target.exists():
                target.unlink(missing_ok=True)

            self.root.after(0, lambda: set_progress(start, "Downloading local Python runtime"))
            self.root.after(0, lambda: append_log(f"Download: {PYTHON_INSTALLER_URL}"))
            request = urllib.request.Request(
                PYTHON_INSTALLER_URL,
                headers={"User-Agent": f"ImageSuperResolutionTool/{CURRENT_VERSION}"},
            )
            downloaded = 0
            with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
                total = int(response.headers.get("Content-Length", "0") or 0)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = start + (end - start) * min(1.0, downloaded / total)
                        self.root.after(0, lambda value=percent: set_progress(value))
            if target.stat().st_size < 20_000_000:
                raise RuntimeError("Python installer download is incomplete. Check the network and try again.")
            self.root.after(0, lambda: set_progress(end, "Python installer downloaded"))
            return target

        def checked_python(executable: str | Path) -> str:
            executable = str(executable)
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                [
                    executable,
                    "-c",
                    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stdout.strip() or "Python version check failed.")
            try:
                major, minor = (int(part) for part in result.stdout.strip().split(".", 1))
            except Exception as exc:
                raise RuntimeError(f"Could not detect Python version: {result.stdout.strip()}") from exc
            if major != 3 or minor < 10 or minor > 12:
                raise RuntimeError("Use Python 3.10, 3.11, or 3.12. Python 3.12 is recommended.")
            return executable

        def checked_python_candidates(include_local: bool = True) -> str | None:
            for candidate in python_candidate_paths(include_local=include_local):
                try:
                    return checked_python(candidate)
                except Exception as exc:
                    self.root.after(0, lambda path=candidate, error=exc: append_log(f"Ignoring unusable Python: {path} ({error})"))
            return None

        def install_local_python() -> str:
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            last_error: Exception | None = None
            for attempt in range(2):
                installer = download_python_installer(5, 18, force_download=attempt > 0)
                if LOCAL_PYTHON_DIR.exists() and not LOCAL_PYTHON.exists():
                    shutil.rmtree(LOCAL_PYTHON_DIR, ignore_errors=True)
                LOCAL_PYTHON_DIR.parent.mkdir(parents=True, exist_ok=True)
                self.root.after(0, lambda: append_log(f"Python install directory: {LOCAL_PYTHON_DIR}"))
                self.root.after(0, lambda: append_log(f"Python installer log: {PYTHON_INSTALLER_LOG}"))
                try:
                    run_command(
                        [
                            str(installer),
                            "/quiet",
                            "/log",
                            str(PYTHON_INSTALLER_LOG),
                            "InstallAllUsers=0",
                            "PrependPath=0",
                            "Include_launcher=0",
                            "InstallLauncherAllUsers=0",
                            "Include_pip=1",
                            "Include_tcltk=1",
                            "Include_test=0",
                            "Shortcuts=0",
                            f"TargetDir={LOCAL_PYTHON_DIR}",
                        ],
                        18,
                        34,
                        "Installing local Python runtime",
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        self.root.after(0, lambda: append_log("Python installer failed. Downloading a fresh installer and retrying."))
                        continue
                    raise RuntimeError(f"Python installer failed: {exc}; installer log: {PYTHON_INSTALLER_LOG}") from exc

                if LOCAL_PYTHON.exists():
                    return checked_python(LOCAL_PYTHON)

                fallback = checked_python_candidates(include_local=False)
                if fallback:
                    self.root.after(
                        0,
                        lambda path=fallback: append_log(f"The installer did not write to the target directory. Using detected Python instead: {path}"),
                    )
                    return fallback

                if attempt == 0:
                    self.root.after(0, lambda: append_log("Installed Python was not found. Downloading a fresh installer and retrying."))
                    continue

            detail = f"; last error: {last_error}" if last_error else ""
            raise RuntimeError(f"Local Python installation failed; missing: {LOCAL_PYTHON}; installer log: {PYTHON_INSTALLER_LOG}{detail}")

        def ensure_python() -> str:
            existing = checked_python_candidates()
            if existing:
                return existing
            return install_local_python()

        def worker() -> None:
            self.installer_running = True
            try:
                self.root.after(0, lambda: set_progress(3, "Checking Python runtime"))
                source_python = ensure_python()
                self.root.after(0, lambda: append_log(f"Python: {source_python}"))

                active_venv_python = installed_venv_python()
                if not active_venv_python:
                    if VENV_DIR.exists() and not VENV_PYTHON.exists():
                        shutil.rmtree(VENV_DIR, ignore_errors=True)
                    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
                    run_command([source_python, "-m", "venv", str(VENV_DIR)], 34, 46, "Creating local runtime environment")
                    active_venv_python = VENV_PYTHON
                    if not active_venv_python.exists():
                        raise RuntimeError(f"Local runtime environment creation failed; missing: {active_venv_python}")
                else:
                    self.root.after(0, lambda: append_log("Local runtime environment already exists."))
                    self.root.after(0, lambda: set_progress(46, "Local runtime environment already exists"))

                run_command([str(active_venv_python), "-m", "pip", "install", "--upgrade", "pip"], 46, 54, "Upgrading pip")
                run_command(
                    [
                        str(active_venv_python),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu124",
                    ],
                    54,
                    80,
                    "Installing CUDA PyTorch",
                )
                run_command([str(active_venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], 80, 92, "Installing other dependencies")
                run_command(
                    [
                        str(active_venv_python),
                        "-c",
                        (
                            "import torch; "
                            "print('CUDA available:', torch.cuda.is_available()); "
                            "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not found')"
                        ),
                    ],
                    92,
                    98,
                    "Checking CUDA and GPU",
                )
                self.root.after(0, lambda: finish(True, "Runtime installed and checked successfully."))
            except Exception as exc:
                self.root.after(0, lambda error=exc: finish(False, f"Runtime installation failed: {error}"))

        self.installer_running = True
        window.protocol("WM_DELETE_WINDOW", lambda: None if self.installer_running else window.destroy())
        threading.Thread(target=worker, daemon=True).start()

    def check_environment(self, silent: bool) -> None:
        def worker() -> None:
            try:
                command = [
                    runtime_python(),
                    "-c",
                    (
                        "import sys\n"
                        "print('Python:', sys.executable)\n"
                        "try:\n"
                        "    import torch\n"
                        "    print('PyTorch:', torch.__version__)\n"
                        "    print('CUDA:', torch.cuda.is_available())\n"
                        "    if torch.cuda.is_available():\n"
                        "        props = torch.cuda.get_device_properties(0)\n"
                        "        free, total = torch.cuda.mem_get_info(0)\n"
                        "        print('GPU:', torch.cuda.get_device_name(0))\n"
                        "        print('VRAM:', round(total / 1024**3, 1), 'GB')\n"
                        "        print('Free VRAM:', round(free / 1024**3, 1), 'GB')\n"
                        "        print('SM:', getattr(props, 'multi_processor_count', 0))\n"
                        "except Exception as e:\n"
                        "    print('PyTorch/CUDA check failed:', e)\n"
                    ),
                ]
            except Exception as exc:
                if not silent:
                    self.log_queue.put("")
                    self.log_queue.put(str(exc))
                    self.root.after(
                        0,
                        lambda: messagebox.showwarning("Runtime Required", "Click Install/Check Runtime first."),
                    )
                else:
                    self.root.after(0, lambda: self.status.set("Click Install/Check Runtime first"))
                return
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            output = result.stdout.strip()
            if not silent:
                self.log_queue.put("")
                for line in output.splitlines():
                    self.log_queue.put(line)
            if "CUDA: True" in output:
                gpu_line = next((line for line in output.splitlines() if line.startswith("GPU:")), "GPU: detected")
                self.root.after(0, lambda: self.status.set(gpu_line))
            elif not silent:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning("Runtime Required", "No usable CUDA runtime was detected. Click Install/Check Runtime first."),
                )

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    SuperResolutionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
