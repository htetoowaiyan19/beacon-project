"""Lightweight Tkinter control panel. Training runs in a separate Python process."""
from __future__ import annotations
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from scripts.utils.training_status import read_status
    from scripts.utils.training_checkpoints import latest_checkpoint
except ModuleNotFoundError:
    from utils.training_status import read_status
    from utils.training_checkpoints import latest_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(epochs="2", batch_size="1", grad_accum="8", learning_rate="0.0001",
                lora_r="32", lora_alpha="64", max_seq_length="2048", max_steps="-1", save_steps="25")


def active_files(root=ROOT):
    active = json.loads((root / "datasets/active.json").read_text(encoding="utf-8"))
    release = root / "datasets" / active["release_directory"]
    return {"train_file": str(release / "train/train_combined.jsonl"),
            "val_file": str(release / "validation/validation_combined.jsonl"),
            "test_file": str(release / "test/test_combined.jsonl")}


def validate_config(config):
    for key in ("model_path", "train_file", "val_file", "test_file"):
        path = Path(config[key])
        if not (path.is_dir() if key == "model_path" else path.is_file()):
            raise ValueError(f"Missing {key}: {path}")
    for key in DEFAULTS:
        try:
            value = float(config[key]) if key == "learning_rate" else int(config[key])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {key}") from None
        if not math.isfinite(value) or (value <= 0 and not (key == "max_steps" and value == -1)):
            raise ValueError(f"{key} must be positive (max_steps can be -1)")
    if int(config["max_seq_length"]) > 4096:
        raise ValueError("Use a sequence length of 4096 or less for this training UI.")


class TrainingRun:
    """Subprocess lifecycle without importing torch into the UI."""
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.process = None
        self.directory = None
        self.events = queue.Queue()
        self.started = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, config, command=None, resume_output=None):
        if self.running:
            raise RuntimeError("A training run is already active.")
        validate_config(config)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.directory = self.root / "outputs/training_runs" / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        output = Path(resume_output) if resume_output else self.directory / "adapter"
        settings = dict(config, output_dir=str(output), status_file=str(self.directory / "status.json"),
                        stop_file=str(self.directory / "stop.request"))
        if resume_output:
            settings['resume_from_checkpoint'] = str(latest_checkpoint(output))
        python = self.root / ".venv/Scripts/python.exe"
        if not python.exists():
            python = self.root / ".venv/bin/python"
        if not python.exists():
            python = Path(sys.executable)
        arguments = [str(python), "-u", str(self.root / "scripts/train_lora.py")]
        for key, value in settings.items():
            arguments.extend([f"--{key}", str(value)])
        (self.directory / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.started = time.time()
        try:
            self.process = subprocess.Popen(command or arguments, cwd=self.root, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            (self.directory / "exit.json").write_text(json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8")
            raise
        (self.directory / "process.json").write_text(json.dumps({"pid": self.process.pid, "started_at": self.started}), encoding="utf-8")
        threading.Thread(target=self._collect, args=(self.process, self.directory), daemon=True).start()

    def _collect(self, process, directory):
        with (directory / "train.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line); log.flush()
                self.events.put(line)
        code = process.wait()
        final = read_status(directory / "status.json")
        status = final.get("status") if code == 0 else "failed"
        if status not in {"completed", "stopped", "failed"}:
            status = "finished" if code == 0 else "failed"
        (directory / "exit.json").write_text(json.dumps({"status": status, "returncode": code,
            "ended_at": time.time()}), encoding="utf-8")

    def stop(self):
        if self.running:
            (self.directory / "stop.request").write_text("stop", encoding="utf-8")

    def resume(self, directory):
        if self.running: raise RuntimeError('A training run is already active.')
        settings = json.loads((Path(directory) / 'config.json').read_text(encoding='utf-8'))
        config = {k:settings.get(k,v) for k,v in DEFAULTS.items()}
        config.update({k:settings[k] for k in ('model_path','train_file','val_file','test_file')})
        latest_checkpoint(settings['output_dir'])
        self.start(config, resume_output=settings['output_dir'])

    def stats(self, directory=None):
        path = Path(directory) if directory else self.directory
        if not path:
            return {}
        stats = read_status(path / "status.json")
        stats.update(read_status(path / "exit.json"))
        if not stats:
            stats = {"status": "starting"}
        if path == self.directory and self.started and self.running:
            stats["elapsed_seconds"] = time.time() - self.started
            if (path / "stop.request").exists():
                stats["status"] = "stop requested; waiting for optimizer step / save"
        return stats


def duration(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class TrainingUI:
    def __init__(self, window):
        self.window = window
        self.run = TrainingRun()
        self.view_directory = None
        self.variables = {}
        window.title("BEACON · Training control")
        window.geometry("1000x820")
        window.minsize(850, 720)
        style = ttk.Style(window)
        style.theme_use("clam")
        outer = ttk.Frame(window, padding=18); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="BEACON training", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Burmese companion · Qwen LoRA · IT seminar dataset").pack(anchor="w", pady=(0, 12))
        settings = ttk.LabelFrame(outer, text="Training settings", padding=12); settings.pack(fill="x")
        paths = dict(model_path=str(ROOT / "models/qwen3-4b"), **active_files())
        for row, (key, value) in enumerate(paths.items()):
            ttk.Label(settings, text=key.replace('_', ' ').title()).grid(row=row, column=0, sticky="w", padx=(0, 10))
            variable = tk.StringVar(value=value); self.variables[key] = variable
            ttk.Entry(settings, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
            ttk.Button(settings, text="Browse", command=lambda k=key: self.browse(k)).grid(row=row, column=4, padx=8)
        settings.columnconfigure(1, weight=1); settings.columnconfigure(3, weight=1)
        for i, (key, value) in enumerate(DEFAULTS.items()):
            row, col = 4 + i // 2, (i % 2) * 2
            ttk.Label(settings, text=key.replace('_', ' ').title()).grid(row=row, column=col, sticky="w", padx=(0, 10))
            variable = tk.StringVar(value=value); self.variables[key] = variable
            ttk.Entry(settings, textvariable=variable, width=16).grid(row=row, column=col+1, sticky="ew", pady=3)
        actions = ttk.Frame(outer); actions.pack(fill="x", pady=12)
        self.start_button = ttk.Button(actions, text="Start training", command=self.start); self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop safely", command=self.stop, state="disabled"); self.stop_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Open run folder", command=self.open_folder).pack(side="left")
        ttk.Button(actions, text="View previous run", command=self.view_previous).pack(side="left", padx=8)
        self.resume_button = ttk.Button(actions, text="Resume run", command=self.resume)
        self.resume_button.pack(side="left")
        self.state = tk.StringVar(value="Ready · review the dataset before starting.")
        ttk.Label(outer, textvariable=self.state, wraplength=950).pack(anchor="w")
        self.progress = ttk.Progressbar(outer, maximum=100); self.progress.pack(fill="x", pady=8)
        self.stats_text = tk.StringVar(value="Stats appear when training begins.")
        ttk.Label(outer, textvariable=self.stats_text, wraplength=950, justify="left").pack(anchor="w")
        self.chart = tk.Canvas(outer, height=95, bg="#17212e", highlightthickness=0); self.chart.pack(fill="x", pady=8)
        self.losses = []
        self.last_loss_step = None
        logs = ttk.Frame(outer); logs.pack(fill="both", expand=True)
        self.log = tk.Text(logs, height=8, wrap="word", bg="#111923", fg="#dce7f3", font=("Consolas", 10), state="disabled")
        scrollbar = ttk.Scrollbar(logs, command=self.log.yview); self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(500, self.refresh)

    def browse(self, key):
        selected = filedialog.askdirectory() if key == "model_path" else filedialog.askopenfilename(filetypes=[("JSONL dataset", "*.jsonl"), ("All files", "*.*")])
        if selected:
            self.variables[key].set(selected)

    def start(self):
        try:
            self.run.start({key: variable.get().strip() for key, variable in self.variables.items()})
        except Exception as exc:
            messagebox.showerror("Cannot start training", str(exc)); return
        self.view_directory = None; self.losses = []; self.last_loss_step = None
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

    def stop(self):
        self.run.stop()

    def resume(self):
        path = filedialog.askdirectory(initialdir=str(ROOT / 'outputs/training_runs'), title='Select the previous run folder')
        if not path: return
        try: self.run.resume(path)
        except Exception as exc:
            messagebox.showerror('Cannot resume', str(exc)); return
        self.view_directory = None; self.losses = []; self.last_loss_step = None
        self.log.configure(state='normal'); self.log.delete('1.0','end'); self.log.configure(state='disabled')

    def open_folder(self):
        directory = self.view_directory or self.run.directory or ROOT / "outputs"
        if os.name == "nt":
            os.startfile(str(directory))
        else:
            import webbrowser
            webbrowser.open(Path(directory).as_uri())

    def view_previous(self):
        if self.run.running:
            messagebox.showinfo("Training active", "Previous runs can be viewed after this run finishes."); return
        path = filedialog.askdirectory(initialdir=str(ROOT / "outputs/training_runs"))
        if not path:
            return
        self.view_directory = Path(path)
        self.losses = []; self.last_loss_step = None
        try:
            content = (self.view_directory / "train.log").read_text(encoding="utf-8")[-100000:]
        except OSError:
            content = "No log found in this folder."
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.insert("end", content); self.log.configure(state="disabled")

    def refresh(self):
        self.start_button.configure(state="disabled" if self.run.running else "normal")
        self.resume_button.configure(state="disabled" if self.run.running else "normal")
        self.stop_button.configure(state="normal" if self.run.running else "disabled")
        lines = []
        for _ in range(400):
            try: lines.append(self.run.events.get_nowait())
            except queue.Empty: break
        if lines and not self.view_directory:
            self.log.configure(state="normal"); self.log.insert("end", ''.join(lines))
            if int(self.log.index('end-1c').split('.')[0]) > 2000:
                self.log.delete('1.0', '500.0')
            self.log.see("end"); self.log.configure(state="disabled")
        stats = self.run.stats(self.view_directory)
        if stats:
            step, maximum = stats.get("step", 0), stats.get("max_steps", 0)
            self.state.set(f"{stats.get('status', 'unknown')} · {self.view_directory or self.run.directory}")
            self.progress['value'] = 100 * step / maximum if maximum else 0
            self.stats_text.set(f"Step {step}/{maximum or '—'} · Epoch {stats.get('epoch', '—')} · Elapsed {duration(stats.get('elapsed_seconds'))} · ETA {duration(stats.get('eta_seconds'))}\n"
                f"Loss {stats.get('loss', '—')} · Validation loss {stats.get('eval_loss', '—')} · LR {stats.get('learning_rate', '—')}\n"
                f"{stats.get('gpu', 'Device pending')} · VRAM {stats.get('vram_allocated_gb', '—')} GB allocated / {stats.get('vram_reserved_gb', '—')} GB reserved · RAM {stats.get('ram_gb', '—')} GB\n"
                f"Best checkpoint: {stats.get('best_checkpoint') or '—'}")
            loss = stats.get('loss')
            loss_step = stats.get('loss_step', step)
            if loss is not None and self.last_loss_step != loss_step:
                self.losses.append(float(loss)); self.last_loss_step = loss_step
            self.draw_chart()
        self.window.after(500, self.refresh)

    def draw_chart(self):
        self.chart.delete('all'); self.chart.create_text(12, 12, anchor='nw', fill='#83ddc2', text='Training loss (logged updates)')
        if len(self.losses) < 2: return
        low, high = min(self.losses), max(self.losses)
        width = max(100, self.chart.winfo_width() - 24)
        points = []
        for i, value in enumerate(self.losses):
            points.extend([12 + width * i / (len(self.losses)-1), 80 - 45 * (value-low) / (high-low or 1)])
        self.chart.create_line(*points, fill='#83ddc2', width=2)

    def close(self):
        if self.run.running:
            if messagebox.askyesno("Training active", "Request a safe stop and keep the UI open until it saves?"):
                self.run.stop()
            return
        self.window.destroy()


def main():
    window = tk.Tk(); TrainingUI(window); window.mainloop()


if __name__ == "__main__":
    main()
