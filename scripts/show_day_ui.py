"""Members' desktop dashboard; visitors use the independent web frontend."""
from __future__ import annotations
import argparse
from datetime import datetime
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser

try:
    from scripts.utils.show_day_control import ShowDayControl, gpu_telemetry
except ModuleNotFoundError:
    from utils.show_day_control import ShowDayControl, gpu_telemetry


class ShowDayUI:
    def __init__(self, window, *, preview=False, backend_port=8000, frontend_port=8080):
        self.window = window
        self.control = ShowDayControl()
        self.updates = queue.Queue(maxsize=100)
        self.stop_polling = threading.Event()
        self.action_lock = threading.Lock()
        self.busy = self.closing = False
        self.sequence = 0; self.chats = {}; self.last_transcript = ''
        self.backend_port = tk.StringVar(value=str(backend_port))
        self.frontend_port = tk.StringVar(value=str(frontend_port))
        self.preview = tk.BooleanVar(value=preview)
        self.follow = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Start Backend, then Frontend. Move the visitor browser to the front monitor.')
        self.server_labels = {}; self.stat_labels = {}; self.buttons = []; self.logs = {}
        window.title('BEACON · Show day — Members control')
        width = max(1040, min(1240, window.winfo_screenwidth() - 80))
        height = max(720, min(860, window.winfo_screenheight() - 120))
        window.geometry(f'{width}x{height}'); window.minsize(1040, 720)
        window.configure(background='#f1f5ef')
        style = ttk.Style(window); style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background='#f1f5ef', foreground='#20352d')
        style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'))
        style.configure('Card.TLabelframe', background='#ffffff', borderwidth=1)
        style.configure('Card.TLabelframe.Label', background='#f1f5ef', foreground='#526b5c')
        style.configure('TButton', padding=(12, 8))
        style.configure('Accent.TButton', background='#255c48', foreground='white')
        style.map('Accent.TButton', background=[('active', '#36715a')])
        outer = ttk.Frame(window, padding=18); outer.pack(fill='both', expand=True)
        heading = ttk.Frame(outer); heading.pack(fill='x')
        ttk.Label(heading, text='BEACON / Show day', style='Title.TLabel').pack(side='left')
        self._button(heading, 'Open visitor page ↗', self.open_visitor, side='right', accent=True)
        ttk.Label(outer, text='MEMBERS MONITOR  ·  Server controls, model stats and live visitor conversations').pack(anchor='w', pady=(3, 14))
        servers = ttk.Frame(outer); servers.pack(fill='x')
        for column, name, variable, start, stop in (
            (0, 'Backend', self.backend_port, self.start_backend, lambda: self.action('Stopping backend', self.control.stop_backend)),
            (1, 'Frontend', self.frontend_port, self.start_frontend, lambda: self.action('Stopping frontend', self.control.frontend.stop)),
        ):
            servers.columnconfigure(column, weight=1)
            card = ttk.LabelFrame(servers, text=name + ' server', padding=12)
            card.grid(row=0, column=column, sticky='nsew', padx=(0, 6) if column == 0 else (6, 0))
            ttk.Label(card, text='Port').pack(side='left')
            ttk.Entry(card, textvariable=variable, width=7).pack(side='left', padx=(6, 12))
            self._button(card, 'Start ' + name, start, side='left', accent=True)
            self._button(card, 'Stop', stop, side='left')
            self.server_labels[name] = tk.StringVar(value='Stopped')
            ttk.Label(card, textvariable=self.server_labels[name]).pack(side='right', padx=8)
        ttk.Checkbutton(outer, text='Preview mode — synthetic responses, no GPU/model (restart backend to change)', variable=self.preview).pack(anchor='w', pady=(10, 14))
        stats = ttk.Frame(outer); stats.pack(fill='x', pady=(0, 14))
        for column, key in enumerate(('Model', 'Adapter', 'VRAM', 'GPU', 'Requests', 'Last reply')):
            stats.columnconfigure(column, weight=1, uniform='stats')
            card = ttk.LabelFrame(stats, text=key, padding=9)
            card.grid(row=0, column=column, sticky='nsew', padx=(0, 6))
            variable = tk.StringVar(value='—'); self.stat_labels[key] = variable
            ttk.Label(card, textvariable=variable, wraplength=165, font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        workspace = ttk.Panedwindow(outer, orient='horizontal'); workspace.pack(fill='both', expand=True)
        left = ttk.Frame(workspace); right = ttk.Frame(workspace); workspace.add(left, weight=4); workspace.add(right, weight=6)
        logs = ttk.Notebook(left); logs.pack(fill='both', expand=True, padx=(0, 10))
        for name in ('Backend', 'Frontend'):
            frame = ttk.Frame(logs); logs.add(frame, text=name + ' logs')
            self.logs[name] = self._text(frame, font=('Consolas', 9), color='#132a20', foreground='#d3e3d1')
        tabs = ttk.Notebook(right); tabs.pack(fill='both', expand=True)
        chat_frame = ttk.Frame(tabs); tabs.add(chat_frame, text='Live conversations')
        toolbar = ttk.Frame(chat_frame); toolbar.pack(fill='x', pady=6)
        ttk.Checkbutton(toolbar, text='Follow latest visitor', variable=self.follow).pack(side='left')
        self.visitor_label = tk.StringVar(value='Visitors online: 0')
        ttk.Label(toolbar, textvariable=self.visitor_label).pack(side='right')
        self.tree = ttk.Treeview(chat_frame, columns=('visitor', 'state', 'question'), show='headings', height=5, selectmode='browse')
        for key, title, width in (('visitor', 'Visitor', 95), ('state', 'State', 80), ('question', 'Question', 310)):
            self.tree.heading(key, text=title); self.tree.column(key, width=width, minwidth=60)
        self.tree.pack(fill='x'); self.tree.bind('<<TreeviewSelect>>', self.select_chat)
        self.tree.bind('<ButtonRelease-1>', lambda event: self.follow.set(False) if self.tree.identify_row(event.y) else None)
        self.transcript = self._text(chat_frame, font=('Myanmar Text', 11), color='#ffffff', foreground='#20352d')
        activity = ttk.Frame(tabs); tabs.add(activity, text='Visitor activity')
        self.activity = self._text(activity, font=('Segoe UI', 10), color='#ffffff', foreground='#20352d')
        ttk.Label(outer, textvariable=self.status, wraplength=1160).pack(anchor='w', pady=(12, 3))
        ttk.Label(outer, text='Live chat feed stays in memory. Server logs are saved under outputs/show_day/. Visitors see a monitoring notice.', foreground='#6a7b6d').pack(anchor='w')
        window.protocol('WM_DELETE_WINDOW', self.close)
        threading.Thread(target=self._poll, daemon=True).start()
        window.after(150, self._drain)

    def _button(self, parent, text, command, side, accent=False):
        button = ttk.Button(parent, text=text, command=command, style='Accent.TButton' if accent else 'TButton')
        button.pack(side=side, padx=3); self.buttons.append(button)

    def _text(self, parent, font, color, foreground):
        frame = ttk.Frame(parent); frame.pack(fill='both', expand=True, pady=(6, 0))
        text = tk.Text(frame, wrap='word', background=color, foreground=foreground, font=font,
                       relief='flat', padx=12, pady=10, state='disabled')
        scrollbar = ttk.Scrollbar(frame, command=text.yview); text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y'); text.pack(fill='both', expand=True)
        return text

    def ports(self):
        try:
            return int(self.backend_port.get()), int(self.frontend_port.get())
        except ValueError:
            raise ValueError('Ports must be whole numbers.') from None

    def start_backend(self):
        if self.control.backend.running:
            return messagebox.showinfo('Backend', 'The backend is already running.')
        try:
            backend, frontend = self.ports(); preview = self.preview.get()
            if backend == frontend: raise ValueError('Choose different ports for the two servers.')
        except ValueError as exc:
            return messagebox.showerror('Server settings', str(exc))
        self.sequence = 0; self.chats = {}; self.last_transcript = ''
        self.tree.delete(*self.tree.get_children())
        for widget in (self.transcript, self.activity):
            widget.configure(state='normal'); widget.delete('1.0', 'end'); widget.configure(state='disabled')
        self.action('Starting backend', lambda: self.control.start_backend(backend, preview))

    def start_frontend(self):
        try:
            backend, frontend = self.ports()
        except ValueError as exc:
            return messagebox.showerror('Server settings', str(exc))
        self.action('Starting frontend', lambda: self.control.start_frontend(frontend, backend))

    def start_both(self):
        backend, frontend = self.ports(); preview = self.preview.get()
        def run():
            self.control.start_backend(backend, preview)
            self.control.start_frontend(frontend, backend)
        self.action('Starting both servers', run)

    def open_visitor(self):
        if not self.control.frontend.running:
            return messagebox.showinfo('Visitor page', 'Start the frontend server first.')
        webbrowser.open(f'http://127.0.0.1:{self.control.frontend_port}/')

    def action(self, label, operation):
        if self.busy or self.closing: return
        self.busy = True; self.status.set(label + '…')
        for button in self.buttons: button.configure(state='disabled')
        def run():
            with self.action_lock:
                try:
                    operation(); self.updates.put(('action', label + ' — done'))
                except Exception as exc:
                    self.updates.put(('error', str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        tick = 0; driver = {}
        while not self.stop_polling.wait(1):
            tick += 1
            if tick % 5 == 0: driver = gpu_telemetry()
            if not self.control.backend.running: continue
            try:
                snapshot = self.control.request(f'/api/operator/snapshot?after={self.sequence}')
                self.updates.put(('snapshot', (snapshot, driver)), timeout=1)
            except Exception as exc:
                if not self.stop_polling.is_set():
                    try: self.updates.put(('poll_error', str(exc)), timeout=1)
                    except queue.Full: pass

    def _drain(self):
        for name, server in (('Backend', self.control.backend), ('Frontend', self.control.frontend)):
            status = f'Running · PID {server.process.pid}' if server.running else 'Stopped'
            if not server.running and server.process is not None and server.process.poll() not in (None, 0):
                status = f'Exited ({server.process.poll()}) · see logs'
            self.server_labels[name].set(status)
            lines = []
            for _ in range(150):
                try: lines.append(server.events.get_nowait())
                except queue.Empty: break
            if lines: self._append(self.logs[name], ''.join(lines), 1500)
        while True:
            try: kind, value = self.updates.get_nowait()
            except queue.Empty: break
            if kind == 'closed': self.window.destroy(); return
            if kind == 'snapshot': self._snapshot(*value)
            elif kind in ('action', 'error'):
                self.busy = False; self.status.set(value)
                if not self.closing:
                    for button in self.buttons: button.configure(state='normal')
                    if kind == 'error': messagebox.showerror('Server control', value)
            elif kind == 'poll_error' and not self.busy:
                self.status.set('Waiting for backend status: ' + value)
        if not self.control.backend.running:
            for key in self.stat_labels: self.stat_labels[key].set('Offline')
        self.window.after(150, self._drain)

    def _append(self, widget, text, max_lines):
        bottom = widget.yview()[1] > .97
        widget.configure(state='normal'); widget.insert('end', text)
        lines = int(widget.index('end-1c').split('.')[0])
        if lines > max_lines: widget.delete('1.0', f'{lines-max_lines}.0')
        widget.configure(state='disabled')
        if bottom: widget.see('end')

    def _snapshot(self, snapshot, driver):
        self.sequence = snapshot['sequence']
        gpu = snapshot['gpu']; counts = snapshot['counters']; metrics = snapshot['latest_metrics']
        mock = 'mock' in gpu.get('device', '').lower()
        self.stat_labels['Model'].set('PREVIEW · synthetic' if mock else gpu.get('device', 'Unknown'))
        adapter = ('Preview only' if mock else 'Trained · CPU offload' if gpu.get('has_lora') and gpu.get('cpu_offload')
                   else 'Trained · loaded' if gpu.get('has_lora')
                   else 'Base only · adapter missing' if gpu.get('is_loaded')
                   else 'Adapter missing' if gpu.get('adapter_available') is False
                   else 'Loading / waiting' if counts['active'] else 'Not loaded yet')
        self.stat_labels['Adapter'].set(adapter)
        self.stat_labels['VRAM'].set(f"Native RAM: {gpu.get('native_process_ram_gb') if gpu.get('native_process_ram_gb') is not None else 'unavailable'} GB\nSystem available: {gpu.get('system_available_ram_gb') if gpu.get('system_available_ram_gb') is not None else 'unavailable'} GB" if gpu.get('shared_memory') else f"{gpu.get('vram_allocated_gb',0)} GB allocated\n{gpu.get('vram_reserved_gb',0)} GB reserved")
        self.stat_labels['GPU'].set('Preview · no model' if mock else f"{driver['utilization']:.0f}% busy · {driver['temperature']:.0f}°C\n{driver['used_mb']/1024:.1f}/{driver['total_mb']/1024:.1f} GB" if driver else 'Driver stats unavailable')
        self.stat_labels['Requests'].set(f"{counts['active']} active · {counts['total']} total\n{counts['completed']} done · {counts['failed']} failed\n{counts['cancelled']} cancelled")
        self.stat_labels['Last reply'].set('Synthetic metrics' if mock else f"First text: {metrics.get('time_to_first_text_seconds','—')}s\n{metrics.get('prompt_tokens','—')} in / {metrics.get('total_tokens','—')} out\n{metrics.get('tokens_per_second','—')} tok/s · queue {metrics.get('queue_seconds','—')}s")
        online = sum(v['online'] for v in snapshot['visitors'])
        self.visitor_label.set(f'Visitors online: {online}')
        self.chats = {chat['id']: chat for chat in snapshot['chats']}
        existing = set(self.tree.get_children())
        for stale in existing - self.chats.keys(): self.tree.delete(stale)
        for chat in snapshot['chats']:
            values = (chat['client_id'][:10], chat['status'], chat['message'].replace('\n', ' ')[:90])
            if chat['id'] in existing: self.tree.item(chat['id'], values=values)
            else: self.tree.insert('', 'end', iid=chat['id'], values=values)
        if self.follow.get() and self.chats:
            latest = next(reversed(self.chats)); self.tree.selection_set(latest); self.tree.see(latest)
        self.select_chat()
        activity = []
        for event in snapshot['events']:
            if event['type'] == 'heartbeat': continue
            at = datetime.fromtimestamp(event['at']).strftime('%H:%M:%S')
            activity.append(f"{at}  {event['client_id'][:10]}  ·  {event['type'].replace('_',' ')}\n")
        if activity: self._append(self.activity, ''.join(activity), 750)
        if not self.busy and not self.closing:
            self.status.set(f"Backend connected · uptime {snapshot['uptime_seconds']}s · {gpu.get('adapter_name','preview')} · Visitors: http://127.0.0.1:{self.control.frontend_port or self.frontend_port.get()}/")

    def select_chat(self, event=None):
        selected = self.tree.selection()
        if not selected or selected[0] not in self.chats: return
        chat = self.chats[selected[0]]; lines = []
        for row in self.chats.values():
            if row['session_id'] != chat['session_id'] or row['client_id'] != chat['client_id']: continue
            lines.append(f"VISITOR {row['client_id'][:10]} · {datetime.fromtimestamp(row['started_at']).strftime('%H:%M:%S')}\n{row['message']}\n\nBEACON · {row['model']} · {row['status']}\n{row['answer'] or '[Waiting for reply…]'}")
            if row.get('error'): lines.append('Error: ' + row['error'])
            lines.append('\n' + '─' * 40 + '\n')
        text = '\n'.join(lines)
        if text == self.last_transcript: return
        bottom = self.transcript.yview()[1] > .97
        self.transcript.configure(state='normal'); self.transcript.delete('1.0', 'end'); self.transcript.insert('1.0', text); self.transcript.configure(state='disabled')
        self.last_transcript = text
        if bottom or self.follow.get(): self.transcript.see('end')

    def close(self):
        if self.closing: return
        self.closing = True; self.stop_polling.set(); self.status.set('Stopping owned servers and closing…')
        for button in self.buttons: button.configure(state='disabled')
        def stop():
            with self.action_lock:
                self.control.stop_all()
            self.updates.put(('closed', None))
        threading.Thread(target=stop, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description='BEACON members show-day dashboard')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--start', action='store_true', help='Start both servers on launch')
    parser.add_argument('--backend-port', type=int, default=8000)
    parser.add_argument('--frontend-port', type=int, default=8080)
    args = parser.parse_args()
    window = tk.Tk()
    ui = ShowDayUI(window, preview=args.preview, backend_port=args.backend_port, frontend_port=args.frontend_port)
    if args.start: window.after(200, ui.start_both)
    window.mainloop()


if __name__ == '__main__':
    main()
