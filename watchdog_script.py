import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_server()

    def start_server(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.process = subprocess.Popen(
            ["mesop", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.print_output(self.process)

    def print_output(self, process):
        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.decode().strip())
        rc = process.poll()
        return rc

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"{event.src_path} modified; restarting server...")
            self.start_server()

if __name__ == "__main__":
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()
    print("Watching for file changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
