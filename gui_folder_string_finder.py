#!/usr/bin/env python3
"""
gui_folder_string_finder.py

GUI recursive search:
- Choose folder
- Enter search string (literal; optional outer quotes stripped)
- Toggle case sensitivity
- Filename match for all files
- Content match for supported types: plain text, csv/tsv, pdf (optimized), docx (optional), xlsx (optional)
- Writes matches.txt in selected folder
"""

import os
import csv
import shutil
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Union

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging


# ----------------------------
# Extraction config
# ----------------------------

PLAIN_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".log",
    ".py", ".js", ".ts", ".html", ".css",
    ".json", ".xml", ".yaml", ".yml",
    ".ini", ".cfg", ".toml",
    ".sql",
}

CSV_EXTS = {".csv", ".tsv"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
XLSX_EXTS = {".xlsx"}

MAX_TEXT_CHARS = 5_000_000
PDFTOTEXT_TIMEOUT_SEC = 120
ROBUST_DECODING = True

# PDF optimization controls
PDF_PAGE_CHUNK_MAX_CHARS = 250_000   # cap per page text (safety)
PDF_MAX_PAGES = None                # set to an int (e.g., 500) to cap worst-case PDFs

# module logger
logger = logging.getLogger(__name__)


# ----------------------------
# File walking / reading
# ----------------------------

def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield Path(dirpath) / fn


def read_text_safely(path: Path, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    if not ROBUST_DECODING:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars), None
        except Exception as e:
            return None, str(e)

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, errors="strict") as f:
                data = f.read(max_chars + 1)
                if len(data) > max_chars:
                    data = data[:max_chars]
                return data, None
        except Exception as e:
            last_err = e
            continue
    return None, f"Decode failed ({type(last_err).__name__}: {last_err})" if last_err else "Decode failed"


def extract_csv_text(path: Path, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    raw, err = read_text_safely(path, max_chars)
    if err or raw is None:
        return None, err or "CSV read failed"

    try:
        lines_out: List[str] = []
        size = 0
        reader = csv.reader(raw.splitlines(), delimiter=delimiter)
        for row in reader:
            line = "\t".join(row)
            if line:
                lines_out.append(line)
                size += len(line) + 1
                if size >= max_chars:
                    break
        return "\n".join(lines_out)[:max_chars], None
    except Exception as e:
        # Fallback: search raw text
        return raw[:max_chars], f"CSV parse failed; searched raw text instead ({e})"


# ----------------------------
# Optimized PDF search
# ----------------------------

def _norm(s: str, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower()


def pdf_contains_needle_pdftotext(pdf_path: Path, needle: str, case_sensitive: bool) -> Tuple[Optional[bool], Optional[str]]:
    """
    Use pdftotext if available. We run it once, but we stream-search the output
    rather than building a giant string in Python.

    pdftotext writes to stdout; we read in chunks and check needle incrementally.
    """
    if shutil.which("pdftotext") is None:
        return None, "pdftotext not found"

    needle_n = _norm(needle, case_sensitive)

    try:
        # -layout keeps spacing somewhat stable, but doesn't matter for substring search.
        # Output to stdout: pdftotext <pdf> -
        proc = subprocess.Popen(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
        )

        assert proc.stdout is not None
        assert proc.stderr is not None

        # Streaming substring search needs overlap between chunks
        overlap = max(1024, len(needle) * 2)
        buf = ""

        while True:
            chunk = proc.stdout.read(64 * 1024)  # 64KB chunks
            if not chunk:
                break

            buf = (buf + chunk)
            hay = _norm(buf, case_sensitive)
            if needle_n in hay:
                try:
                    logger.debug("pdftotext: terminating process after match found")
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    logger.debug("pdftotext: terminate/wait failed; ignoring")
                    pass
                return True, None

            # Keep only tail to handle boundary-crossing matches
            if len(buf) > overlap:
                buf = buf[-overlap:]

        # Wait for completion and check errors
        try:
            rc = proc.wait(timeout=PDFTOTEXT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                logger.debug("pdftotext: timeout, terminating process")
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                logger.debug("pdftotext: terminate/wait after timeout failed; ignoring")
                pass
            return None, f"pdftotext timed out after {PDFTOTEXT_TIMEOUT_SEC}s"

        if rc != 0:
            err = proc.stderr.read()
            return None, f"pdftotext failed (code {rc}): {err.strip()[:500]}"

        return False, None

    except Exception as e:
        return None, f"pdftotext error: {e}"


def pdf_contains_needle_pypdf(pdf_path: Path, needle: str, case_sensitive: bool) -> Tuple[Optional[bool], Optional[str]]:
    """
    Fallback: page-by-page extraction via pypdf; early exit when match is found.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None, "pypdf not installed (pip install pypdf)"

    needle_n = _norm(needle, case_sensitive)

    try:
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        limit = PDF_MAX_PAGES if PDF_MAX_PAGES is not None else num_pages

        for i in range(min(num_pages, limit)):
            page = reader.pages[i]
            text = page.extract_text() or ""
            if len(text) > PDF_PAGE_CHUNK_MAX_CHARS:
                text = text[:PDF_PAGE_CHUNK_MAX_CHARS]

            if needle_n in _norm(text, case_sensitive):
                return True, None

        return False, None
    except Exception as e:
        return None, f"pypdf failed: {e}"


def pdf_contains_needle(pdf_path: Path, needle: str, case_sensitive: bool) -> Tuple[Optional[bool], Optional[str]]:
    """
    Prefer pdftotext (streamed). Fallback to pypdf page-by-page.
    """
    ok, err = pdf_contains_needle_pdftotext(pdf_path, needle, case_sensitive)
    if ok is not None:
        return ok, err
    return pdf_contains_needle_pypdf(pdf_path, needle, case_sensitive)


# ----------------------------
# Optional Office extraction
# ----------------------------

def extract_docx_text(path: Path, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        import docx  # type: ignore
    except Exception:
        return None, "python-docx not installed (pip install python-docx)"

    try:
        d = docx.Document(str(path))
        parts = []
        total = 0
        for p in d.paragraphs:
            s = p.text
            if s:
                parts.append(s)
                total += len(s)
                if total >= max_chars:
                    break
        return "\n".join(parts)[:max_chars], None
    except Exception as e:
        return None, f"docx read failed: {e}"


def extract_xlsx_text(path: Path, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        import openpyxl  # type: ignore
    except Exception:
        return None, "openpyxl not installed (pip install openpyxl)"

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        parts: List[str] = []
        total = 0
        for ws in wb.worksheets:
            parts.append(f"[SHEET] {ws.title}")
            for row in ws.iter_rows(values_only=True):
                line = "\t".join("" if v is None else str(v) for v in row)
                if line.strip():
                    parts.append(line)
                    total += len(line) + 1
                    if total >= max_chars:
                        wb.close()
                        return "\n".join(parts)[:max_chars], None
        wb.close()
        return "\n".join(parts)[:max_chars], None
    except Exception as e:
        return None, f"xlsx read failed: {e}"


# ----------------------------
# Dispatch
# ----------------------------

# For PDFs we return a sentinel that tells the worker to do a direct PDF search without full extraction.
PDF_DIRECT_SEARCH = object()

ExtractResult = Union[str, object]  # str text blob, or PDF_DIRECT_SEARCH sentinel


def extract_text_for_file(path: Path) -> Tuple[Optional[ExtractResult], Optional[str]]:
    ext = path.suffix.lower()

    if ext in PLAIN_TEXT_EXTS:
        text, err = read_text_safely(path, MAX_TEXT_CHARS)
        return text, err

    if ext in CSV_EXTS:
        text, err = extract_csv_text(path, MAX_TEXT_CHARS)
        return text, err

    if ext in PDF_EXTS:
        # optimized direct search instead of full extraction
        return PDF_DIRECT_SEARCH, None

    if ext in DOCX_EXTS:
        text, err = extract_docx_text(path, MAX_TEXT_CHARS)
        return text, err

    if ext in XLSX_EXTS:
        text, err = extract_xlsx_text(path, MAX_TEXT_CHARS)
        return text, err

    return None, "Unsupported for content search"


# ----------------------------
# Search string handling
# ----------------------------

def parse_user_needle(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


# ----------------------------
# GUI App
# ----------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Folder String Finder (Filename + Content)")
        self.root.geometry("900x600")

        self.folder_var = tk.StringVar(value="")
        self.needle_var = tk.StringVar(value="")
        self.case_var = tk.BooleanVar(value=True)

        self._worker_thread: Optional[threading.Thread] = None
        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_flag = threading.Event()

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        folder_row = ttk.Frame(frm)
        folder_row.pack(fill="x", pady=(0, 8))

        ttk.Label(folder_row, text="Folder:").pack(side="left")
        self.folder_entry = ttk.Entry(folder_row, textvariable=self.folder_var)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(folder_row, text="Choose…", command=self.choose_folder).pack(side="left")

        needle_row = ttk.Frame(frm)
        needle_row.pack(fill="x", pady=(0, 8))

        ttk.Label(needle_row, text="Search string:").pack(side="left")
        self.needle_entry = ttk.Entry(needle_row, textvariable=self.needle_var)
        self.needle_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

        opt_row = ttk.Frame(frm)
        opt_row.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(opt_row, text="Case-sensitive", variable=self.case_var).pack(side="left")

        ttk.Label(
            opt_row,
            text='Tip: Optional outer quotes mean “literal”: e.g., "foo bar?"'
        ).pack(side="left", padx=(12, 0))

        ctrl_row = ttk.Frame(frm)
        ctrl_row.pack(fill="x", pady=(0, 8))

        self.start_btn = ttk.Button(ctrl_row, text="Start Search", command=self.start_search)
        self.start_btn.pack(side="left")

        self.stop_btn = ttk.Button(ctrl_row, text="Stop", command=self.stop_search, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(ctrl_row, textvariable=self.status_var).pack(side="right")

        res_frame = ttk.LabelFrame(frm, text="Matches (full paths)", padding=8)
        res_frame.pack(fill="both", expand=True)

        self.results = tk.Text(res_frame, wrap="none")
        self.results.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(res_frame, orient="vertical", command=self.results.yview)
        yscroll.pack(side="right", fill="y")
        self.results.configure(yscrollcommand=yscroll.set)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select a folder to scan")
        if folder:
            self.folder_var.set(folder)

    def start_search(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Search running", "A search is already running.")
            return

        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Missing folder", "Please choose a folder.")
            return

        raw = self.needle_var.get()
        needle = parse_user_needle(raw)
        if not needle:
            messagebox.showerror("Missing search string", "Please enter a search string.")
            return

        root_path = Path(folder)
        if not root_path.exists() or not root_path.is_dir():
            messagebox.showerror("Invalid folder", "Selected folder does not exist or is not a directory.")
            return

        self.results.delete("1.0", "end")
        self.status_var.set("Starting…")
        self._stop_flag.clear()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        case_sensitive = self.case_var.get()
        self._worker_thread = threading.Thread(
            target=self._search_worker,
            args=(root_path, needle, case_sensitive),
            daemon=True,
        )
        self._worker_thread.start()

    def stop_search(self):
        self._stop_flag.set()
        self.status_var.set("Stopping…")

    def _search_worker(self, root_path: Path, needle: str, case_sensitive: bool):
        needle_norm = needle if case_sensitive else needle.lower()

        def norm(s: str) -> str:
            return s if case_sensitive else s.lower()

        scanned = 0
        matches: List[str] = []
        pdf_scanned = 0

        for path in iter_files(root_path):
            if self._stop_flag.is_set():
                break

            scanned += 1

            # Filename match (all files)
            if needle_norm in norm(path.name):
                matches.append(str(path))
                self._msg_queue.put(("match", str(path)))
                continue

            # Content match (supported types)
            extracted, _err = extract_text_for_file(path)
            if extracted is None:
                continue

            # PDF: direct optimized search
            if extracted is PDF_DIRECT_SEARCH:
                pdf_scanned += 1
                ok, _perr = pdf_contains_needle(path, needle, case_sensitive)
                if ok is True:
                    matches.append(str(path))
                    self._msg_queue.put(("match", str(path)))
                # If ok is None, extraction failed; we silently skip (could log if you want)
            else:
                # Text blob search
                if needle_norm in norm(str(extracted)):
                    matches.append(str(path))
                    self._msg_queue.put(("match", str(path)))

            if scanned % 200 == 0:
                self._msg_queue.put(("status", f"Scanned {scanned:,} files (PDFs checked: {pdf_scanned:,})… Matches: {len(matches):,}"))

        out_path = root_path / "matches.txt"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for m in matches:
                    f.write(m + "\n")
            self._msg_queue.put(("done", scanned, len(matches), str(out_path), self._stop_flag.is_set()))
        except Exception as e:
            self._msg_queue.put(("error", f"Failed writing matches.txt: {e}"))

    def _poll_queue(self):
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                kind = msg[0]

                if kind == "match":
                    path = msg[1]
                    self.results.insert("end", path + "\n")
                    self.results.see("end")

                elif kind == "status":
                    self.status_var.set(msg[1])

                elif kind == "error":
                    self.status_var.set("Error.")
                    messagebox.showerror("Error", msg[1])
                    self._reset_buttons()

                elif kind == "done":
                    scanned, match_count, out_path, stopped = msg[1], msg[2], msg[3], msg[4]
                    if stopped:
                        self.status_var.set(f"Stopped. Scanned {scanned:,} files. Matches: {match_count:,}. Saved: {out_path}")
                    else:
                        self.status_var.set(f"Done. Scanned {scanned:,} files. Matches: {match_count:,}. Saved: {out_path}")
                    self._reset_buttons()

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _reset_buttons(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
