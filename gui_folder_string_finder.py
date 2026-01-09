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
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Union
import fnmatch
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAVE_TK = True
except Exception:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    HAVE_TK = False
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

# Scan limits
FILE_SIZE_CAP_BYTES = 50 * 1024 * 1024  # 50 MB per file
MAX_FILES_DEFAULT: Optional[int] = 200_000
MAX_DEPTH_DEFAULT: Optional[int] = 3  # None = unlimited
ESTIMATE_WARN_FILE_COUNT = 50_000
ESTIMATE_HARD_CAP = 250_000  # stop estimating after this many to keep UI responsive
ESTIMATE_TIME_BUDGET_SEC = 5  # time cap for estimation loop
HOME_ROOT = Path.home().resolve()
EXCLUDE_DIR_NAMES = {
    ".git",
    ".svn",
    ".hg",
    ".Spotlight-V100",
    ".fseventsd",
    ".DocumentRevisions-V100",
    ".TemporaryItems",
    ".Trash",
    "Library",  # user Library
    "Applications",
}
EXCLUDE_FILE_NAMES = {".DS_Store"}
WORKER_THREADS_DEFAULT = 4
AUTHOR_NAME = "Frederick Gault"
RELEASE_DATE = "2025-12-28"  # TODO: update to today if different
RELEASE_LEVEL = "v1.0.0"

EXTENSION_OPTIONS = [
    ("Text / Code", PLAIN_TEXT_EXTS, True),
    ("CSV / TSV", CSV_EXTS, True),
    ("PDF", PDF_EXTS, False),    # expensive
    ("DOCX", DOCX_EXTS, False),  # expensive
    ("XLSX", XLSX_EXTS, False),  # expensive
]


# ----------------------------
# ETA helpers
# ----------------------------

def _parse_estimated_total(estimated_total_str: Optional[str]) -> Optional[int]:
    if not estimated_total_str:
        return None
    digits = "".join(ch for ch in estimated_total_str if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _format_eta(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    m, s = divmod(int(round(seconds)), 60)
    if m == 0:
        return f"{s}s"
    h, m = divmod(m, 60)
    if h == 0:
        return f"{m}m{s:02d}s"
    return f"{h}h{m:02d}m"


# Simple tooltip helper
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#333333",
            foreground="#ffffff",
        )
        label.pack(ipadx=4, ipady=2)

    def hide_tip(self, _event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# module logger
logger = logging.getLogger(__name__)


# ----------------------------
# File walking / reading
# ----------------------------

def _is_under_home(path: Path) -> bool:
    try:
        path.resolve().relative_to(HOME_ROOT)
        return True
    except Exception:
        return False


def iter_files(root: Path, skip_hidden: bool = False, max_depth: Optional[int] = None, skip_symlinks: bool = True, skip_system: bool = True) -> Iterable[Path]:
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]
        if skip_system:
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            filenames = [f for f in filenames if f not in EXCLUDE_FILE_NAMES]
        if skip_symlinks:
            dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for fn in filenames:
            candidate = Path(dirpath) / fn
            if skip_symlinks and candidate.is_symlink():
                continue
            yield candidate


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


def pdf_contains_needle_pdftotext(pdf_path: Path, needle: str, case_sensitive: bool, stop_event: Optional[threading.Event] = None) -> Tuple[Optional[bool], Optional[str]]:
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
            if stop_event is not None and stop_event.is_set():
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1)
                    except Exception:
                        pass
                return None, "stopped"
            try:
                chunk = proc.stdout.read(64 * 1024)  # 64KB chunks
            except Exception as e:
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except Exception:
                    pass
                return None, f"pdftotext read error: {e}"
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
        if stop_event is not None and stop_event.is_set():
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            return None, "stopped"
        try:
            rc = proc.wait(timeout=PDFTOTEXT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                logger.debug("pdftotext: timeout, terminating process")
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.debug("pdftotext: terminate failed, killing process")
                proc.kill()
                proc.wait()
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


def pdf_contains_needle_pypdf(pdf_path: Path, needle: str, case_sensitive: bool, stop_event: Optional[threading.Event] = None) -> Tuple[Optional[bool], Optional[str]]:
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
            if stop_event is not None and stop_event.is_set():
                return None, "stopped"
            page = reader.pages[i]
            text = page.extract_text() or ""
            if len(text) > PDF_PAGE_CHUNK_MAX_CHARS:
                text = text[:PDF_PAGE_CHUNK_MAX_CHARS]

            if needle_n in _norm(text, case_sensitive):
                return True, None

        return False, None
    except Exception as e:
        return None, f"pypdf failed: {e}"


def pdf_contains_needle(pdf_path: Path, needle: str, case_sensitive: bool, stop_event: Optional[threading.Event] = None) -> Tuple[Optional[bool], Optional[str]]:
    """
    Prefer pdftotext (streamed). Fallback to pypdf page-by-page.
    """
    ok, err = pdf_contains_needle_pdftotext(pdf_path, needle, case_sensitive, stop_event=stop_event)
    if ok is not None:
        return ok, err
    return pdf_contains_needle_pypdf(pdf_path, needle, case_sensitive, stop_event=stop_event)


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
# Headless search API
# ----------------------------

from typing import Callable


def search_folder(
    root_path: Path,
    needle: str,
    case_sensitive: bool,
    stop_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[str, str], None]] = None,
    skip_hidden: bool = True,
    skip_system: bool = True,
    skip_symlinks: bool = True,
    file_size_cap_bytes: int = FILE_SIZE_CAP_BYTES,
    max_files: Optional[int] = MAX_FILES_DEFAULT,
    max_depth: Optional[int] = MAX_DEPTH_DEFAULT,
    search_content: bool = True,
    filename_wildcard: bool = False,
    include_exts: Optional[set[str]] = None,
    worker_threads: int = WORKER_THREADS_DEFAULT,
    estimated_total_str: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """Search folder for needle. Returns (scanned_count, matches_list).

    progress(kind, data) is called for 'match' and 'status' updates if provided.
    """
    needle_norm = needle if case_sensitive else needle.lower()

    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    def filename_matches(name: str) -> bool:
        target = name if case_sensitive else name.lower()
        patt = needle if case_sensitive else needle.lower()
        if filename_wildcard:
            return fnmatch.fnmatchcase(target, patt)
        return patt in target

    if not _is_under_home(root_path):
        raise ValueError(f"Selected folder must be under your home directory: {HOME_ROOT}")

    scanned = 0
    matches: List[str] = []
    pdf_scanned = 0
    skipped_oversize = 0
    skipped_errors = 0
    last_status = time.monotonic()
    start_time = last_status
    est_total = _parse_estimated_total(estimated_total_str)

    def worker_fn(path: Path):
        matched: List[str] = []
        oversize = 0
        errors = 0
        pdf_checked = 0
        if stop_event is not None and stop_event.is_set():
            return matched, oversize, errors, pdf_checked
        ext = path.suffix.lower()
        if include_exts is not None and ext not in include_exts:
            return matched, oversize, errors, pdf_checked
        try:
            # Filename match
            if filename_matches(path.name):
                matched.append(str(path))
                if not search_content:
                    return matched, oversize, errors, pdf_checked
                # fall through if we also scan content
        except Exception:
            errors += 1
            return matched, oversize, errors, pdf_checked

        if search_content:
            try:
                size = path.stat().st_size
            except Exception:
                errors += 1
                return matched, oversize, errors, pdf_checked

            if size > file_size_cap_bytes:
                oversize += 1
                return matched, oversize, errors, pdf_checked

            try:
                extracted, _err = extract_text_for_file(path)
            except Exception:
                errors += 1
                return matched, oversize, errors, pdf_checked

            if extracted is None:
                errors += 1
                return matched, oversize, errors, pdf_checked

            if extracted is PDF_DIRECT_SEARCH:
                pdf_checked += 1
                ok, _perr = pdf_contains_needle(path, needle, case_sensitive, stop_event=stop_event)
                if ok is True:
                    matched.append(str(path))
                elif ok is None:
                    errors += 1
                return matched, oversize, errors, pdf_checked

            try:
                if needle_norm in norm(str(extracted)):
                    matched.append(str(path))
            except Exception:
                errors += 1

        return matched, oversize, errors, pdf_checked

    def file_iter():
        count = 0
        for path in iter_files(root_path, skip_hidden=skip_hidden, max_depth=max_depth, skip_symlinks=skip_symlinks, skip_system=skip_system):
            if stop_event is not None and stop_event.is_set():
                break
            if max_files is not None and count >= max_files:
                break
            yield path
            count += 1

    with ThreadPoolExecutor(max_workers=max(1, worker_threads)) as executor:
        for matched, oversize, errors, pdf_checked in executor.map(worker_fn, file_iter()):
            if stop_event is not None and stop_event.is_set():
                break
            scanned += 1
            if matched:
                matches.extend(matched)
                if progress:
                    for m in matched:
                        progress('match', m)
            skipped_oversize += oversize
            skipped_errors += errors
            pdf_scanned += pdf_checked

        if progress:
            now = time.monotonic()
            if scanned % 50 == 0 or (now - last_status) >= 0.5:
                eta_str = ""
                if est_total is not None and scanned > 0:
                    remaining = max(est_total - scanned, 0)
                    eta = (now - start_time) / scanned * remaining if scanned else None
                    if eta is not None:
                        eta_str = f" ETA ~{_format_eta(eta)}"
                progress('status', f"Scanned {scanned:,} files (PDFs checked: {pdf_scanned:,})… Matches: {len(matches):,}. Oversize skipped: {skipped_oversize:,}. Files skipped: {skipped_errors:,}.{eta_str}")
                last_status = now

    if progress and skipped_errors:
        progress('status', f"Finished with {skipped_errors:,} files skipped due to errors.")

    return scanned, matches


# ----------------------------
# GUI App
# ----------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Search My Mac")
        self.root.geometry("900x600")

        self.folder_var = tk.StringVar(value="")
        self.needle_var = tk.StringVar(value="")
        self.case_var = tk.BooleanVar(value=True)
        self.skip_hidden_var = tk.BooleanVar(value=True)
        self.skip_system_var = tk.BooleanVar(value=True)
        self.search_mode_var = tk.StringVar(value="name")  # "name" or "both"
        self.ext_vars = {}
        self.size_cap_mb_var = tk.StringVar(value=str(FILE_SIZE_CAP_BYTES // (1024 * 1024)))
        self.max_files_var = tk.StringVar(value=str(MAX_FILES_DEFAULT) if MAX_FILES_DEFAULT is not None else "")
        self.max_depth_var = tk.StringVar(value="" if MAX_DEPTH_DEFAULT is None else str(MAX_DEPTH_DEFAULT))
        self.skip_symlinks = True

        self._worker_thread: Optional[threading.Thread] = None
        self._estimate_thread: Optional[threading.Thread] = None
        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_flag = threading.Event()
        self._pending_search_args: Optional[tuple] = None

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        folder_row = ttk.Frame(frm)
        folder_row.pack(fill="x", pady=(0, 8))

        ttk.Label(folder_row, text="Folder to Search:").pack(side="left")
        self.folder_entry = ttk.Entry(folder_row, textvariable=self.folder_var)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        choose_btn = ttk.Button(folder_row, text="Choose…", command=self.choose_folder)
        choose_btn.pack(side="left")
        ToolTip(choose_btn, "Click here to select the top most folder to search.")

        needle_row = ttk.Frame(frm)
        needle_row.pack(fill="x", pady=(0, 8))

        ttk.Label(needle_row, text="Search string:").pack(side="left")
        self.needle_entry = ttk.Entry(needle_row, textvariable=self.needle_var)
        self.needle_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        ToolTip(self.needle_entry, 'Use "*" as a wildcard for filenames only. If content search is enabled, "*" is treated as a literal.')

        opt_row = ttk.Frame(frm)
        opt_row.pack(fill="x", pady=(0, 8))

        case_cb = ttk.Checkbutton(opt_row, text="Case-sensitive", variable=self.case_var)
        case_cb.pack(side="left")
        ToolTip(case_cb, "Example: PartyList.txt vs partylist.txt")
        skip_hidden_cb = ttk.Checkbutton(opt_row, text="Skip hidden files/folders", variable=self.skip_hidden_var)
        skip_hidden_cb.pack(side="left", padx=(12, 0))
        ToolTip(skip_hidden_cb, "You probably don't want to search these files.")
        skip_system_cb = ttk.Checkbutton(opt_row, text="Skip system/cache folders", variable=self.skip_system_var)
        skip_system_cb.pack(side="left", padx=(12, 0))
        ToolTip(skip_system_cb, "You probably don't want to search these files.")

        limits_row = ttk.Frame(frm)
        limits_row.pack(fill="x", pady=(0, 8))

        max_size_label = ttk.Label(limits_row, text="Max file size (MB):")
        max_size_label.pack(side="left")
        max_size_entry = ttk.Entry(limits_row, width=8, textvariable=self.size_cap_mb_var)
        max_size_entry.pack(side="left", padx=(4, 12))
        ToolTip(max_size_label, "Lowering max file size can speed up searches.")
        ToolTip(max_size_entry, "Lowering max file size can speed up searches.")

        max_files_label = ttk.Label(limits_row, text="Max files (0/blank = unlimited):")
        max_files_label.pack(side="left")
        max_files_entry = ttk.Entry(limits_row, width=10, textvariable=self.max_files_var)
        max_files_entry.pack(side="left", padx=(4, 12))
        ToolTip(max_files_label, "Stop at this many files.")
        ToolTip(max_files_entry, "Stop at this many files.")

        depth_label = ttk.Label(limits_row, text="Folder depth (blank = unlimited):")
        depth_label.pack(side="left")
        depth_entry = ttk.Entry(limits_row, width=6, textvariable=self.max_depth_var)
        depth_entry.pack(side="left", padx=(4, 0))
        ToolTip(depth_label, "How many folder levels below the selected folder to include.")
        ToolTip(depth_entry, "How many folder levels below the selected folder to include.")

        ext_row = ttk.Frame(frm)
        ext_row.pack(fill="x", pady=(0, 8))
        ttk.Label(ext_row, text="File types:").pack(side="left")
        for label, _exts, default_on in EXTENSION_OPTIONS:
            var = tk.BooleanVar(value=default_on)
            self.ext_vars[label] = var
            cb = ttk.Checkbutton(ext_row, text=label, variable=var)
            cb.pack(side="left", padx=(6, 0))
            if label == "Text / Code":
                ToolTip(cb, "Includes: .txt, .md, .rst, .log, .py, .js, .ts, .html, .css, .json, .xml, .yaml, .yml, .ini, .cfg, .toml, .sql")

        ttk.Label(
            opt_row,
            text='Tip: Optional outer quotes mean “literal”: e.g., "foo bar?"'
        ).pack(side="left", padx=(12, 0))

        ctrl_row = ttk.Frame(frm)
        ctrl_row.pack(fill="x", pady=(0, 8))

        self.start_btn = ttk.Button(ctrl_row, text="Start Search", command=self.start_search)
        self.start_btn.pack(side="left")
        ToolTip(self.start_btn, "Click here to start the search. Results are saved to your Documents folder as _Search_My_Mac_YYYY-MM-DD_HH-MM-SS.txt")

        self.stop_btn = ttk.Button(ctrl_row, text="Stop", command=self.stop_search, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ToolTip(self.stop_btn, "You can safely stop at any point.")

        mode_frame = ttk.Frame(ctrl_row)
        mode_frame.pack(side="left", padx=(12, 0))
        name_only_rb = ttk.Radiobutton(mode_frame, text="File name only", variable=self.search_mode_var, value="name")
        name_only_rb.pack(side="left")
        ToolTip(name_only_rb, "Only look for the search string in file names. * wildcard is okay.")
        both_rb = ttk.Radiobutton(mode_frame, text="File name AND File contents", variable=self.search_mode_var, value="both")
        both_rb.pack(side="left", padx=(6, 0))
        ToolTip(both_rb, "Search for the search string in the filename AND inside the file. '*' is just another character.")

        about_btn = ttk.Button(ctrl_row, text="About", width=6, command=self.show_about)
        about_btn.pack(side="right")
        ToolTip(about_btn, "Show author, date, and release level.")

        self.progress_var = tk.StringVar(value="")
        progress_row = ttk.Frame(frm)
        progress_row.pack(fill="x", pady=(0, 4))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(progress_row, textvariable=self.status_var).pack(side="left", padx=(0, 8))
        ttk.Label(progress_row, textvariable=self.progress_var).pack(side="left")

        res_frame = ttk.LabelFrame(frm, text="Matches (full paths)", padding=8)
        res_frame.pack(fill="both", expand=True)

        self.results = tk.Text(res_frame, wrap="none", bg="white", fg="black", insertbackground="black")
        self.results.pack(side="left", fill="both", expand=True)
        # Results selection UX:
        # - Single click selects exactly one full line (path) under the cursor.
        # - Double click opens ONLY the currently selected line (if any).
        self.results.bind("<Button-1>", self.results_single_click)
        # Prevent click-drag from selecting multiple lines (keeps selection single-line and stable).
        self.results.bind("<B1-Motion>", lambda e: "break")
        self.results.bind("<Shift-Button-1>", lambda e: "break")
        self.results.bind("<Double-Button-1>", self.open_selected_event)
        ToolTip(self.results, "Double-click a single line to open that file. Select a single line and click Open to launch it.")

        yscroll = ttk.Scrollbar(res_frame, orient="vertical", command=self.results.yview)
        yscroll.pack(side="right", fill="y")
        self.results.configure(yscrollcommand=yscroll.set)

        actions_row = ttk.Frame(frm)
        actions_row.pack(fill="x", pady=(4, 0))
        open_btn = ttk.Button(actions_row, text="Open selected file", command=self.open_selected)
        open_btn.pack(side="left")
        ToolTip(open_btn, "Opens the highlighted result with the default app for that file.")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select a folder to scan")
        if folder:
            self.folder_var.set(folder)

    def start_search(self):
        if (self._worker_thread and self._worker_thread.is_alive()) or (self._estimate_thread and self._estimate_thread.is_alive()):
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

        # Parse limits
        try:
            cap_mb = int(self.size_cap_mb_var.get().strip()) if self.size_cap_mb_var.get().strip() else FILE_SIZE_CAP_BYTES // (1024 * 1024)
            if cap_mb <= 0:
                raise ValueError
            file_size_cap_bytes = cap_mb * 1024 * 1024
        except ValueError:
            messagebox.showerror("Invalid max file size", "Max file size must be a positive integer (MB).")
            return

        try:
            max_files_str = self.max_files_var.get().strip()
            if not max_files_str:
                max_files = MAX_FILES_DEFAULT
            else:
                max_files_val = int(max_files_str)
                max_files = None if max_files_val <= 0 else max_files_val
        except ValueError:
            messagebox.showerror("Invalid max files", "Max files must be an integer.")
            return

        try:
            max_depth_str = self.max_depth_var.get().strip()
            if not max_depth_str:
                max_depth = MAX_DEPTH_DEFAULT
            else:
                max_depth_val = int(max_depth_str)
                max_depth = None if max_depth_val <= 0 else max_depth_val
        except ValueError:
            messagebox.showerror("Invalid max depth", "Max depth must be an integer.")
            return

        self.results.delete("1.0", "end")
        self.status_var.set("Estimating…")
        self.progress_var.set("Estimating file count…")
        self._stop_flag.clear()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        case_sensitive = self.case_var.get()
        skip_hidden = self.skip_hidden_var.get()
        skip_system = self.skip_system_var.get()
        search_content = self.search_mode_var.get() == "both"
        has_star = "*" in needle
        filename_wildcard = (not search_content) and has_star
        include_exts: set[str] = set()
        for label, exts, _default_on in EXTENSION_OPTIONS:
            if self.ext_vars[label].get():
                include_exts.update(exts)
        if not include_exts:
            messagebox.showerror("No file types selected", "Please select at least one file type.")
            self._reset_buttons()
            return

        try:
            if not _is_under_home(root_path):
                messagebox.showerror("Invalid folder", f"Please choose a folder under your home directory: {HOME_ROOT}")
                self._reset_buttons()
                return
        except Exception as e:
            messagebox.showerror("Invalid folder", f"Could not validate folder: {e}")
            self._reset_buttons()
            return

        self._pending_search_args = (root_path, needle, case_sensitive, skip_hidden, skip_system, self.skip_symlinks, file_size_cap_bytes, max_files, max_depth, search_content, include_exts, filename_wildcard, WORKER_THREADS_DEFAULT)
        self._estimate_thread = threading.Thread(
            target=self._estimate_worker,
            args=(root_path, skip_hidden, skip_system, self.skip_symlinks, max_depth, max_files),
            daemon=True,
        )
        self._estimate_thread.start()

    def stop_search(self):
        self._stop_flag.set()
        self.status_var.set("Stopping…")

    def show_about(self):
        msg = f"Author: {AUTHOR_NAME}\nRelease date: {RELEASE_DATE}\nRelease level: {RELEASE_LEVEL}"
        messagebox.showinfo("About", msg)

    def results_single_click(self, event):
        """Single-click selects exactly one full line (path) under the cursor.

        Clicking below the last line clears the selection.
        """
        w = self.results
        w.focus_set()

        # If you click below the last rendered line, clear selection (avoid selecting last item).
        try:
            last_idx = w.index("end-1c")
            last_line = int(last_idx.split(".")[0])
            bbox = w.bbox(f"{last_line}.0")
            if bbox is not None:
                _x, y, _w, h = bbox
                if event.y > (y + h):
                    w.tag_remove("sel", "1.0", "end")
                    return "break"
        except Exception:
            # If anything goes wrong, fall back to normal selection behavior.
            pass

        idx = w.index(f"@{event.x},{event.y}")
        line = int(idx.split(".")[0])
        line_start = f"{line}.0"
        line_end = f"{line}.0 lineend"

        # If the clicked line is empty/whitespace, treat it as 'no selection'.
        if not w.get(line_start, line_end).strip():
            w.tag_remove("sel", "1.0", "end")
            return "break"

        w.tag_remove("sel", "1.0", "end")
        w.tag_add("sel", line_start, line_end)
        w.mark_set("insert", line_start)
        w.see(line_start)
        return "break"

    def open_selected_event(self, event):
        # Double-click should open ONLY an already-selected single line.
        # If nothing is selected, do nothing (avoid surprising opens).
        if not self.results.tag_ranges("sel"):
            return "break"
        self.open_selected(event=None, force_line=False)
        return "break"
    def open_selected(self, event=None, force_line: bool = False):
        try:
            selection = ""
            if not force_line and self.results.tag_ranges("sel"):
                selection = self.results.get("sel.first", "sel.last")
            else:
                # Always use the single line under cursor/insertion when forced
                idx = None
                if event is not None:
                    idx = self.results.index(f"@{event.x},{event.y}")
                if idx is None:
                    idx = self.results.index("insert")
                selection = self.results.get(f"{idx} linestart", f"{idx} lineend")

            lines = [line.strip() for line in selection.splitlines() if line.strip()]
            if not lines:
                messagebox.showinfo("Open file", "Select a single file path in the results, then click Open.")
                return
            if len(lines) != 1:
                messagebox.showinfo("Open file", "Please select only one file path at a time.")
                return

            path = lines[0]
            p = Path(path)
            if not p.exists():
                messagebox.showerror("Open file", f"File does not exist:\n{path}")
                return
            if not p.is_file():
                messagebox.showerror("Open file", f"Selection is not a file:\n{path}")
                return

            try:
                proc = subprocess.run(["open", str(p)], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                msg = e.stderr.strip() or e.stdout.strip() or str(e)
                messagebox.showerror("Open file", f"Failed to open file:\n{path}\n\n{msg}")
            except Exception as e:
                messagebox.showerror("Open file", f"Failed to open file:\n{path}\n\n{e}")
        except Exception as e:
            messagebox.showerror("Open file", f"Unexpected error: {e}")

    def _estimate_worker(self, root_path: Path, skip_hidden: bool, skip_system: bool, skip_symlinks: bool, max_depth: Optional[int], max_files: Optional[int]):
        count = 0
        truncated = False
        stopped = False
        start = time.monotonic()
        for _ in iter_files(root_path, skip_hidden=skip_hidden, max_depth=max_depth, skip_symlinks=skip_symlinks, skip_system=skip_system):
            if self._stop_flag.is_set():
                stopped = True
                break
            count += 1
            if max_files is not None and count >= max_files:
                break
            if count >= ESTIMATE_HARD_CAP:
                truncated = True
                break
            if (time.monotonic() - start) >= ESTIMATE_TIME_BUDGET_SEC:
                truncated = True
                break

        self._msg_queue.put(("estimate_done", count, truncated, stopped))

    def _search_worker(self, root_path: Path, needle: str, case_sensitive: bool, skip_hidden: bool, skip_system: bool, skip_symlinks: bool, file_size_cap_bytes: int, max_files: Optional[int], max_depth: Optional[int], search_content: bool, include_exts: set[str], filename_wildcard: bool, worker_threads: int, estimated_total_str: Optional[str]):
        needle_norm = needle if case_sensitive else needle.lower()

        def norm(s: str) -> str:
            return s if case_sensitive else s.lower()

        def filename_matches(name: str) -> bool:
            target = name if case_sensitive else name.lower()
            patt = needle if case_sensitive else needle.lower()
            if filename_wildcard:
                return fnmatch.fnmatchcase(target, patt)
            return patt in target

        scanned = 0
        matches: List[str] = []
        pdf_scanned = 0
        skipped_oversize = 0
        limit_hit = False
        skipped_errors = 0
        last_status = time.monotonic()
        start_time = last_status
        est_total = _parse_estimated_total(estimated_total_str)

        def worker_fn(path: Path):
            matched: List[str] = []
            oversize = 0
            errors = 0
            pdf_checked = 0
            if self._stop_flag.is_set():
                return matched, oversize, errors, pdf_checked
            ext = path.suffix.lower()
            if include_exts is not None and ext not in include_exts:
                return matched, oversize, errors, pdf_checked
            try:
                if filename_matches(path.name):
                    matched.append(str(path))
                    if not search_content:
                        return matched, oversize, errors, pdf_checked
            except Exception:
                errors += 1
                return matched, oversize, errors, pdf_checked

            if search_content:
                try:
                    size = path.stat().st_size
                except Exception:
                    errors += 1
                    return matched, oversize, errors, pdf_checked

                if size > file_size_cap_bytes:
                    oversize += 1
                    return matched, oversize, errors, pdf_checked

                try:
                    extracted, _err = extract_text_for_file(path)
                except Exception:
                    errors += 1
                    return matched, oversize, errors, pdf_checked
                if extracted is None:
                    errors += 1
                    return matched, oversize, errors, pdf_checked

                if extracted is PDF_DIRECT_SEARCH:
                    pdf_checked += 1
                    ok, _perr = pdf_contains_needle(path, needle, case_sensitive, stop_event=self._stop_flag)
                    if ok is True:
                        matched.append(str(path))
                    elif ok is None:
                        errors += 1
                    return matched, oversize, errors, pdf_checked

                try:
                    if needle_norm in norm(str(extracted)):
                        matched.append(str(path))
                except Exception:
                    errors += 1
            return matched, oversize, errors, pdf_checked

        def file_iter():
            count = 0
            for path in iter_files(root_path, skip_hidden=skip_hidden, max_depth=max_depth, skip_symlinks=skip_symlinks, skip_system=skip_system):
                if self._stop_flag.is_set():
                    break
                if max_files is not None and count >= max_files:
                    nonlocal limit_hit
                    limit_hit = True
                    break
                yield path
                count += 1
        with ThreadPoolExecutor(max_workers=max(1, worker_threads)) as executor:
            for matched, oversize, errors, pdf_checked in executor.map(worker_fn, file_iter()):
                if self._stop_flag.is_set():
                    break
                scanned += 1
                if matched:
                    matches.extend(matched)
                    for m in matched:
                        self._msg_queue.put(("match", m))
                skipped_oversize += oversize
                skipped_errors += errors
                pdf_scanned += pdf_checked

                now = time.monotonic()
                if scanned % 50 == 0 or (now - last_status) >= 0.5:
                    total_hint = estimated_total_str or "?"
                    eta_str = ""
                    if est_total is not None and scanned > 0:
                        remaining = max(est_total - scanned, 0)
                        eta = (now - start_time) / scanned * remaining if scanned else None
                        if eta is not None:
                            eta_str = f" ETA ~{_format_eta(eta)}"
                    self._msg_queue.put(("status", f"Scanned {scanned:,} / {total_hint} files (PDFs checked: {pdf_scanned:,})… Matches: {len(matches):,}. Oversize skipped: {skipped_oversize:,}. Files skipped: {skipped_errors:,}.{eta_str}"))
                    last_status = now

        # limit_hit is set inside file_iter when max_files is reached

        now = datetime.now().astimezone()
        timestamp_file = now.strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_human = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        out_dir = Path.home() / "Documents"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"_Search_My_Mac_{timestamp_file}.txt"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(f"Search started: {timestamp_human}\n")
                f.write(f"Folder: {root_path}\n")
                f.write(f'Search string: "{needle}"\n')
                f.write(f"Case-sensitive: {case_sensitive}\n")
                f.write(f"Mode: {'Filename + content' if search_content else 'Filename only'}\n")
                f.write(f"Filename wildcard: {filename_wildcard}\n")
                f.write(f"Skip hidden: {skip_hidden}\n")
                f.write(f"Skip system: {skip_system}\n")
                f.write(f"Skip symlinks: {skip_symlinks}\n")
                f.write(f"File size cap: {file_size_cap_bytes} bytes (~{file_size_cap_bytes / (1024*1024):.2f} MB)\n")
                f.write(f"Max files: {max_files if max_files is not None else 'unlimited'}\n")
                f.write(f"Folder depth: {max_depth if max_depth is not None else 'unlimited'}\n")
                f.write(f"Included extensions: {', '.join(sorted(include_exts)) if include_exts else 'all'}\n")
                f.write("\nMatches:\n")
                for m in matches:
                    f.write(m + "\n")
            self._msg_queue.put(("done", scanned, len(matches), str(out_path), self._stop_flag.is_set(), limit_hit, skipped_oversize, skipped_errors))
        except Exception as e:
            self._msg_queue.put(("error", f"Failed writing output file: {e}"))

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
                    self.status_var.set("Running…")
                    self.progress_var.set(msg[1])

                elif kind == "estimate_done":
                    count, truncated, stopped = msg[1], msg[2], msg[3]
                    self._estimate_thread = None
                    if stopped or self._stop_flag.is_set():
                        self.status_var.set("Cancelled before start.")
                        self.progress_var.set("")
                        self._reset_buttons()
                        continue
                    approx = f"at least {count:,}" if truncated else f"{count:,}"
                    self.progress_var.set(f"Estimated {approx} files to scan.")
                    proceed = messagebox.askyesno("Confirm scan", f"This folder has {approx} files to scan.\n\nContinue?")
                    if not proceed:
                        self.status_var.set("Cancelled before start.")
                        self.progress_var.set("")
                        self._reset_buttons()
                        continue
                    args = self._pending_search_args
                    if not args:
                        self.status_var.set("Cancelled (no pending args).")
                        self.progress_var.set("")
                        self._reset_buttons()
                        continue
                    self.status_var.set("Starting…")
                    self.progress_var.set(f"Estimated {approx} files. Starting scan…")
                    try:
                        self._worker_thread = threading.Thread(
                            target=self._search_worker,
                            args=(*args, approx),
                            daemon=True,
                        )
                        self._worker_thread.start()
                    except Exception as e:
                        self.status_var.set("Error.")
                        messagebox.showerror("Error", f"Failed to start search: {e}")
                        self._reset_buttons()

                elif kind == "error":
                    self.status_var.set("Error.")
                    self.progress_var.set("")
                    messagebox.showerror("Error", msg[1])
                    self._reset_buttons()

                elif kind == "done":
                    scanned, match_count, out_path, stopped, limit_hit, skipped_oversize, skipped_errors = msg[1], msg[2], msg[3], msg[4], msg[5], msg[6], msg[7]
                    extra = f" Oversize skipped: {skipped_oversize:,}. Files skipped: {skipped_errors:,}."
                    if limit_hit:
                        extra = " Hit file limit." + extra
                    if stopped:
                        summary = f"Stopped. Scanned {scanned:,} files. Matches: {match_count:,}. Saved: {out_path}.{extra}"
                        self.status_var.set(summary)
                        messagebox.showinfo("Search stopped", summary)
                    else:
                        summary = f"Done. Scanned {scanned:,} files. Matches: {match_count:,}. Saved: {out_path}.{extra}"
                        self.status_var.set(summary)
                        messagebox.showinfo("Search complete", summary)
                    self.progress_var.set("")
                    self._reset_buttons()

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _reset_buttons(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._pending_search_args = None
        self._estimate_thread = None


def main() -> int:
    if not HAVE_TK:
        print("Error: tkinter is not available. Please install it.")
        return 1
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
