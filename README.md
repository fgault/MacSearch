# Folder String Finder

Small GUI tool to recursively search folder filenames and file contents.

Features
- Choose a folder and enter a search string (optional outer quotes are treated as literal)
- Filename search and content search for supported types (plain text, CSV/TSV, PDF, DOCX, XLSX)
- Options: skip hidden files/folders, max file size (default 50 MB), max files (default 200,000, 0 = unlimited), max depth (blank = unlimited)
- Early file-count estimate with warning before large scans
- Writes `matches.txt` into the scanned folder with matched full paths

Dependencies
- Python 3.8+
- Optional Python packages (see `requirements.txt`): `pypdf`, `python-docx`, `openpyxl`
- Optional system dependency for faster PDF scanning: `pdftotext` (part of Poppler)

Quick start
1. Create and activate a virtualenv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the GUI:

```bash
python gui_folder_string_finder.py
```

Notes
- If you want to push this repo to GitHub, add a remote and push the branch you create.
