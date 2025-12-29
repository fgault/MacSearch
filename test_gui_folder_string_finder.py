import os
from pathlib import Path
import tempfile
from contextlib import contextmanager

import gui_folder_string_finder as gfs


@contextmanager
def home_tempdir():
    base = Path(__file__).resolve().parent
    tmp = tempfile.mkdtemp(dir=base)
    try:
        yield Path(tmp)
    finally:
        try:
            for root, dirs, files in os.walk(tmp, topdown=False):
                for f in files:
                    Path(root, f).unlink(missing_ok=True)
                for d in dirs:
                    Path(root, d).rmdir()
            Path(tmp).rmdir()
        except Exception:
            pass


def write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_user_needle_strips_optional_quotes():
    assert gfs.parse_user_needle('"hello"') == "hello"
    assert gfs.parse_user_needle("hello") == "hello"
    assert gfs.parse_user_needle('  " spaced "  ') == " spaced "


def test_iter_files_skip_hidden_and_depth():
    with home_tempdir() as root:
        write_file(root / "visible.txt", "a")
        write_file(root / ".hidden" / "secret.txt", "b")
        write_file(root / "sub" / "deep.txt", "c")
        write_file(root / "sub" / "child" / "deeper.txt", "d")
        write_file(root / "Library" / "sys.txt", "x")

        all_files = {p.name for p in gfs.iter_files(root, skip_hidden=False, max_depth=None)}
        assert "visible.txt" in all_files
        assert "secret.txt" in all_files
        assert "deep.txt" in all_files

        hidden_skipped = {p.name for p in gfs.iter_files(root, skip_hidden=True, max_depth=None)}
        assert "secret.txt" not in hidden_skipped

        depth_limited = {p.name for p in gfs.iter_files(root, skip_hidden=True, max_depth=1)}
        assert "deep.txt" in depth_limited  # depth 1 allowed
        assert "deeper.txt" not in depth_limited  # depth 2 pruned

        system_skipped = {p.name for p in gfs.iter_files(root, skip_hidden=True, max_depth=None, skip_system=True)}
        assert "sys.txt" not in system_skipped


def test_search_folder_matches_filename_and_content():
    with home_tempdir() as root:
        write_file(root / "foo_match.txt", "nothing")
        write_file(root / "other.txt", "needle inside content")

        scanned, matches = gfs.search_folder(
            root,
            needle="needle",
            case_sensitive=True,
            skip_hidden=True,
            file_size_cap_bytes=gfs.FILE_SIZE_CAP_BYTES,
            max_files=None,
            max_depth=None,
        )

        assert scanned >= 2
        assert str(root / "foo_match.txt") in matches or str(root / "other.txt") in matches
        assert any("needle" in Path(m).read_text(encoding="utf-8") for m in matches)


def test_search_folder_honors_file_size_cap():
    with home_tempdir() as root:
        small = write_file(root / "small.txt", "needle here")
        large = write_file(root / "large.txt", "needle here too but should be skipped")
        # Make large file exceed cap
        large.write_bytes(b"x" * 10_000)

        scanned, matches = gfs.search_folder(
            root,
            needle="needle",
            case_sensitive=True,
            skip_hidden=True,
            file_size_cap_bytes=50,  # bytes; smaller than large file
            max_files=None,
            max_depth=None,
        )

        assert scanned >= 1
        assert str(small) in matches
        assert str(large) not in matches


def test_search_folder_honors_max_files():
    with home_tempdir() as root:
        for i in range(5):
            write_file(root / f"file{i}.txt", "needle")

        scanned, matches = gfs.search_folder(
            root,
            needle="needle",
            case_sensitive=True,
            skip_hidden=True,
            skip_system=True,
            file_size_cap_bytes=gfs.FILE_SIZE_CAP_BYTES,
            max_files=2,
            max_depth=None,
        )

        assert scanned == 2  # should stop after reaching the limit
        assert len(matches) <= 2

