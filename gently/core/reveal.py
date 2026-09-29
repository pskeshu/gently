"""Showing the operator a file: in the file manager, or in Fiji.

Gently keeps everything as files a person can browse, and until now the way
to one was to know the layout and walk to it. This is the part that acts on
the operating system. What to show is decided elsewhere
(``gently/ui/web/routes/reveal.py``), from the store, never from a path a
browser sent.

Everything here happens on the machine the backend runs on. Under the desktop
shell that is the operator's own screen. From a browser on another computer
it is somebody else's, which is why the route asks ``is_local`` first.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Micro-Manager ships an ImageJ of its own, and starting it starts
# Micro-Manager, which loads the hardware configuration and takes the
# microscope's ports from under the device layer. It is never found, and it
# is refused when it is configured.
_NEVER = ("micro-manager", "micromanager")

_WINDOWS_EXES = ("fiji-windows-x64.exe", "ImageJ-win64.exe", "fiji.exe")
_LINUX_EXES = ("fiji-linux-x64", "ImageJ-linux64", "fiji")
_MAC_EXES = ("fiji-macos", "ImageJ-macosx", "fiji-macosx")
_FOLDERS = ("Fiji", "Fiji.app")


def file_manager_name() -> str:
    """What the operator calls the thing a folder opens in."""
    if sys.platform.startswith("win"):
        return "Explorer"
    if sys.platform == "darwin":
        return "Finder"
    return "the file manager"


def open_folder(path: Path) -> None:
    """Open a folder in the file manager."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])  # noqa: S603, S607
    else:
        subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607


def select_file(path: Path) -> None:
    """Open the folder a file is in, with the file selected where the file
    manager can do that."""
    if sys.platform.startswith("win"):
        # One argument, comma and all: Explorer parses its own command line.
        subprocess.Popen(["explorer", f"/select,{path}"])  # noqa: S603, S607
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])  # noqa: S603, S607
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])  # noqa: S603, S607


def show(path: Path) -> str:
    """Show a folder, or a file in its folder. Returns which it was."""
    if path.is_dir():
        open_folder(path)
        return "folder"
    select_file(path)
    return "file"


def is_micro_manager(path: Path | str) -> bool:
    low = str(path).replace("\\", "/").lower()
    return any(word in low for word in _NEVER)


def _candidates() -> list[Path]:
    home = Path.home()
    out: list[Path] = []
    if sys.platform.startswith("win"):
        bases = [
            home / "Documents",
            home,
            home / "Desktop",
            home / "Downloads",
            home / "AppData" / "Local",
            Path("C:/"),
            Path("D:/"),
            Path(os.environ.get("ProgramFiles", "C:/Program Files")),
        ]
        out += [b / f / e for b in bases for f in _FOLDERS for e in _WINDOWS_EXES]
    elif sys.platform == "darwin":
        for base in (Path("/Applications"), home / "Applications"):
            out += [base / "Fiji.app" / "Contents" / "MacOS" / e for e in _MAC_EXES]
    else:
        bases = [home, home / "Applications", Path("/opt"), Path("/usr/local")]
        out += [b / f / e for b in bases for f in _FOLDERS for e in _LINUX_EXES]
        for name in ("fiji", "Fiji", "ImageJ", "imagej"):
            found = shutil.which(name)
            if found:
                out.append(Path(found))
    return out


def find_fiji(configured: str | None = None) -> Path | None:
    """Where Fiji is: the configured path if there is one, else the places it
    is usually unpacked. ``None`` if it is in none of them.

    A configured path that is not a file is not quietly replaced by a found
    one: the operator said where it is, and should hear that it is not there.
    """
    if configured and str(configured).strip():
        path = Path(str(configured).strip().strip('"')).expanduser()
        if path.is_dir():
            inside = [path / e for e in (*_WINDOWS_EXES, *_LINUX_EXES)]
            inside += [path / "Contents" / "MacOS" / e for e in _MAC_EXES]
            path = next((p for p in inside if p.is_file()), path)
        if is_micro_manager(path) or not path.is_file():
            return None
        return path
    for path in _candidates():
        try:
            if path.is_file() and not is_micro_manager(path):
                return path
        except OSError:
            continue
    return None


def open_in_fiji(path: Path, fiji: Path) -> None:
    """Open a file in Fiji. If Fiji is running it is handed the file; if not
    it starts. Either way this returns at once, and Fiji outlives Gently."""
    if is_micro_manager(fiji):
        raise ValueError("that is Micro-Manager's ImageJ, which would start Micro-Manager")
    kwargs: dict = {"cwd": str(fiji.parent), "close_fds": True}
    if sys.platform.startswith("win"):
        # Out of Gently's job object and console, so closing Gently does not
        # close the image the operator is looking at.
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([str(fiji), str(path)], **kwargs)  # noqa: S603
    except OSError:
        # A job object that does not allow breakaway refuses the flag.
        if "creationflags" not in kwargs:
            raise
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen([str(fiji), str(path)], **kwargs)  # noqa: S603


_LOOPBACK = ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")


def _own_addresses() -> set[str]:
    own: set[str] = set(_LOOPBACK)
    try:
        name = socket.gethostname()
        own.update(str(info[4][0]) for info in socket.getaddrinfo(name, None))
    except OSError:
        pass
    return {str(a) for a in own}


def is_local(host: str | None) -> bool:
    """Is the request from the machine the backend runs on?

    A window opened for a request from another computer opens on the
    microscope's screen, in front of whoever is sitting there, and shows the
    person who asked nothing.
    """
    if not host:
        return False
    if host in _LOOPBACK:
        return True
    return host in _own_addresses()
