"""Load camera endpoints from runtime JSON and credentials from a private file."""

from __future__ import annotations

import ipaddress
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit


EXPECTED_INTERFACES = {
    "front": "eth1",
    "rear": "eth2",
}


class CameraConfigError(RuntimeError):
    """Raised when runtime camera configuration is missing or invalid."""


@dataclass(frozen=True)
class CameraEndpoint:
    role: str
    interface: str
    camera_ip: str
    mac_address: str | None
    rtsp_port: int
    rtsp_path: str
    rtsp_url: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise CameraConfigError(f"Camera config was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CameraConfigError(f"Camera config is not valid JSON: {path}") from exc
    except OSError as exc:
        raise CameraConfigError(f"Camera config cannot be read: {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise CameraConfigError("Camera config root must be a JSON object")
    return value


def _check_secret_permissions(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise CameraConfigError(f"Cannot inspect credentials file: {path}: {exc}") from exc

    if mode & 0o077:
        raise CameraConfigError(
            f"Credentials file permissions are too open ({mode:04o}); "
            f"run: chmod 600 {path}"
        )


def _load_secrets(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise CameraConfigError(f"Camera credentials file was not found: {path}")
    _check_secret_permissions(path)

    secrets: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CameraConfigError(
            f"Camera credentials file cannot be read: {path}: {exc}"
        ) from exc

    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise CameraConfigError(
                f"Invalid credentials line {line_number} in {path}"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            raise CameraConfigError(
                f"Empty credentials key on line {line_number} in {path}"
            )
        secrets[key] = value

    return secrets


def _role_credentials(role: str, secrets: dict[str, str]) -> tuple[str, str]:
    prefix = role.upper()
    username = secrets.get(f"{prefix}_CAMERA_USERNAME")
    password = secrets.get(f"{prefix}_CAMERA_PASSWORD")

    if username is None:
        username = secrets.get("CAMERA_USERNAME")
    if password is None:
        password = secrets.get("CAMERA_PASSWORD")

    if not username:
        raise CameraConfigError(
            f"No username is configured for the {role} camera"
        )
    if password is None:
        raise CameraConfigError(
            f"No password is configured for the {role} camera"
        )
    return username, password


def _validate_camera_entry(role: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise CameraConfigError(f"Camera entry {role!r} must be a JSON object")

    interface = entry.get("interface")
    expected_interface = EXPECTED_INTERFACES[role]
    if interface != expected_interface:
        raise CameraConfigError(
            f"{role} camera must use {expected_interface}, got {interface!r}"
        )

    try:
        camera_ip = str(ipaddress.IPv4Address(entry.get("camera_ip")))
    except ValueError as exc:
        raise CameraConfigError(
            f"Invalid camera_ip for the {role} camera"
        ) from exc

    rtsp_port = entry.get("rtsp_port", 554)
    if isinstance(rtsp_port, bool) or not isinstance(rtsp_port, int):
        raise CameraConfigError(f"Invalid rtsp_port for the {role} camera")
    if not 1 <= rtsp_port <= 65535:
        raise CameraConfigError(f"Invalid rtsp_port for the {role} camera")

    rtsp_path = entry.get("rtsp_path", "/stream1")
    if not isinstance(rtsp_path, str) or not rtsp_path.startswith("/"):
        raise CameraConfigError(f"Invalid rtsp_path for the {role} camera")

    mac_address = entry.get("mac_address")
    if mac_address is not None and not isinstance(mac_address, str):
        raise CameraConfigError(f"Invalid mac_address for the {role} camera")

    return {
        "interface": interface,
        "camera_ip": camera_ip,
        "mac_address": mac_address,
        "rtsp_port": rtsp_port,
        "rtsp_path": rtsp_path,
    }


def load_camera_endpoints(
    config_path: str | os.PathLike[str],
    secrets_path: str | os.PathLike[str],
) -> dict[str, CameraEndpoint]:
    """Return configured camera endpoints keyed by physical role."""
    config_file = Path(config_path)
    secrets_file = Path(secrets_path)
    config = _load_json(config_file)

    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise CameraConfigError("Camera config does not contain any cameras")

    secrets = _load_secrets(secrets_file)
    endpoints: dict[str, CameraEndpoint] = {}

    for role in ("front", "rear"):
        if role not in cameras:
            continue
        values = _validate_camera_entry(role, cameras[role])
        username, password = _role_credentials(role, secrets)
        encoded_username = quote(username, safe="")
        encoded_password = quote(password, safe="")
        rtsp_url = (
            f"rtsp://{encoded_username}:{encoded_password}"
            f"@{values['camera_ip']}:{values['rtsp_port']}{values['rtsp_path']}"
        )
        endpoints[role] = CameraEndpoint(
            role=role,
            interface=values["interface"],
            camera_ip=values["camera_ip"],
            mac_address=values["mac_address"],
            rtsp_port=values["rtsp_port"],
            rtsp_path=values["rtsp_path"],
            rtsp_url=rtsp_url,
        )

    if not endpoints:
        raise CameraConfigError("Camera config contains no supported camera roles")
    return endpoints


def redact_rtsp_url(url: str) -> str:
    """Remove credentials from an RTSP URL before it is written to logs."""
    parsed = urlsplit(url)
    if parsed.username is None:
        return url

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, ""))
