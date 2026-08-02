#!/usr/bin/env python3
"""Read-only discovery and validation for cameras connected to NanoPi LAN ports.

This first-stage manager deliberately does not modify NetworkManager profiles,
interface addresses, camera addresses, or the ALPR pipeline.  It discovers a
camera on one physical interface, validates the RTSP endpoint, and can
optionally write the discovered runtime metadata to a JSON file.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ONVIF_MULTICAST_ADDRESS = ("239.255.255.250", 3702)
INTERFACE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_interface_name(interface: str) -> str:
    if not INTERFACE_PATTERN.fullmatch(interface):
        raise ValueError(f"Invalid network interface name: {interface!r}")
    return interface


def run_ip_json(arguments: list[str]) -> list[dict[str, Any]]:
    command = ["ip", "-j", *arguments]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The Linux 'ip' command is not available") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"{' '.join(command)} failed: {error}") from exc

    try:
        return json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned by {' '.join(command)}") from exc


def read_carrier(interface: str) -> bool:
    carrier_path = Path("/sys/class/net") / interface / "carrier"
    try:
        return carrier_path.read_text(encoding="ascii").strip() == "1"
    except OSError:
        return False


def get_interface_state(interface: str) -> dict[str, Any]:
    link_rows = run_ip_json(["link", "show", "dev", interface])
    address_rows = run_ip_json(["-4", "address", "show", "dev", interface])

    if not link_rows:
        raise RuntimeError(f"Network interface {interface!r} was not found")

    link = link_rows[0]
    addresses: list[dict[str, Any]] = []
    if address_rows:
        for item in address_rows[0].get("addr_info", []):
            if item.get("family") != "inet":
                continue
            local = item.get("local")
            prefixlen = item.get("prefixlen")
            if local is not None and prefixlen is not None:
                addresses.append(
                    {
                        "address": local,
                        "prefix_length": prefixlen,
                        "cidr": f"{local}/{prefixlen}",
                    }
                )

    return {
        "name": interface,
        "mac_address": link.get("address"),
        "operstate": link.get("operstate"),
        "carrier": read_carrier(interface),
        "addresses": addresses,
    }


def local_ipv4_addresses(interface_state: dict[str, Any]) -> list[str]:
    return [entry["address"] for entry in interface_state["addresses"]]


def source_address_for(
    target_ip: str,
    interface_state: dict[str, Any],
) -> str | None:
    target = ipaddress.ip_address(target_ip)
    for entry in interface_state["addresses"]:
        network = ipaddress.ip_interface(entry["cidr"]).network
        if target in network:
            return entry["address"]
    return None


def ping_expected_ip(interface: str, expected_ip: str | None) -> dict[str, Any]:
    if not expected_ip:
        return {"attempted": False, "reachable": None}

    command = [
        "ping",
        "-c",
        "1",
        "-W",
        "1",
        "-I",
        interface,
        expected_ip,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return {
            "attempted": True,
            "reachable": completed.returncode == 0,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "attempted": True,
            "reachable": False,
            "warning": str(exc),
        }


def read_neighbors(interface: str) -> list[dict[str, Any]]:
    rows = run_ip_json(["neigh", "show", "dev", interface])
    neighbors: list[dict[str, Any]] = []
    for row in rows:
        ip_value = row.get("dst")
        if not ip_value:
            continue
        try:
            if ipaddress.ip_address(ip_value).version != 4:
                continue
        except ValueError:
            continue

        state = row.get("state")
        if isinstance(state, list):
            state = ",".join(state)
        neighbors.append(
            {
                "ip": ip_value,
                "mac_address": row.get("lladdr"),
                "state": state,
            }
        )
    return neighbors


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_onvif_response(payload: bytes, sender_ip: str) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return None

    xaddrs: list[str] = []
    scopes: list[str] = []
    endpoint_reference = None

    for element in root.iter():
        name = local_name(element.tag)
        text = (element.text or "").strip()
        if not text:
            continue
        if name == "XAddrs":
            xaddrs.extend(text.split())
        elif name == "Scopes":
            scopes.extend(text.split())
        elif name == "Address" and endpoint_reference is None:
            endpoint_reference = text

    if not xaddrs and not scopes and not endpoint_reference:
        return None

    discovered_ips = {sender_ip}
    for address in xaddrs:
        parsed = urlparse(address)
        if parsed.hostname:
            try:
                discovered_ips.add(str(ipaddress.ip_address(parsed.hostname)))
            except ValueError:
                pass

    return {
        "sender_ip": sender_ip,
        "discovered_ips": sorted(discovered_ips),
        "xaddrs": sorted(set(xaddrs)),
        "scopes": sorted(set(scopes)),
        "endpoint_reference": endpoint_reference,
    }


def build_onvif_probe() -> bytes:
    message_id = f"uuid:{uuid.uuid4()}"
    message = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>"""
    return message.encode("utf-8")


def discover_onvif(
    interface_state: dict[str, Any],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    responses: dict[tuple[str, str | None], dict[str, Any]] = {}
    warnings: list[str] = []
    probe = build_onvif_probe()

    for local_ip in local_ipv4_addresses(interface_state):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(local_ip),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.bind((local_ip, 0))
            sock.settimeout(0.25)
            sock.sendto(probe, ONVIF_MULTICAST_ADDRESS)

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                try:
                    payload, sender = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                parsed = parse_onvif_response(payload, sender[0])
                if parsed is None:
                    continue
                key = (parsed["sender_ip"], parsed["endpoint_reference"])
                responses[key] = parsed
        except OSError as exc:
            warnings.append(f"ONVIF discovery from {local_ip} failed: {exc}")
        finally:
            sock.close()

    return list(responses.values()), warnings


def probe_rtsp(
    target_ip: str,
    source_ip: str | None,
    port: int,
    path: str,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "port": port,
        "path": path,
        "source_ip": source_ip,
        "tcp_reachable": False,
        "rtsp_status": None,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    try:
        if source_ip:
            sock.bind((source_ip, 0))
        sock.connect((target_ip, port))
        result["tcp_reachable"] = True
        request = (
            f"OPTIONS rtsp://{target_ip}:{port}{path} RTSP/1.0\r\n"
            "CSeq: 1\r\n"
            "User-Agent: NanoPi-Camera-Manager/1.0\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        try:
            response = sock.recv(4096).decode("iso-8859-1", errors="replace")
            first_line = response.splitlines()[0] if response else None
            if first_line and first_line.startswith("RTSP/"):
                result["rtsp_status"] = first_line
        except socket.timeout:
            result["warning"] = "RTSP port opened but no OPTIONS response was received"
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        sock.close()
    return result


def merge_candidates(
    expected_ip: str | None,
    ping_result: dict[str, Any],
    neighbors: list[dict[str, Any]],
    onvif_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    def candidate(ip_value: str) -> dict[str, Any]:
        return candidates.setdefault(
            ip_value,
            {
                "ip": ip_value,
                "mac_address": None,
                "sources": [],
                "onvif": [],
            },
        )

    if expected_ip and ping_result.get("reachable"):
        candidate(expected_ip)["sources"].append("expected_ip")

    for neighbor in neighbors:
        item = candidate(neighbor["ip"])
        item["mac_address"] = neighbor.get("mac_address")
        item["neighbor_state"] = neighbor.get("state")
        item["sources"].append("neighbor")

    for response in onvif_results:
        for discovered_ip in response["discovered_ips"]:
            item = candidate(discovered_ip)
            item["sources"].append("onvif")
            item["onvif"].append(response)

    for item in candidates.values():
        item["sources"] = sorted(set(item["sources"]))

    return candidates


def candidate_score(candidate: dict[str, Any], expected_ip: str | None) -> int:
    score = 0
    if candidate["ip"] == expected_ip:
        score += 100
    if candidate.get("rtsp", {}).get("tcp_reachable"):
        score += 50
    if "onvif" in candidate["sources"]:
        score += 20
    if candidate.get("mac_address"):
        score += 10
    return score


def update_runtime_config(
    config_path: Path,
    role: str,
    interface: str,
    selected: dict[str, Any],
    rtsp_port: int,
    rtsp_path: str,
) -> None:
    config: dict[str, Any] = {"version": 1, "cameras": {}}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            config.update(existing)
            config.setdefault("cameras", {})

    config["updated_at"] = utc_now()
    config["cameras"][role] = {
        "interface": interface,
        "camera_ip": selected["ip"],
        "mac_address": selected.get("mac_address"),
        "rtsp_port": rtsp_port,
        "rtsp_path": rtsp_path,
        "discovered_at": utc_now(),
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, config_path)


def discover(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    interface = validate_interface_name(args.interface)
    expected_ip = None
    if args.expected_ip:
        expected_ip = str(ipaddress.IPv4Address(args.expected_ip))

    interface_state = get_interface_state(interface)
    report: dict[str, Any] = {
        "mode": "discover-only",
        "changes_applied": False,
        "generated_at": utc_now(),
        "role": args.role,
        "interface": interface_state,
        "expected_ip": expected_ip,
        "ping": None,
        "onvif_warnings": [],
        "candidates": [],
        "selected_camera": None,
    }

    if not interface_state["carrier"]:
        report["error"] = f"No Ethernet carrier detected on {interface}"
        return report, 2
    if not interface_state["addresses"]:
        report["error"] = f"No IPv4 address is active on {interface}"
        return report, 2

    ping_result = ping_expected_ip(interface, expected_ip)
    report["ping"] = ping_result

    onvif_results, warnings = discover_onvif(interface_state, args.timeout)
    report["onvif_warnings"] = warnings
    neighbors = read_neighbors(interface)
    candidates = merge_candidates(
        expected_ip,
        ping_result,
        neighbors,
        onvif_results,
    )

    for item in candidates.values():
        source_ip = source_address_for(item["ip"], interface_state)
        item["rtsp"] = probe_rtsp(
            item["ip"],
            source_ip,
            args.rtsp_port,
            args.rtsp_path,
        )

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: candidate_score(item, expected_ip),
        reverse=True,
    )
    report["candidates"] = ordered_candidates

    selected = next(
        (
            item
            for item in ordered_candidates
            if item.get("rtsp", {}).get("tcp_reachable")
        ),
        None,
    )
    if selected is None:
        report["error"] = "No candidate with a reachable RTSP port was found"
        return report, 3

    report["selected_camera"] = selected
    if args.write_config:
        config_path = Path(args.write_config)
        update_runtime_config(
            config_path,
            args.role,
            interface,
            selected,
            args.rtsp_port,
            args.rtsp_path,
        )
        report["runtime_config_written"] = str(config_path)

    return report, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and validate one camera on a dedicated NanoPi LAN port. "
            "This version never changes network settings or camera settings."
        )
    )
    parser.add_argument("--role", choices=("front", "rear"), required=True)
    parser.add_argument("--interface", required=True, help="For example: eth1")
    parser.add_argument(
        "--expected-ip",
        help="Optional current camera IP used to prime neighbor discovery.",
    )
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--rtsp-path", default="/stream1")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="ONVIF discovery timeout per active interface address.",
    )
    parser.add_argument(
        "--write-config",
        help=(
            "Optional JSON output path. Omit this option for a completely "
            "read-only discovery test."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.rtsp_port <= 65535:
        parser.error("--rtsp-port must be between 1 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not args.rtsp_path.startswith("/"):
        parser.error("--rtsp-path must start with '/'")

    try:
        report, exit_code = discover(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "mode": "discover-only",
            "changes_applied": False,
            "generated_at": utc_now(),
            "error": str(exc),
        }
        exit_code = 1

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
