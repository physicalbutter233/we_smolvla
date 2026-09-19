#!/usr/bin/env python3
"""Safely disable torque on all six SO-101 follower motors.

The script writes Torque_Enable=0 first, then Lock=0. It verifies the torque
state by reading the register back and retries motors that remain enabled.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from serial.tools import list_ports

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


SO101_FOLLOWER_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# Disable distal joints first so the proximal joints continue holding the arm
# while the lighter end joints are released.
DISABLE_ORDER = (
    "gripper",
    "wrist_roll",
    "wrist_flex",
    "elbow_flex",
    "shoulder_lift",
    "shoulder_pan",
)


def _detect_port() -> str | None:
    candidates = sorted(set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")))
    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        print("Multiple serial devices detected. Re-run with --port:")
        for candidate in candidates:
            print(f"  {candidate}")
        return None

    ports = sorted(port.device for port in list_ports.comports())
    if len(ports) == 1:
        return ports[0]

    if ports:
        print("Serial devices detected. Re-run with --port:")
        for port in ports:
            print(f"  {port}")
    else:
        print("No serial device detected.")
        print("Expected a device such as /dev/ttyACM0 or /dev/ttyUSB0.")

    return None


def _read_register(
    bus: FeetechMotorsBus,
    motor: str,
    register: str,
    retries: int,
) -> tuple[int | None, str | None]:
    try:
        value = bus.read(register, motor, normalize=False, num_retry=retries)
        return int(value), None
    except Exception as exc:  # noqa: BLE001 - report the exact communication failure
        return None, f"{type(exc).__name__}: {exc}"


def _write_register(
    bus: FeetechMotorsBus,
    motor: str,
    register: str,
    value: int,
    retries: int,
) -> str | None:
    try:
        bus.write(register, motor, value, normalize=False, num_retry=retries)
        return None
    except Exception as exc:  # noqa: BLE001 - continue disabling the remaining motors
        return f"{type(exc).__name__}: {exc}"


def _read_torque_states(
    bus: FeetechMotorsBus,
    retries: int,
) -> tuple[dict[str, int | None], dict[str, str]]:
    states = {name: None for name in SO101_FOLLOWER_MOTORS}
    errors: dict[str, str] = {}

    for name in SO101_FOLLOWER_MOTORS:
        value, error = _read_register(bus, name, "Torque_Enable", retries)
        states[name] = value
        if error is not None:
            errors[name] = error

    return states, errors


def _wait_for_operator(delay_s: float) -> None:
    if delay_s <= 0:
        return

    print(f"Disabling torque in {delay_s:.1f} seconds.")
    remaining = delay_s
    while remaining > 0:
        interval = min(0.5, remaining)
        time.sleep(interval)
        remaining -= interval
        if remaining > 0:
            print(f"  {remaining:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely disable torque on all six SO-101 follower motors."
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port, for example /dev/ttyACM0. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries after the first write or read attempt. Default: 5.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Whole-arm verification passes. Default: 3.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait after confirmation before disabling. Default: 3.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation. Use only when the arm is supported.",
    )
    args = parser.parse_args()

    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    port = args.port or _detect_port()
    if not port:
        return 2

    if not Path(port).exists():
        print(f"Serial port does not exist: {port}")
        return 2

    if not args.yes:
        print()
        print("WARNING: disabling torque can make the arm fall under gravity.")
        print("Support the arm or place it in a stable rest position.")
        answer = input("Type DISABLE to continue: ").strip()
        if answer != "DISABLE":
            print("Cancelled. Torque was not changed.")
            return 2

    _wait_for_operator(args.delay)

    print(f"Port: {port}")
    print("Disabling torque...")

    bus = FeetechMotorsBus(port=port, motors=SO101_FOLLOWER_MOTORS)
    try:
        bus.connect(handshake=False)
    except Exception as exc:  # noqa: BLE001 - report the device error directly
        print(f"Failed to open serial port: {type(exc).__name__}: {exc}")
        return 2

    final_states: dict[str, int | None] = {}
    final_errors: dict[str, str] = {}

    try:
        bus.set_timeout(3000)

        positions_before: dict[str, int | None] = {}
        for name in SO101_FOLLOWER_MOTORS:
            value, error = _read_register(bus, name, "Present_Position", args.retries)
            positions_before[name] = value
            if error is not None:
                print(f"Warning: could not read position for {name}: {error}")

        print("Current positions:", positions_before)

        for pass_index in range(1, args.passes + 1):
            print(f"Disable pass {pass_index}/{args.passes}")

            for name in DISABLE_ORDER:
                torque_error = _write_register(
                    bus, name, "Torque_Enable", 0, args.retries
                )
                if torque_error is not None:
                    print(f"  {name}: Torque_Enable write failed: {torque_error}")
                else:
                    print(f"  {name}: Torque_Enable=0 written")

                lock_error = _write_register(bus, name, "Lock", 0, args.retries)
                if lock_error is not None:
                    print(f"  {name}: Lock write failed: {lock_error}")
                else:
                    print(f"  {name}: Lock=0 written")

            time.sleep(0.2)
            final_states, final_errors = _read_torque_states(bus, args.retries)

            if all(value == 0 for value in final_states.values()):
                break

            print("  Some motors remain enabled; retrying the whole arm.")

    finally:
        bus.disconnect(disable_torque=False)

    print()
    print("Final Torque_Enable states:")
    for name in SO101_FOLLOWER_MOTORS:
        value = final_states.get(name)
        error = final_errors.get(name)
        if error is not None:
            print(f"  {name}: READ FAILED ({error})")
        else:
            print(f"  {name}: {value}")

    all_disabled = all(value == 0 for value in final_states.values())
    if all_disabled:
        print("Result: all six motors are disabled.")
        return 0

    print("Result: not all motors could be confirmed disabled.")
    print("Check power and wiring, then run the script again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
