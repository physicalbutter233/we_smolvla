#!/usr/bin/env python3
"""Read-only diagnostic for the SO-101 follower arm.

The script never enables torque and never writes a goal position. It checks bus
communication, model numbers, feedback registers, voltage, temperature and
motor status flags for motor IDs 1 through 6.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import scservo_sdk as scs
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

EXPECTED_MODEL_NUMBER = 777

CORE_REGISTERS = (
    "Present_Position",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
)

OPTIONAL_REGISTERS = (
    "Present_Velocity",
    "Present_Load",
    "Torque_Enable",
    "Moving",
    "Present_Current",
)

LIMIT_REGISTERS = (
    "Min_Voltage_Limit",
    "Max_Voltage_Limit",
    "Max_Temperature_Limit",
)

STATUS_BITS = {
    scs.ERRBIT_VOLTAGE: "voltage",
    scs.ERRBIT_ANGLE: "angle",
    scs.ERRBIT_OVERHEAT: "overheat",
    scs.ERRBIT_OVERELE: "over-current",
    scs.ERRBIT_OVERLOAD: "overload",
}


@dataclass
class MotorResult:
    name: str
    motor_id: int
    ping_successes: int
    model_number: int | None
    values: dict[str, int]
    read_errors: dict[str, str]
    limit_errors: list[str]
    status_errors: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.ping_successes > 0
            and self.model_number == EXPECTED_MODEL_NUMBER
            and not any(register in self.read_errors for register in CORE_REGISTERS)
            and not self.limit_errors
            and not self.status_errors
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
    motor_name: str,
    register: str,
    retries: int,
) -> tuple[int | None, str | None]:
    try:
        value = bus.read(register, motor_name, normalize=False, num_retry=retries)
        return int(value), None
    except Exception as exc:  # noqa: BLE001 - diagnostics must retain the exact failure
        return None, f"{type(exc).__name__}: {exc}"


def _check_motor(
    bus: FeetechMotorsBus,
    name: str,
    motor: Motor,
    samples: int,
    retries: int,
) -> MotorResult:
    ping_successes = 0
    model_number = None

    for _ in range(samples):
        ping_model = bus.ping(motor.id, num_retry=retries)
        if ping_model is not None:
            ping_successes += 1
            model_number = int(ping_model)

    values: dict[str, int] = {}
    read_errors: dict[str, str] = {}

    for register in (*CORE_REGISTERS, *OPTIONAL_REGISTERS, *LIMIT_REGISTERS):
        value, error = _read_register(bus, name, register, retries)
        if error is not None:
            read_errors[register] = error
        elif value is not None:
            values[register] = value

    limit_errors: list[str] = []
    voltage = values.get("Present_Voltage")
    min_voltage = values.get("Min_Voltage_Limit")
    max_voltage = values.get("Max_Voltage_Limit")
    if voltage is not None and min_voltage is not None and voltage < min_voltage:
        limit_errors.append(
            f"voltage {voltage / 10:.1f} V is below configured minimum {min_voltage / 10:.1f} V"
        )
    if voltage is not None and max_voltage is not None and voltage > max_voltage:
        limit_errors.append(
            f"voltage {voltage / 10:.1f} V is above configured maximum {max_voltage / 10:.1f} V"
        )

    temperature = values.get("Present_Temperature")
    max_temperature = values.get("Max_Temperature_Limit")
    if temperature is not None and max_temperature is not None and temperature > max_temperature:
        limit_errors.append(
            f"temperature {temperature} C is above configured maximum {max_temperature} C"
        )

    status_errors: list[str] = []
    status = values.get("Status")
    if status is not None:
        status_errors = [
            label for bit, label in STATUS_BITS.items() if status & bit
        ]

    return MotorResult(
        name=name,
        motor_id=motor.id,
        ping_successes=ping_successes,
        model_number=model_number,
        values=values,
        read_errors=read_errors,
        limit_errors=limit_errors,
        status_errors=status_errors,
    )


def _print_result(result: MotorResult, samples: int) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    print()
    print(f"[ID {result.motor_id}] {result.name}: {verdict}")
    print(f"  ping: {result.ping_successes}/{samples}")
    print(f"  model_number: {result.model_number} (expected {EXPECTED_MODEL_NUMBER})")

    for register, value in result.values.items():
        if register == "Present_Voltage":
            print(f"  {register}: {value} ({value / 10:.1f} V)")
        elif register == "Min_Voltage_Limit":
            print(f"  {register}: {value} ({value / 10:.1f} V)")
        elif register == "Max_Voltage_Limit":
            print(f"  {register}: {value} ({value / 10:.1f} V)")
        else:
            print(f"  {register}: {value}")

    if result.read_errors:
        print("  read_errors:")
        for register, error in result.read_errors.items():
            print(f"    {register}: {error}")

    if result.status_errors:
        print(f"  status_errors: {', '.join(result.status_errors)}")

    if result.limit_errors:
        print("  limit_errors:")
        for error in result.limit_errors:
            print(f"    {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only communication and feedback test for an SO-101 follower arm."
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port, for example /dev/ttyACM0. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of ping attempts per motor. Default: 3.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Register-read retries after the first attempt. Default: 2.",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.retries < 0:
        parser.error("--retries must be non-negative")

    port = args.port or _detect_port()
    if not port:
        return 2

    if not Path(port).exists():
        print(f"Serial port does not exist: {port}")
        return 2

    print(f"Port: {port}")
    print("Mode: read-only (no torque changes and no goal-position writes)")

    bus = FeetechMotorsBus(port=port, motors=SO101_FOLLOWER_MOTORS)
    try:
        bus.connect(handshake=False)
    except Exception as exc:  # noqa: BLE001 - report the device error directly
        print(f"Failed to open serial port: {type(exc).__name__}: {exc}")
        return 2

    try:
        results = [
            _check_motor(
                bus=bus,
                name=name,
                motor=motor,
                samples=args.samples,
                retries=args.retries,
            )
            for name, motor in SO101_FOLLOWER_MOTORS.items()
        ]
    finally:
        bus.disconnect(disable_torque=False)

    for result in results:
        _print_result(result, args.samples)

    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]

    print()
    print(f"Summary: {len(passed)}/{len(results)} motors passed")
    if failed:
        print("Failed motors: " + ", ".join(f"{r.name}(ID {r.motor_id})" for r in failed))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
