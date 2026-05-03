from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


Signal = dict[str, Any]


class MatlabSereegaBridge:
    """Small MATLAB Engine bridge for original SEREEGA signal generation."""

    def __init__(self, sereega_path: Path) -> None:
        if not sereega_path.exists():
            raise RuntimeError(f"SEREEGA MATLAB path does not exist: {sereega_path}")
        try:
            import matlab.engine  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB Engine for Python is required. Install it from your MATLAB "
                "installation, then rerun generation."
            ) from exc

        self._engine = matlab.engine.start_matlab()
        self._engine.addpath(self._engine.genpath(str(sereega_path)), nargout=0)

    def generate_signal(
        self,
        signal_class: Signal,
        T: int,
        sfreq: float,
    ) -> np.ndarray:
        try:
            self._engine.eval(_signal_code(signal_class, T, sfreq), nargout=0)
            signal = np.asarray(self._engine.workspace["synthgen_signal"], dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"MATLAB/SEREEGA failed while generating {signal_class['type']!r}. "
                "Check matlab_sereega_path and the original SEREEGA toolbox."
            ) from exc

        signal = signal.reshape(-1)
        if signal.size != T:
            raise RuntimeError(
                f"MATLAB/SEREEGA returned {signal.size} samples, expected {T}."
            )
        return signal.astype(np.float32)


_GENERATORS = {
    "erp": ("utl_check_class", "erp_generate_signal_fromclass"),
    "ersp": ("utl_check_class", "ersp_generate_signal_fromclass"),
    "noise": ("utl_check_class", "noise_generate_signal_fromclass"),
    "arm": ("arm_check_class", "arm_generate_signal_fromclass"),
}


def _matlab_value(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + " ".join(_matlab_value(v).strip("'") for v in value) + "]"
    return f"{float(value):.17g}"


def _matlab_struct(fields: dict[str, Any]) -> str:
    args = []
    for key, value in fields.items():
        args.extend([_matlab_value(key), _matlab_value(value)])
    return "struct(" + ", ".join(args) + ")"


def _signal_code(signal_class: Signal, T: int, sfreq: float) -> str:
    check_fn, generate_fn = _GENERATORS[signal_class["type"]]
    epoch_ms = 1000.0 * T / sfreq
    return "\n".join(
        [
            f"rng({int(signal_class['params'].get('seed', 0))}, 'twister');",
            (
                "synthgen_epochs = struct('n', 1, "
                f"'srate', {_matlab_value(sfreq)}, "
                f"'length', {_matlab_value(epoch_ms)}, 'prestim', 0);"
            ),
            "try, synthgen_epochs = utl_check_epochs(synthgen_epochs); catch, end;",
            f"synthgen_class = {_matlab_struct(_class_fields(signal_class))};",
            f"synthgen_class = {check_fn}(synthgen_class);",
            f"synthgen_signal = {generate_fn}(synthgen_class, synthgen_epochs, 1);",
        ]
    )


def _class_fields(signal_class: Signal) -> dict[str, Any]:
    p = signal_class["params"]
    if signal_class["type"] == "erp":
        return {
            "type": "erp",
            "peakLatency": p["peak_latencies_ms"],
            "peakWidth": p["peak_widths_ms"],
            "peakAmplitude": p["peak_amplitudes"],
        }
    if signal_class["type"] == "ersp":
        return {
            "type": "ersp",
            "frequency": p["frequency_hz"],
            "amplitude": p["amplitude"],
            "phase": p["phase_cycles"],
            "modulation": "burst",
            "modLatency": p["mod_latency_ms"],
            "modWidth": p["mod_width_ms"],
            "modTaper": p["mod_taper"],
        }
    if signal_class["type"] == "noise":
        return {"type": "noise", "color": p["color"], "amplitude": p["amplitude"]}
    if signal_class["type"] == "arm":
        return {"order": p["order"], "amplitude": p["amplitude"]}
    raise ValueError(f"Unknown SEREEGA signal class: {signal_class['type']!r}")
