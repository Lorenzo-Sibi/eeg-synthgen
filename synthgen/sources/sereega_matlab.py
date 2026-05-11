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
    "noise": ("utl_check_class", "noise_generate_signal_fromclass"),
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
    epoch_ms = 1000.0 * T / sfreq
    if signal_class["type"] == "erp":
        return _erp_signal_code(signal_class, T, sfreq)
    if signal_class["type"] == "ersp":
        return _ersp_signal_code(signal_class, T, sfreq)
    if signal_class["type"] == "arm":
        return _arm_signal_code(signal_class, T, sfreq, epoch_ms)

    check_fn, generate_fn = _GENERATORS[signal_class["type"]]
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
            f"synthgen_signal = {generate_fn}(synthgen_class, synthgen_epochs, 'baseonly', 1);",
        ]
    )


def _erp_signal_code(signal_class: Signal, T: int, sfreq: float) -> str:
    p = signal_class["params"]
    return "\n".join(
        [
            f"rng({int(p.get('seed', 0))}, 'twister');",
            f"synthgen_peak_latency = {_matlab_value(p['peak_latencies_ms'])};",
            f"synthgen_peak_width = {_matlab_value(p['peak_widths_ms'])};",
            f"synthgen_peak_amplitude = {_matlab_value(p['peak_amplitudes'])};",
            f"synthgen_signal = zeros(1, {int(T)});",
            "for synthgen_i = 1:length(synthgen_peak_latency)",
            f"    synthgen_latency = (synthgen_peak_latency(synthgen_i) / 1000) * {_matlab_value(sfreq)} + 1;",
            f"    synthgen_width = (synthgen_peak_width(synthgen_i) / 1000) * {_matlab_value(sfreq)};",
            f"    synthgen_peak = exp(-0.5 .* (((1:{int(T)}) - synthgen_latency) ./ (synthgen_width / 6)) .^ 2) ./ (sqrt(2 .* pi) .* (synthgen_width / 6));",
            "    if any(synthgen_peak), synthgen_peak = utl_normalise(synthgen_peak, synthgen_peak_amplitude(synthgen_i)); end",
            "    synthgen_signal = synthgen_signal + synthgen_peak;",
            "end",
        ]
    )


def _ersp_signal_code(signal_class: Signal, T: int, sfreq: float) -> str:
    p = signal_class["params"]
    return "\n".join(
        [
            f"rng({int(p.get('seed', 0))}, 'twister');",
            f"synthgen_t = (0:{int(T) - 1}) / {_matlab_value(sfreq)};",
            (
                "synthgen_signal = sin("
                f"{_matlab_value(p['phase_cycles'])} * 2 * pi + "
                f"2 * pi * {_matlab_value(p['frequency_hz'])} * synthgen_t);"
            ),
            f"synthgen_signal = utl_normalise(synthgen_signal, {_matlab_value(p['amplitude'])});",
            f"synthgen_latency = floor(({_matlab_value(p['mod_latency_ms'])} / 1000) * {_matlab_value(sfreq)}) + 1;",
            f"synthgen_width = floor(({_matlab_value(p['mod_width_ms'])} / 1000) * {_matlab_value(sfreq)});",
            f"synthgen_taper = {_matlab_value(p['mod_taper'])};",
            "if synthgen_width < 1",
            "    synthgen_win = zeros(1, 0);",
            "elseif synthgen_width == 1",
            "    synthgen_win = 1;",
            "else",
            "    synthgen_x = (0:synthgen_width-1) ./ (synthgen_width - 1);",
            "    synthgen_win = ones(1, synthgen_width);",
            "    if synthgen_taper >= 1",
            "        synthgen_win = 0.5 .* (1 - cos(2 .* pi .* synthgen_x));",
            "    elseif synthgen_taper > 0",
            "        synthgen_left = synthgen_x < synthgen_taper / 2;",
            "        synthgen_right = synthgen_x >= 1 - synthgen_taper / 2;",
            "        synthgen_win(synthgen_left) = 0.5 .* (1 + cos(pi .* (2 .* synthgen_x(synthgen_left) ./ synthgen_taper - 1)));",
            "        synthgen_win(synthgen_right) = 0.5 .* (1 + cos(pi .* (2 .* synthgen_x(synthgen_right) ./ synthgen_taper - 2 ./ synthgen_taper + 1)));",
            "    end",
            "end",
            "if synthgen_latency > ceil(synthgen_width / 2)",
            "    synthgen_win = [zeros(1, synthgen_latency - ceil(synthgen_width / 2)), synthgen_win];",
            "else",
            "    synthgen_win(1:ceil(synthgen_width / 2) - synthgen_latency) = [];",
            "end",
            f"if length(synthgen_win) > {int(T)}",
            f"    synthgen_win({int(T)}+1:length(synthgen_win)) = [];",
            f"elseif length(synthgen_win) < {int(T)}",
            f"    synthgen_win = [synthgen_win, zeros(1, {int(T)} - length(synthgen_win))];",
            "end",
            "synthgen_signal = synthgen_signal .* synthgen_win;",
        ]
    )


def _arm_signal_code(signal_class: Signal, T: int, sfreq: float, epoch_ms: float) -> str:
    p = signal_class["params"]
    return "\n".join(
        [
            f"rng({int(p.get('seed', 0))}, 'twister');",
            (
                "synthgen_epochs = struct('n', 1, "
                f"'srate', {_matlab_value(sfreq)}, "
                f"'length', {_matlab_value(epoch_ms)}, 'prestim', 0);"
            ),
            f"synthgen_arm = {_matlab_value(_arm_coefficients(p['order'], p.get('seed', 0)))};",
            (
                "synthgen_signal = arm_generate_signal("
                f"1, {int(T)}, {int(p['order'])}, 0, 0, synthgen_arm);"
            ),
            f"synthgen_signal = utl_normalise(synthgen_signal, {_matlab_value(p['amplitude'])});",
        ]
    )


def _arm_coefficients(order: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    coeffs = rng.normal(0.0, 0.03, size=int(order))
    return coeffs.astype(np.float64)


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
        return {"type": "arm", "order": p["order"], "amplitude": p["amplitude"]}
    raise ValueError(f"Unknown SEREEGA signal class: {signal_class['type']!r}")
