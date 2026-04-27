from __future__ import annotations

from typing import Any


def get_tvb_model(name: str) -> Any:
    """Return a fresh TVB model instance by short name.

    Import is lazy so callers that never hit this function don't need tvb.
    """
    from tvb.simulator import models

    name = name.lower()
    if name == "jansen_rit":
        return models.JansenRit()
    if name == "generic_2d_oscillator":
        return models.Generic2dOscillator()
    if name == "wilson_cowan":
        return models.WilsonCowan()
    raise ValueError(
        f"Unknown TVB model {name!r}. Supported: jansen_rit, generic_2d_oscillator, wilson_cowan"
    )


def model_output_channel(name: str) -> int:
    """Index into state variables used as the observable (source activity proxy).

    Jansen-Rit: y1 - y2 (pyramidal cell output) is at index 1; we return 1 here
    and let the caller compute y1 - y2 if needed.
    """
    name = name.lower()
    if name == "jansen_rit":
        return 1
    if name == "generic_2d_oscillator":
        return 0
    if name == "wilson_cowan":
        return 0
    raise ValueError(f"Unknown TVB model {name!r}")