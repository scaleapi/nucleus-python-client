from typing import Optional


def try_convert_float(float_str: Optional[str] = None) -> Optional[float]:
    if float_str is None:
        return None
    try:
        return float(float_str)
    except ValueError:
        return None
