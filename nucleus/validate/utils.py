from typing import Optional


def try_convert_float(float_str: str) -> Optional[float]:
    try:
        return float(float_str)
    except (ValueError, TypeError):
        return None
