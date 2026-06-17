"""Validate that the dist directory contains the expected release bundle."""

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

PYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313", "cp314")


NATIVE_EXTENSION_PATTERNS = {
    "manylinux": re.compile(r"^nucleus/_native_dedup\..*-linux-gnu\.so$"),
    "macosx": re.compile(r"^nucleus/_native_dedup\..*-darwin\.so$"),
    "win_amd64": re.compile(r"^nucleus/_native_dedup\..*\.pyd$"),
}


def _version_from_pyproject(pyproject_path: Path) -> str:
    for line in pyproject_path.read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError(f"{pyproject_path} is missing a version field")


def _matching_platform_pattern(wheel_name: str) -> Optional[re.Pattern[str]]:
    for platform_key, pattern in NATIVE_EXTENSION_PATTERNS.items():
        if platform_key in wheel_name:
            return pattern
    return None


def _has_matching_wheel(
    wheels: List[str],
    python_tag: str,
    platform_match: Callable[[str], bool],
) -> bool:
    return any(
        f"-{python_tag}-{python_tag}-" in wheel and platform_match(wheel)
        for wheel in wheels
    )


def validate_release_artifacts(
    dist_dir: Path,
    pyproject_path: Path,
) -> List[str]:
    version = _version_from_pyproject(pyproject_path)
    files = sorted(path.name for path in dist_dir.iterdir() if path.is_file())
    wheels = [filename for filename in files if filename.endswith(".whl")]
    errors = []

    print("Release artifacts:")
    for filename in files:
        print(f"  {filename}")

    expected_sdist = f"scale_nucleus-{version}.tar.gz"
    if expected_sdist not in files:
        errors.append(f"missing sdist {expected_sdist}")

    if len(wheels) != 15:
        errors.append(f"expected 15 wheels, found {len(wheels)}")

    expected_platforms = {
        "linux": lambda name: "manylinux" in name,
        "macos": lambda name: "macosx_" in name
        and name.endswith("_arm64.whl"),
        "windows": lambda name: name.endswith("win_amd64.whl"),
    }
    for python_tag in PYTHON_TAGS:
        for platform_name, platform_match in expected_platforms.items():
            if not _has_matching_wheel(wheels, python_tag, platform_match):
                errors.append(f"missing {python_tag} {platform_name} wheel")

    for wheel in wheels:
        wheel_path = dist_dir / wheel
        with zipfile.ZipFile(wheel_path) as wheel_zip:
            native_files = [
                name
                for name in wheel_zip.namelist()
                if "_native_dedup" in name
                and (name.endswith(".so") or name.endswith(".pyd"))
            ]

        if len(native_files) != 1:
            errors.append(
                f"{wheel} should contain one native binary, found {native_files}"
            )
            continue

        expected_pattern = _matching_platform_pattern(wheel)
        if expected_pattern is None:
            errors.append(f"{wheel} has an unexpected platform tag")
        elif not expected_pattern.match(native_files[0]):
            errors.append(
                f"{wheel} contains unexpected native binary {native_files[0]}"
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="dist", type=Path)
    parser.add_argument("--pyproject", default="pyproject.toml", type=Path)
    args = parser.parse_args()

    errors = validate_release_artifacts(args.dist_dir, args.pyproject)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
