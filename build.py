from __future__ import annotations

import sys

from setuptools import Extension


def build(setup_kwargs):
    extra_compile_args = []
    if sys.platform != "win32":
        extra_compile_args.extend(["-std=c11", "-O3"])

    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "nucleus._native_dedup",
                    ["nucleus/_native_dedup.c"],
                    extra_compile_args=extra_compile_args,
                    optional=True,
                )
            ],
        }
    )
