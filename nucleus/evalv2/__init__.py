"""Local EvaluationV2 harness: run the shared evalv2-core kernel over parquet on a laptop.

This subpackage is intentionally import-light — it does NOT import the heavy top-level ``nucleus``
SDK. The parquet adapter depends only on ``evalv2_core`` + ``pyarrow``, mapping parquet columns
straight to the kernel's storage-agnostic types.
"""
