"""Compatibility wrapper for running the new V2 topology pipeline.

Usage:
  python -m reconcile.cli_v2 --building ... --scan-cache ... --output-v2 ...
"""

from reconcile_v2.cli import main

if __name__ == "__main__":
    main()
