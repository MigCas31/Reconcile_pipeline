"""Re-export shim - canonical implementation lives in reconcile_v2.polyhedral_kernel."""

from reconcile_v2.polyhedral_kernel import (
    HalfspacePlane,
    build_arranged_polyhedral_cell,
)

__all__ = ["HalfspacePlane", "build_arranged_polyhedral_cell"]
