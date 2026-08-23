"""Demonstration, perception, and real-log data interfaces.

Concrete classes intentionally live in their named submodules. Keeping this
package initializer side-effect free avoids an environment/data import cycle
when Gymnasium loads the environment entry point.
"""

from smartpick_vla.data.audit import AUDIT_SCHEMA_VERSION, audit_perception_dataset

__all__ = ["AUDIT_SCHEMA_VERSION", "audit_perception_dataset"]
