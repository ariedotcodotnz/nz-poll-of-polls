"""Model package. Sets the XLA host device count before JAX is imported so NUTS chains run in parallel."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
