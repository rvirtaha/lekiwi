import sys

p = sys.argv[1]
t = open(p).read()
t = t.replace(
    "if not width_success or self.capture_width != actual_width:",
    "if self.capture_width != actual_width:",
)
t = t.replace(
    "if not height_success or self.capture_height != actual_height:",
    "if self.capture_height != actual_height:",
)
t = t.replace(
    "if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):",
    "if not math.isclose(self.fps, actual_fps, rel_tol=1e-3):",
)
t = t.replace(
    "def async_read(self, timeout_ms: float = 200)",
    "def async_read(self, timeout_ms: float = 2000)",
)
open(p, "w").write(t)
print("done")
