from contextlib import contextmanager
import sys, os
import io

# Only import wurlitzer on POSIX
if sys.platform != "win32":
    from wurlitzer import pipes
else:
    # Define a dummy context manager for Windows
    from contextlib import contextmanager

    @contextmanager
    def pipes(*args, **kwargs):
        yield


@contextmanager
def suppress_stdout():
    buf = io.StringIO()
    with open(os.devnull, "w") as devnull:
        # Use pipes context (dummy on Windows)
        with pipes(stdout=buf, stderr=buf):
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
