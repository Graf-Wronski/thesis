from contextlib import contextmanager
import sys, os
from wurlitzer import pipes
import io


@contextmanager
def suppress_stdout():
    buf = io.StringIO()
    with open(os.devnull, "w") as devnull:
        with pipes(stdout=buf, stderr=buf):
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
