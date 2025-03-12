import sys
import contextlib

class DummyStdOut():
    def write(self, x): pass

@contextlib.contextmanager
def no_stdout():
    save_stdout = sys.stdout
    sys.stdout = DummyStdOut()
    yield
    sys.stdout = save_stdout
    return
