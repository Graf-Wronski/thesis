from datetime import datetime


def now() -> str:
    """ Return timestamp for file names."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")