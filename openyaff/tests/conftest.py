import yaff
import numpy as np


yaff.log.set_level(yaff.log.silent)


def assert_tol(a, b, tol):
    """Asserts the relative error of b with respect to a is less than tol

    Parameters
    ----------

    a, b : array_like
        arrays to be compared against each other

    tol : float
        error tolerance

    """
    norm = np.linalg.norm(a)
    if norm > 0.0:
        if isinstance(a, np.ndarray): # compare average error
            delta = np.mean(np.linalg.norm(a - b, axis=1))
        else:
            delta = np.abs(a - b)
        assert np.all(delta / norm < tol)



#def pytest_addoption(parser):
#    """Add a command line option to disable logger."""
#    parser.addoption(
#        "--disable-log", action="append", default=[], help="disable specific loggers"
#    )
#
#
#def pytest_configure(config):
#    """Disable the loggers."""
#    for name in config.getoption("--disable-log", default=[]):
#        logger = logging.getLogger(name)
#        logger.propagate = False
