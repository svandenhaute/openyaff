import yaff


yaff.log.set_level(yaff.log.silent)




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
