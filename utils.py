from datetime import datetime
import logging

logger = logging.getLogger('rtv3')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def timed(func):
    def decorated(*args, **kwargs):
        s = datetime.now()
        ret = func(*args, **kwargs)
        logger.info("%s - %.4f", func.__name__, (datetime.now() - s).total_seconds())
        return ret
    return decorated
