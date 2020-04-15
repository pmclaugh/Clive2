from datetime import datetime


def timed(func):
    def decorated(*args, **kwargs):
        s = datetime.now()
        ret = func(*args, **kwargs)
        print(func.__name__, "{:.4f}".format((datetime.now() - s).total_seconds()))
        return ret
    return decorated
