import os
import sys
import logging

from toolkits.dist_utils import get_dist_info

__all__ = ['init_logger', 'get_logger']


class Dumb_Logger:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def init_logger(args, logger_name='LLCV_ZOO'):
    # if this is a slave thread, returns an empty logger
    rank, _ = get_dist_info()
    if rank != 0:
        dumb_logger = Dumb_Logger()
        return dumb_logger

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger(logger_name)
    if args.save_flag:
        file_handler = logging.FileHandler(os.path.join(args.save_path, "test.log" if
                                                        args.evaluate else "train.log"))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger


def get_logger(logger_name='LLCV_ZOO'):
    rank, _ = get_dist_info()
    if rank != 0:
        dumb_logger = Dumb_Logger()
        return dumb_logger

    logger = logging.getLogger(logger_name)
    assert logger.hasHandlers()

    return logger
