import logging
import os
import sys

def get_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(os.path.join(args.save_path, "test.log" if args.evaluate 
                                                    else "train.log"))
    #file_format = "'%(asctime)s %(levelname)s %(message)s'"
    file_handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger
