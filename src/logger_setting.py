# -*- coding utf-8 -*-
"""
Multiple calls to logging.getLogger('someLogger') return a reference to the same logger object.
This is true not only within the same module, but also across modules as long as it is in the same Python interpreter process.
    https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""
import yaml
import logging.config
import os
from src import config


def setup_logging(yaml_conf=config.LOG_YAML, log_level=config.ALT_LOG_LEVEL):
    if os.path.exists(yaml_conf):
        with open(yaml_conf, "r") as f:
            logging.config.dictConfig(yaml.load(f))
    else:
        logging.basicConfig(level=log_level)
    return logging.getLogger(config.LOG_NAME)


if __name__ == '__main__':
    logger = setup_logging()

    logger.info("info log test")
    logger.debug("debug log test")
    logger.warning("warning log test")
    logger.warning("error log test")
    logger.critical("critical log test")