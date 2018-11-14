import configparser
import src
import os
LOG_PATH = os.path.dirname(src.__file__)


LOG_NAME = 'fileLogger'  # logger 实例名称，在logging.yaml中定义
LOG_YAML = LOG_PATH + '/logging.yaml'
ALT_LOG_LEVEL = 'DEBUG'  # 当logging.yaml文件不存在时，备选的LOG级别



