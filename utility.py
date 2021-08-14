import math
import logging
import os
import subprocess
import yaml
import pandas as pd
import datetime 
import gc
import re
import gzip
import shutil
import numpy as np
import seaborn as sns

################
# File Reading #
################

def read_config_file(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)