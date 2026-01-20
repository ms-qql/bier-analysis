import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from . import database 

from . import matrix_strategy 
from . import graphs



######## Main  ############################################################################################################

# ----------------------------- Own support functions -------------------------------------------------------------


def get_start_date(timewindow=30):
  time_now = datetime.now(timezone.utc)
  time_rounded = time_now.replace(minute=0, second=0, microsecond=0) # round to previous full hour
  start_time = time_rounded - timedelta(days=timewindow)
  start_date = start_time.strftime("%Y-%m-%d %H:%M") 
  return start_date

def get_end_date():
  time_now = datetime.now(timezone.utc)
  time_rounded = time_now.replace(minute=0, second=0, microsecond=0) # round to previous full hour
  end_date = time_rounded.strftime("%Y-%m-%d %H:%M") 
  return end_date
