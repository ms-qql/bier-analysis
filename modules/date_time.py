
import datetime

# ---------------------- rounding functions -----------------------------------------------------------

def rounddown_to_hour(t):
    # Rounds to current hour
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour))

def round_to_hour(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30))

  
# ---------------------- date (only) conversions -----------------------------------------------------------
  
def convert_date_str_datetime(date, output_mode):
    # Output Modes:
    # german: '24.12.2021'
    # world: '2021-12-24'
    # unix: unix timestamp

    input_mode = 'date_format'
    try:
        unix_float = float(date) # check if number can be converted, works only for unix
        input_mode = 'unix'
        date_str = str(date)[:10]
        date_conv = datetime.date.fromtimestamp(int(date_str))
    except:
        pass      
    
    date_check = str(date)[:10] + ' ' # add space to enable check for german long/short format
    
    if input_mode == 'date_format':
        if date_check[2] == '.' and date_check[5] == '.' and date_check[8] == ' ' :
            #print('german short format')
            date_str = str(date)[:8]
            date_conv = datetime.datetime.strptime(date_str, "%d.%m.%y") 
        elif date_check[2] == '.' and date_check[5] == '.' and date_check[10] == ' ' :
            #print('german long format')
            date_str = str(date)[:10]
            date_conv = datetime.datetime.strptime(date_str, "%d.%m.%Y")
        elif date_check[2] == '/' and date_check[5] == '/' and date_check[8] == ' ' :
            #print('uk short format')
            date_str = str(date)[:8]
            date_conv = datetime.datetime.strptime(date_str, "%d/%m/%y") 
        elif date_check[2] == '/' and date_check[5] == '/' and date_check[10] == ' ' :
            #print('uk long format')
            date_str = str(date)[:10]
            date_conv = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        elif date_check[4] == '-' and date_check[7] == '-' :
            #print('world long format')
            date_str = str(date)[:10]
            date_conv = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        elif date_check[2] == '-' and date_check[5] == '-' :
            #print('world short format')
            date_str = str(date)[:8]
            date_conv = datetime.datetime.strptime(date_str, "%y-%m-%d")
        else:
            print('unknown format')

    if output_mode.lower() == 'german':
        date_out = date_conv.strftime("%d.%m.%Y")
    elif output_mode.lower() == 'world':
        date_out = date_conv.strftime("%Y-%m-%d")
    elif output_mode.lower() == 'unix':
        date_supp = datetime.datetime.strptime(str(date_conv)[:10], "%Y-%m-%d")
        date_out = round(datetime.datetime.timestamp(date_supp))
    return date_out
         
  
# ---------------------- date & time conversions -----------------------------------------------------------

def convert_date_time_str_datetime(date, output_mode):
    # Output Modes:
    # german: '24.12.2021 13:00'
    # world: '2021-12-24 13:00'
    # unix: unix timestamp
    # 
    # result rounded to previous full hour
    rounding = False
    
    input_mode = 'date_format'
    try:
        unix_float = float(date) # check if number can be converted, works only for unix
        input_mode = 'unix'
        date_str = str(date)[:16]
        date_conv = datetime.date.fromtimestamp(int(date_str))
    except:
        pass      
    
    date_check = str(date)[:16] + ' ' # add space to enable check for german long/short format
    #print(date_check)
    
    if input_mode == 'date_format':
        if date_check[2] == '.' and date_check[5] == '.' and date_check[8] == ' ' :
            print('german short format')
            date_str = str(date)[:14]
            date_conv = datetime.datetime.strptime(date_str, "%d.%m.%y %H:%M") 
        elif date_check[2] == '.' and date_check[5] == '.' and date_check[10] == ' ' :
            print('german long format')
            date_str = str(date)[:16]
            date_conv = datetime.datetime.strptime(date_str, "%d.%m.%Y %H:%M")
        elif date_check[2] == '/' and date_check[5] == '/' and date_check[8] == ' ' :
            print('uk short format')
            date_str = str(date)[:14]
            date_conv = datetime.datetime.strptime(date_str, "%d/%m/%y %H:%M") 
        elif date_check[2] == '/' and date_check[5] == '/' and date_check[10] == ' ' :
            print('uk long format')
            date_str = str(date)[:16]
            print(date_str)
            date_conv = datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M")
        elif date_check[4] == '-' and date_check[7] == '-' :
            #print('world long format')
            date_str = str(date)[:16]
            date_conv = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        elif date_check[2] == '-' and date_check[5] == '-' :
            print('world short format')
            date_str = str(date)[:16]
            date_conv = datetime.datetime.strptime(date_str, "%y-%m-%d %H:%M")
        else:
            print('unknown format')

    #if rounding:
    #    date_result = rounddown_to_hour(date_conv)
    #else:
    date_result = date_conv
            
    if output_mode.lower() == 'german':
        date_out = date_result.strftime("%d.%m.%Y %H:%M")
    elif output_mode.lower() == 'world':
        date_out = date_result.strftime("%Y-%m-%d %H:%M")
    elif output_mode.lower() == 'unix':
        date_supp = datetime.datetime.strptime(str(date_result)[:10], "%Y-%m-%d")
        date_out = round(datetime.datetime.timestamp(date_supp))
    return date_out
         
# ---------------------- Get current dates & time

def date_today():
    # returns today's date as string
    now = datetime.datetime.now() # current date and time
    today = now.strftime("%Y-%m-%d")
    return today

def date_yesterday():
    # returns yesterday's date as string
    now = datetime.datetime.now() # current date and time
    yesterday = now - datetime.timedelta(days = 1) # Calculate yesterdays' date
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    return yesterday_str
  
