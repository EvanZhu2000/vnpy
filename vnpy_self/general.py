from datetime import datetime, time, date


# start and end must be earlier than crontab start and end
CHINA_DAY_START = time(8, 30)
CHINA_DAY_END = time(15, 35)

CHINA_NIGHT_START = time(20, 30)
CHINA_NIGHT_END = time(2, 35)

def check_trading_period():
    """"""
    current_time = datetime.now().time()

    trading = False
    if (
        (current_time >= CHINA_DAY_START and current_time <= CHINA_DAY_END)
        or (current_time >= CHINA_NIGHT_START)
        or (current_time <= CHINA_NIGHT_END)
    ):
        trading = True

    return trading