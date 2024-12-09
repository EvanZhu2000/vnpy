from datetime import datetime, time, date


# start and end must be earlier than crontab start and end
DAY_START_CHINAFUTURES = time(8, 30)
DAY_END_CHINAFUTURES = time(15, 30)

NIGHT_START_CHINAFUTURES = time(20, 30)
NIGHT_END_CHINAFUTURES = time(2, 30)

def check_trading_period_chinafutures():
    """"""
    current_time = datetime.now().time()

    trading = False
    if (
        (current_time >= DAY_START_CHINAFUTURES and current_time <= DAY_END_CHINAFUTURES)
        or (current_time >= NIGHT_START_CHINAFUTURES)
        or (current_time <= NIGHT_END_CHINAFUTURES)
    ):
        trading = True

    return trading