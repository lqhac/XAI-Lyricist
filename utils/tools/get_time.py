from datetime import datetime
from datetime import timedelta
from datetime import timezone
import time


def get_time():
    SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai',)
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)   # 协调世界时
     # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)
    beijing_now_str = beijing_now.strftime("%Y%m%d:%H%M%S")
    # print("北京时间: ", beijing_now_str)

    # print(utc_now, utc_now.tzname())
    # print(utc_now.date(), utc_now.tzname())

    # print(beijing_now, )
    # print(beijing_now.tzname())
    # print(beijing_now.date())
    # print()
    # exp_date = time.strftime("%Y%m%d_%H%M%S", time.localtime()) # e.g., 20211129
    # print(beijing_now_str)
    return beijing_now_str
    

get_time()