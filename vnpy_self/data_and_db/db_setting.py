import os

current_environment = os.getenv('APP_ENV', 'uat')


db_setting_live = {
    'host': "192.168.91.124",
    'user': "evan",
    'password': "Evan@cash1q2"
}

db_setting_uat = {
    'host': "192.168.91.121",
    'user': "evan",
    'password': "evan@cash1q2"
}

def db_map(option):
    if option == 'uat':
        db_setting = db_setting_uat
    elif option == 'live':
        db_setting = db_setting_live
    else:
        raise Exception(f'Unsupported environment option: {option}')
    return db_setting

def get_db_settings():
    return db_map(current_environment)