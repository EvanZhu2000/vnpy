import os

current_environment = os.getenv('APP_ENV', 'uat')

ctp_setting_uat = {
    "用户名": "224829",
    "密码": "evan@cash1q2",
    "经纪商代码": "9999",
    "交易服务器": "tcp://180.168.146.187:10202",
    "行情服务器": "tcp://180.168.146.187:10212",
    "产品名称": "simnow_client_test",
    "授权编码": "0000000000000000",
    "产品信息": ""
}

ctp_setting_evan = {
    "用户名": "2520000355",
    "密码": "jacky83611",
    "经纪商代码": "4040",
    "交易服务器": "tcp://180.166.103.21:55205",
    "行情服务器": "tcp://180.166.103.21:55213",
    "产品名称": "client_LO_1.0",
    "授权编码": "IM53ZG1HKVEYHPAI",
    "产品信息": ""
}

ctp_setting_dev = {
    "用户名": "226593",
    "密码": "evan@cash1q2",
    "经纪商代码": "9999",
    "交易服务器": "tcp://180.168.146.187:10202",
    "行情服务器": "tcp://180.168.146.187:10212",
    "产品名称": "simnow_client_test",
    "授权编码": "0000000000000000",
    "产品信息": ""
}

def ctp_map(option):
    if option == 'uat':
        ctp_setting = ctp_setting_uat
    elif option == 'evan':
        ctp_setting = ctp_setting_evan
    elif option == 'dev':
        ctp_setting = ctp_setting_dev
    else:
        raise Exception(f'Wrong option input {option}')
    return ctp_setting

def get_ctp_settings():
    return ctp_map(current_environment)