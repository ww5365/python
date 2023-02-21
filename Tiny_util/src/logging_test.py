# -*- coding: utf-8 -*-

from logging import handlers
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


class Logger(object):
    '''
    @ description 
    输出log到控制台以及将日志写入log文件,保存2种类型的log
    all.log 保存debug, info, warning, critical 信息
    error.log则只保存error信息
    同时按照时间自动分割日志文件。

    @ 参考： https://www.cnblogs.com/nancyzhu/p/8551506.html
    '''
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)  # Logger： 记录器
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置记录器的日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出 处理器sh
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


if __name__ == '__main__':

    # 平时常用方式 : logging.basicConfig 往屏幕输出
    logging.info("this is a basicConfig test")

    # 同时往all.log 和 屏幕输出 debug级别以上的日志信息
    log = Logger('all.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')

    # error.log 和 屏幕输出 error日志信息
    Logger('error.log', level='error').logger.error('error')
