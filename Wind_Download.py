from WindPy import w
import pickle
import pandas as pd
import numpy as np


def CN_stock_name(code):
    if len(code) < 6:
        code = '0'*(6-len(code))+code
    if code.startswith('6'):
        code = code + '.SS'
    else:
        code = code + '.SZ'
    return code


def HK_stock_name(code):
    code = code[1:] + '.HK'
    return code


def read_file(filename):
    df = pd.read_csv(filename, header=None, dtype='str')
    CN_stock, HK_stock = df[0].apply(CN_stock_name).values, df[1].apply(HK_stock_name).values
    return CN_stock, HK_stock


def main():
    CN_stock, HK_stock = read_file('NameList.csv')
    w.start()
    res = {}
    fields = 'open,close,turn,volume,amt,holder_totalbyinst,total_shares'
    for i in range(len(CN_stock)):
        item = CN_stock[i]
        if i % 20 == 0:
            print('CN stock', i)
        data = w.wsd(item, fields, '2014-1-1', '2020-12-1', 'Fill=Previous,Currency=CNY,PriceAdj=F', usedf=True)
        res[item] = data
    for i in range(len(HK_stock)):
        item = HK_stock[i]
        if i % 20 == 0:
            print('HK stock', i)
        data = w.wsd(item, fields, '2014-1-1', '2020-12-1', 'Fill=Previous,Currency=CNY,PriceAdj=F', usedf=True)
        res[item] = data
    with open('WindPyData.pkl', 'wb') as file:
        pickle.dump(res, file)


def demo():
    CN_stock, HK_stock = read_file('NameList.csv')
    w.start()
    res = {}
    fields = 'open,close,turn,volume,amt,holder_totalbyinst,total_shares'
    item = CN_stock[0]
    errorcode, data = w.wsd(item, fields, '2014-1-1', '2020-12-1', 'Fill=Previous,Currency=CNY,PriceAdj=F', usedf=True)
    print(errorcode)
    print(data.columns)
    res[item] = data
    with open('Demo.pkl', 'wb') as file:
        pickle.dump(res, file)


if __name__ == '__main__':
    demo()
