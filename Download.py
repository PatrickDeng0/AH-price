import pandas as pd
import numpy as np
import yfinance as yf
import pickle


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
    # Download Stocks
    CN_stock, HK_stock = read_file('NameList.csv')
    stock_list = ' '.join(CN_stock) + ' ' + ' '.join(HK_stock)
    data = yf.download(stock_list, start="2014-01-01", end="2020-12-19")
    with open('Data/Data.pkl', 'wb') as file:
        pickle.dump(data, file)

    # Download Currency Exchange rate
    CNYHKD = yf.download('HKDCNY=X', start="2014-01-01", end="2020-12-19")
    with open('Data/CNYHKD.pkl', 'wb') as file:
        pickle.dump(CNYHKD, file)

    # Download Indexs
    Indices = yf.download('^HSCE 000300.SS', start="2014-01-01", end="2020-12-19", group_by='ticker')
    with open('Data/Indices.pkl', 'wb') as file:
        pickle.dump(Indices, file)


if __name__ == '__main__':
    main()
