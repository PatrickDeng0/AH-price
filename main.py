import pickle
import Strategy_Def, Download, Strats, util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Considering the extreme markets in 2015 and 2016, we start from 2016-01-01
def data_prepare(start_date='2016-01-01'):
    # Data Preparation
    with open('Data/Data.pkl', 'rb') as file:
        df = pickle.load(file)
    with open('Data/CNYHKD.pkl', 'rb') as file:
        CNYHKD = pickle.load(file)
    with open('Data/Indices.pkl', 'rb') as file:
        Indices = pickle.load(file)

    # Read Stock Data
    CN_stock, HK_stock = Download.read_file('NameList.csv')
    df_ac = df['Adj Close'][df['Adj Close'].index >= start_date]
    data = df_ac[list(CN_stock) + list(HK_stock)].values

    # Read Currency Exchange data
    CH_series = CNYHKD['Adj Close'][CNYHKD['Adj Close'].index >= start_date]
    dates = list(df_ac.index)
    CH_series = CH_series.loc[dates].values

    # Read Indices Data
    SSECI, HSCEI = Indices['000001.SS']['Adj Close'], Indices['^HSCE']['Adj Close']
    SSECI, HSCEI = SSECI[SSECI.index >= start_date].values, HSCEI[HSCEI.index >= start_date].values
    HSCEI = HSCEI * CH_series

    # Calculate the AH price multiplier
    pair_loca = data.shape[1] // 2
    CN_data, HK_data = data[:, :pair_loca], data[:, pair_loca:] * CH_series[:, np.newaxis]
    AH_multiple = CN_data / HK_data
    return CN_data, HK_data, AH_multiple, SSECI, HSCEI, dates


def train1(CN_data, HK_data, AH_multiple, SSECI, HSCEI, dates, trans_fee=0.3/100, trade_freq=22, stock_num=20):
    # Strategies
    MA_Least20_A = Strats.LeastNumCurAH(trans_fee=trans_fee, stock_rank=[0,stock_num], market='A', long=True,
                                        MA_window=1, MA_method='fair', weights='fair')
    NegaSSE_A = Strategy_Def.IndexStrategy(trans_fee=trans_fee, market='A', long=False, profit=SSECI/SSECI[0])

    Decision = Strategy_Def.CombineStrategy([MA_Least20_A, NegaSSE_A], profit_merge=True)
    Strategy_Def.Simulate_Strategy(Decision, trade_freq, CN_data, HK_data, AH_multiple)

    # Display
    # Decision.info(dates=dates, name='Long MA Least 20 Short SSECI Merge Profit')
    # MA_Least20_A.info(dates=dates, name='Long MA Least 20')
    #
    # plt.plot(Decision.value_record, label='Long MA Least 20 Short SSECI Merge Profit')
    # plt.plot(MA_Least20_A.value_record, label='Long MA Least 20')
    # plt.plot(NegaSSE_A.value_record, label='Short SSECI')
    # plt.plot(SSECI/SSECI[0], label='SSECI')
    # plt.plot(HSCEI/HSCEI[0], label='HSCEI')
    # plt.title('PnL')
    # plt.legend()
    # plt.show()

    polys = util.get_beta_alpha(MA_Least20_A, NegaSSE_A, 'MA20', 'SSE', neutral=True, display=False)
    return polys[1], Decision.get_sharpe()


def gridsearch(max_trade_freq=10, max_stock_num=10):
    CN_data, HK_data, AH_multiple, SSECI, HSCEI, dates = data_prepare()
    trade_freq_s = np.arange(1, max_trade_freq+1) * 5
    stock_num_s = np.arange(1, max_stock_num+1) * 5

    res_alpha, res_sharpe = np.zeros((max_trade_freq, max_stock_num)), np.zeros((max_trade_freq, max_stock_num))

    for i in range(len(trade_freq_s)):
        for j in range(len(stock_num_s)):
            trade_freq, stock_num = trade_freq_s[i], stock_num_s[j]
            alpha, sharpe = train1(CN_data, HK_data, AH_multiple, SSECI, HSCEI, dates,
                                   trans_fee=0.3/100, trade_freq=trade_freq, stock_num=stock_num)
            res_alpha[i,j] = alpha
            res_sharpe[i,j] = sharpe

    res_alpha, res_sharpe = pd.DataFrame(res_alpha), pd.DataFrame(res_sharpe)
    res_alpha.index = trade_freq_s
    res_alpha.columns = stock_num_s
    res_sharpe.index = trade_freq_s
    res_sharpe.columns = stock_num_s
    return res_alpha, res_sharpe


def bin_effect():
    return


if __name__ == '__main__':
    res_alpha, res_sharpe = gridsearch()
