import pickle
import Strategy_Def, Download, Strats, util
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Considering the extreme markets in 2015 and 2016, we start from 2016-01-01
    start_date = '2016-01-01'

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

    # Simulate Settings
    trans_fee = 0.2/100
    trade_freq = 22

    # Strategies
    # Least20_A = Strats.LeastCurAH(trans_fee=trans_fee, stock_num=20, market='A', long=True)
    MA_Least20_A = Strats.MA_LeastNumCurAH(trans_fee=trans_fee, stock_num=20, market='A', long=True,
                                           MA_window=2*trade_freq)
    NegaSSE_A = Strategy_Def.IndexStrategy(trans_fee=trans_fee, market='A', long=False, profit=SSECI/SSECI[0])

    Decision = Strategy_Def.CombineStrategy([MA_Least20_A, NegaSSE_A], profit_merge=True)
    Strategy_Def.Simulate_Strategy(Decision, trade_freq, CN_data, HK_data, AH_multiple)

    # Display
    Decision.info(dates=dates, name='Long MA Least 20 Short SSECI Merge Profit')
    MA_Least20_A.info(dates=dates, name='Long MA Least 20')
    NegaSSE_A.info(dates=dates, name='Short SSECI')

    plt.plot(Decision.value_record, label='Long MA Least 20 Short SSECI Merge Profit')
    plt.plot(MA_Least20_A.value_record, label='Long MA Least 20')
    plt.plot(NegaSSE_A.value_record, label='Short SSECI')
    plt.plot(SSECI/SSECI[0], label='SSE CI')
    plt.plot(HSCEI/HSCEI[0], label='HSCEI')
    plt.title('PnL')
    plt.legend()
    plt.show()

    util.get_beta_alpha(MA_Least20_A, NegaSSE_A, 'MA20', 'SSE', neutral=True)


if __name__ == '__main__':
    main()
