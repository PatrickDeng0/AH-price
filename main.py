import pickle
import Strategy_Def, Download
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
    Least20 = Strategy_Def.LeastCurAH(trans_fee=trans_fee, stock_num=20, market='A', long=True)
    Highest20 = Strategy_Def.HighestCurAH(trans_fee=trans_fee, stock_num=20, market='A', long=False)
    LH20_profit = Strategy_Def.CombineStrategy([Least20, Highest20], profit_merge=False)
    Strategy_Def.Simulate_Strategy(LH20_profit, trade_freq, CN_data, HK_data, AH_multiple)

    # Display
    LH20_profit.info(dates=dates, name='Long Least 20 A Short Highest 20 A Profit')
    Least20.info(dates=dates, name='Long Least 20 A')
    Highest20.info(dates=dates, name='Short Highest 20 A')

    plt.plot(Least20.profit_record, label='Least20')
    plt.plot(Highest20.profit_record, label='Highest20')
    plt.plot(LH20_profit.profit_record, label='LH20 Profit')
    plt.plot(SSECI/SSECI[0], label='SSE CI')
    plt.plot(HSCEI/HSCEI[0], label='HSCEI')
    plt.title('PnL')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
