from Strategy_Def import BasicStrategy
import numpy as np


# Select the A stocks with least AH multiplier
class LeastCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_num, market, long):
        super().__init__(trans_fee, market, long)
        self.stock_num = stock_num

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        cur = AH_info[-1]
        if np.sum(~np.isnan(cur)) < self.stock_num:
            return 'keep'
        else:
            return np.argsort(cur)[:self.stock_num]


# Select the A stocks with Least AH multiplier in Moving Average window
class MA_LeastCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_num, market, long, MA_window):
        super().__init__(trans_fee, market, long)
        self.stock_num = stock_num
        self.MA_window = MA_window

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        cur = np.nanmean(AH_info_MA, axis=0)
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_num:
            return 'keep'
        else:
            return np.argsort(cur)[:self.stock_num]


# Select the A stocks with highest AH multiplier
class HighestCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_num, market, long):
        super().__init__(trans_fee, market, long)
        self.stock_num = stock_num

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        cur = AH_info[-1]
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_num:
            return 'keep'
        else:
            return np.argsort(cur)[(realvalue-self.stock_num):realvalue]


# Select the A stocks with highest AH multiplier in moving average window
class MA_HighestCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_num, market, long, MA_window):
        super().__init__(trans_fee, market, long)
        self.stock_num = stock_num
        self.MA_window = MA_window

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        cur = np.nanmean(AH_info_MA, axis=0)
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_num:
            return 'keep'
        else:
            return np.argsort(cur)[(realvalue-self.stock_num):realvalue]


# Select the A stocks with Largest AH multiplier difference against moving window
class MA_DiffLeast_CurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_num, market, long, MA_window):
        super().__init__(trans_fee, market, long)
        self.stock_num = stock_num
        self.MA_window = MA_window

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        AH_MAmean = np.nanmean(AH_info_MA, axis=0)
        cur = AH_info[-1] / AH_MAmean
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_num:
            return 'keep'
        else:
            return np.argsort(cur)[(realvalue-self.stock_num):realvalue]
