from Strategy_Def import BasicStrategy
import util
import numpy as np


# Select the A stocks with Least AH multiplier in Moving Average window
class LeastNumCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_rank, market, long, MA_window, MA_method, weights):
        super().__init__(trans_fee, market, long)
        self.stock_rank = stock_rank
        self.MA_window = MA_window
        self.MA_method = MA_method
        self.weights = weights

        # Assert stock rank range
        assert 0 <= stock_rank[0] < stock_rank[1]

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        cur = util.get_MA(AH_info_MA, self.MA_method)
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_rank[1]:
            return 'keep'
        else:
            stocks = np.argsort(cur)[self.stock_rank[0]:self.stock_rank[1]]
            weights = util.get_weights(cur[stocks], self.weights)
            return util.sw_2_position(stocks, weights)


# Select the A stocks with Highest AH multiplier in Moving Average window
class HighestNumCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_rank, market, long, MA_window, MA_method, weights):
        super().__init__(trans_fee, market, long)
        self.stock_rank = stock_rank
        self.MA_window = MA_window
        self.MA_method = MA_method
        self.weights = weights

        # Assert stock rank range
        assert 0 <= stock_rank[0] < stock_rank[1]

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        cur = util.get_MA(AH_info_MA, self.MA_method)
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_rank[1]:
            return 'keep'
        else:
            stocks = np.argsort(cur)[(realvalue-self.stock_rank[1]):(realvalue - self.stock_rank[0])]
            weights = util.get_weights(cur[stocks], self.weights)
            return util.sw_2_position(stocks, weights)


# Select the A stocks with Largest AH multiplier difference against moving window
class DiffLeastNumCurAH(BasicStrategy):
    def __init__(self, trans_fee, stock_rank, market, long, MA_window, MA_method, weights):
        super().__init__(trans_fee, market, long)
        self.stock_rank = stock_rank
        self.MA_window = MA_window
        self.MA_method = MA_method
        self.weights = weights

        # Assert stock rank range
        assert 0 <= stock_rank[0] < stock_rank[1]

    # Only select the 20 least AH multiple stocks on the most current snapshot data
    def stock_select(self, CN_data_info, HK_data_info, AH_info):
        if len(AH_info) <= self.MA_window:
            return 'keep'
        AH_info_MA = AH_info[-self.MA_window:, :]
        AH_MAmean = util.get_MA(AH_info_MA, self.MA_method)
        cur = AH_info[-1] / AH_MAmean
        realvalue = np.sum(~np.isnan(cur))
        if realvalue < self.stock_rank[1]:
            return 'keep'
        else:
            stocks = np.argsort(cur)[self.stock_rank[0]:self.stock_rank[1]]
            weights = util.get_weights(cur[stocks], self.weights)
            return util.sw_2_position(stocks, weights)
