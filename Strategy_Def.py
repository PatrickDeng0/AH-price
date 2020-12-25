import pickle, Download
import numpy as np
import matplotlib.pyplot as plt


def ret_2_profit(rets):
    return np.cumprod(rets+1)


def profit_2_ret(profit):
    add_profit = np.insert(profit, 0, 1)
    profit_chg = np.diff(add_profit)
    return profit_chg / add_profit[:-1]


# If there is too many nan in prices, then we record these new prices as the old prices
def merge_2_prices(old_prices, prices):
    if len(old_prices) == len(prices):
        nanlocas = np.isnan(prices)
        new_prices = prices[:]
        new_prices[nanlocas] = old_prices[nanlocas]
        return new_prices
    else:
        # If in the beginning, old_prices is empty list, then record the prices precisely
        return prices


# A basic strategy object that define the describe statement of a common strategy
class DescribeStrategy(object):
    def __init__(self):
        # profit_record: record everyday profit
        # trans_profit_record: record interval returns between each two transactions
        self.profit_record = np.array([])
        self.trans_profit_record = np.array([])
        self.ret_record = np.array([])
        self.trans_ret_record = np.array([])

    # Get the newest profit value
    @property
    def profit(self):
        if len(self.profit_record) > 0:
            return self.profit_record[-1]
        else:
            return 1

    @property
    def last_trans_profit(self):
        if len(self.trans_profit_record) > 0:
            return self.trans_profit_record[-1]
        else:
            return 1

    def get_sharpe(self):
        self.ret_record = profit_2_ret(self.profit_record)
        self.trans_ret_record = profit_2_ret(self.trans_profit_record)

        # If the transactions happen in a constant frequency, then compute its sharpe
        if len(self.trans_ret_record) > 10:
            return self.trans_ret_record.mean() / self.trans_ret_record.std()

        # If there is no transactions at all, then compute the sharpe from daily return series
        # By default using monthly period as an epoch (22 trading days)
        else:
            period = 22
            cut_ret_record = self.ret_record[-(len(self.ret_record)//period)*period:]
            cut_ret_record = cut_ret_record.reshape((-1, period))
            merge_ret_record = cut_ret_record.sum(axis=1)
            return merge_ret_record.mean() / merge_ret_record.std()

    # Calculate the Max Drawdown and details of this strategy
    def get_mdd(self):
        profit_record = self.profit_record
        former_high, mdd_high, mdd_low, mdd = 0, 0, 0, 0
        for i in range(len(profit_record)):
            if profit_record[i] > profit_record[former_high]:
                former_high = i
            cur_mdd = 1 - profit_record[i] / profit_record[former_high]
            if mdd < cur_mdd:
                mdd = cur_mdd
                mdd_high = former_high
                mdd_low = i

        recover_date, duration = None, None
        for i in range(mdd_high, len(profit_record)):
            if profit_record[i] > profit_record[mdd_high]:
                recover_date = i
                duration = recover_date - mdd_high
                break
        return mdd, mdd_high, mdd_low, recover_date, duration

    def info(self, dates, name):
        sharpe = self.get_sharpe()
        mdd, mdd_high, mdd_low, recover_date, duration = self.get_mdd()
        print('========================================')
        print('========================================')
        print('Display the profit of strategy', name)
        print('Annual Return:', self.ret_record.mean() * 252)
        print('Sharpe:', sharpe)
        print('Max drawdown:', mdd)
        print('Max drawdown start date:', dates[mdd_high])
        print('Max drawdown date:', dates[mdd_low])
        if recover_date is not None:
            print('Max drawdown recover date:', dates[recover_date])
            print('Max drawdown duration:', duration)
        else:
            print('Max drawdown till now')
        print('========================================')
        print('========================================')


class BasicStrategy(DescribeStrategy):
    def __init__(self, trans_fee, market, long):
        super().__init__()
        self.stocks = np.array([])
        self.prices = np.array([])
        self.trans_fee = trans_fee
        self.market = market
        self.long = long

        # Check the input of Basic Strategy
        assert self.market in ['A', 'H']
        assert isinstance(self.long, bool)

    # For each stock, design the stock selection logic according to the data
    # Customize in each strategy
    def stock_select(self, *args):
        pass

    # Given the new stocks to hold, calculate the trading signal to generate
    # Divided into 3 categories: to sale, to buy, and to hold
    def trade_signal(self, stocks):
        cur, aim = set(self.stocks), set(stocks)
        return list(cur - aim), list(aim - cur), list(aim & cur)

    def prices_select(self, CN_prices, HK_prices):
        if self.market == 'A':
            prices = CN_prices
        else:
            prices = HK_prices
        return prices

    # Calculate the profit from transaction
    def transaction(self, stocks, CN_prices, HK_prices):
        prices = self.prices_select(CN_prices, HK_prices)

        # Selection result shows that we should hold all the stocks and wait
        if stocks == 'keep':
            self.hold_profit(CN_prices, HK_prices)

        # Selection result shows that we maybe need some transactions
        else:
            prices = merge_2_prices(self.prices, prices)
            sale, buy, hold = self.trade_signal(stocks)
            sale_ret = prices[sale] / self.prices[sale] - 1 - self.trans_fee
            buy_ret = np.repeat(-self.trans_fee, len(buy))
            hold_ret = prices[hold] / self.prices[hold] - 1
            ret = (np.nansum(sale_ret) + np.nansum(buy_ret) + np.nansum(hold_ret)) / (len(sale)+len(buy)+len(hold))

            # If we are holding short positions, reverse the return
            if not self.long:
                ret = -ret

            # Update records
            new_profit = self.profit * (1 + ret)
            self.profit_record = np.append(self.profit_record, new_profit)
            self.trans_profit_record = np.append(self.trans_profit_record, new_profit)

            # Update properties
            # Store the profit of this transaction to calculate the next transaction return
            self.stocks = stocks
            self.prices = prices

    # Calculate the profit from holding all the remain stocks
    def hold_profit(self, CN_prices, HK_prices):
        prices = self.prices_select(CN_prices, HK_prices)
        prices = merge_2_prices(self.prices, prices)
        if len(self.stocks) > 0:
            hold_ret = prices[self.stocks] / self.prices[self.stocks] - 1
            ret = np.nansum(hold_ret) / len(self.stocks)
        else:
            ret = 0

        # If we are holding short positions, reverse the return
        if not self.long:
            ret = -ret

        # Update records
        new_profit = self.profit * (1 + ret)
        self.profit_record = np.append(self.profit_record, new_profit)

        # Update properties
        self.prices = prices


# Combine the strategies to describe their performance together
class CombineStrategy(DescribeStrategy):
    def __init__(self, Strategies, profit_merge):
        super().__init__()
        self.Strategies = Strategies
        self.profit_merge = profit_merge
        # Check input
        assert isinstance(self.profit_merge, bool)

    def record_merge(self):
        if self.profit_merge:
            # Merge the strategies through profits
            # This is trading mode
            res, res_trans = [], []
            for Strategy in self.Strategies:
                res.append(Strategy.profit_record)
                res_trans.append(Strategy.trans_profit_record)

            self.profit_record = np.array(res).mean(axis=0)
            self.ret_record = profit_2_ret(self.profit_record)

            self.trans_profit_record = np.array(res_trans).mean(axis=0)
            self.trans_ret_record = profit_2_ret(self.trans_profit_record)
        else:
            # Merge the strategies through returns
            # This is index mode
            res, res_trans = [], []
            for Strategy in self.Strategies:
                res.append(Strategy.ret_record)
                res_trans.append(Strategy.trans_ret_record)

            self.ret_record = np.array(res).mean(axis=0)
            self.profit_record = ret_2_profit(self.ret_record)

            self.trans_ret_record = np.array(res_trans).mean(axis=0)
            self.trans_profit_record = ret_2_profit(self.trans_ret_record)


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


def Simulate_Strategy(Strategy, trade_freq, CN_data, HK_data, AH_multiple):
    days, tickers = CN_data.shape
    halted = False
    for i in range(days):
        CN_data_info, HK_data_info, AH_info = CN_data[:i+1], HK_data[:i+1], AH_multiple[:i+1]
        CN_prices, HK_prices = CN_data[i], HK_data[i]

        # Time to decide transactions
        if halted or i % trade_freq == 0:
            selects = Strategy.stock_select(CN_data_info=CN_data_info, HK_data_info=HK_data_info, AH_info=AH_info)
            halted = (selects == 'keep')
            Strategy.transaction(selects, CN_prices, HK_prices)
        else:
            Strategy.hold_profit(CN_prices, HK_prices)
