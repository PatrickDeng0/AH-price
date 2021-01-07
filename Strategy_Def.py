import pickle
import Download, util
import numpy as np
import matplotlib.pyplot as plt


# A basic strategy object that define the describe statement of a common strategy
class DescribeStrategy(object):
    def __init__(self):
        # value_record: record everyday profit
        # trans_value_record: record interval returns between each two transactions
        self.value_record = np.array([])
        self.trans_value_record = np.array([])
        self.trans_date = np.array([], dtype=int)
        self.ret_record = np.array([])
        self.trans_ret_record = np.array([])

    # Get the newest profit value
    @property
    def value(self):
        if len(self.value_record) > 0:
            return self.value_record[-1]
        else:
            return 1

    @property
    def last_trans_value(self):
        if len(self.trans_value_record) > 0:
            return self.trans_value_record[-1]
        else:
            return 1

    def convert_ret(self):
        self.ret_record = util.value_2_ret(self.value_record)
        self.trans_ret_record = util.value_2_ret(self.trans_value_record)

    def get_sharpe(self):
        # If the transactions happen in a constant frequency, then compute its sharpe
        if len(self.trans_ret_record) > 10:
            return np.nanmean(self.trans_ret_record) / np.nanstd(self.trans_ret_record)

        # If there is no transactions at all, then compute the sharpe from daily return series
        # By default using monthly period as an epoch (22 trading days)
        else:
            period = 22
            cut_ret_record = self.ret_record[-(len(self.ret_record)//period)*period:]
            cut_ret_record = cut_ret_record.reshape((-1, period))
            merge_ret_record = np.nansum(cut_ret_record, axis=1)
            return np.nanmean(merge_ret_record) / np.nanstd(merge_ret_record)

    # Calculate the Max Drawdown and details of this strategy
    def get_mdd(self):
        former_high, mdd_high, mdd_low, mdd = 0, 0, 0, 0
        for i in range(len(self.value_record)):
            if self.value_record[i] > self.value_record[former_high]:
                former_high = i
            cur_mdd = 1 - self.value_record[i] / self.value_record[former_high]
            if mdd < cur_mdd:
                mdd = cur_mdd
                mdd_high = former_high
                mdd_low = i

        recover_date, duration = None, None
        for i in range(mdd_high, len(self.value_record)):
            if self.value_record[i] > self.value_record[mdd_high]:
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
        print('Annual Return:', np.nanmean(self.ret_record) * 252)
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
        self.positions = {}
        self.prices = np.array([])
        self.trans_fee = trans_fee
        self.market = market
        self.long = long

        # Check the input of Basic Strategy
        assert self.market in ['A', 'H']
        assert isinstance(self.long, bool)

    # For each stock, design the stock selection logic according to the data
    # Customize in each strategy
    def stock_select(self, *args) -> dict:
        pass

    # Given the new stocks to hold, calculate the trading signal to generate
    # Divided into 3 categories: to sale, to buy, and to hold
    def trade_signal(self, new_positions):
        epsilon = 0.001
        sale_dict, buy_dict, hold_dict = {}, {}, {}
        cur, aim = set(self.positions.keys()), set(new_positions.keys())
        all_stock = list(cur | aim)
        for stock in all_stock:
            cur_pos, aim_pos = self.positions.get(stock, 0), new_positions.get(stock, 0)
            chg = aim_pos - cur_pos
            if chg > epsilon:
                buy_dict[stock] = chg
            elif chg < -epsilon:
                sale_dict[stock] = -chg
            if cur_pos > 0 and aim_pos > 0:
                hold_dict[stock] = min(cur_pos, aim_pos)
        return sale_dict, buy_dict, hold_dict

    def prices_select(self, CN_prices, HK_prices):
        if self.market == 'A':
            prices = CN_prices
        else:
            prices = HK_prices
        return prices

    # Calculate the profit from transaction
    def transaction(self, positions, CN_prices, HK_prices, date):
        prices = self.prices_select(CN_prices, HK_prices)

        # Selection result shows that we should hold all the stocks and wait
        if positions == 'keep':
            self.hold_profit(CN_prices, HK_prices)

        # Selection result shows that we maybe need some transactions
        else:
            # Get position changes
            prices = util.merge_2_prices(self.prices, prices)
            sale_dict, buy_dict, hold_dict = self.trade_signal(positions)
            sale_stock, sale_weights = util.dict_2_array(sale_dict)
            buy_stock, buy_weights = util.dict_2_array(buy_dict)
            hold_stock, hold_weights = util.dict_2_array(hold_dict)

            # Get position change returns
            sale_ret = prices[sale_stock] / self.prices[sale_stock] - 1 - self.trans_fee
            buy_ret = np.repeat(-self.trans_fee, len(buy_stock))
            hold_ret = prices[hold_stock] / self.prices[hold_stock] - 1
            ret = np.dot(sale_ret, sale_weights) + np.dot(buy_ret, buy_weights) + np.dot(hold_ret, hold_weights)

            # If we are holding short positions, reverse the return
            if not self.long:
                ret = -ret

            # Update records
            new_profit = self.value * (1 + ret)
            self.value_record = np.append(self.value_record, new_profit)
            self.trans_value_record = np.append(self.trans_value_record, new_profit)
            self.trans_date = np.append(self.trans_date, date)

            # Update properties
            # Store the profit of this transaction to calculate the next transaction return
            self.positions = positions
            self.prices = prices

    # Calculate the profit from holding all the remain stocks
    def hold_profit(self, CN_prices, HK_prices):
        prices = self.prices_select(CN_prices, HK_prices)
        prices = util.merge_2_prices(self.prices, prices)
        hold_stock, hold_weights = util.dict_2_array(self.positions)
        if len(hold_stock) > 0:
            hold_ret = prices[hold_stock] / self.prices[hold_stock] - 1
            ret = np.dot(hold_ret, hold_weights)
        else:
            ret = 0

        # If we are holding short positions, reverse the return
        if not self.long:
            ret = -ret

        # Update records
        new_profit = self.value * (1 + ret)
        self.value_record = np.append(self.value_record, new_profit)

        # Update properties
        self.prices = prices


# The Strategy that track an INDEX!
class IndexStrategy(BasicStrategy):
    def __init__(self, trans_fee, market, long, profit):
        super().__init__(trans_fee, market, long)

        # Long an index, simply record its value
        if self.long:
            self.value_record = profit
        # Short an index, convert its value
        else:
            self.value_record = 1/profit
        self.ret_record = util.value_2_ret(self.value_record)

    def transaction(self, *args):
        pass

    def hold_profit(self, *args):
        pass

    def stock_select(self, *args):
        pass

    def convert_ret(self):
        pass

    # For IndexStrategy, we didn't record transaction
    # For compute the profit between transactions, we set its profit series manually
    def set_trans_dates(self, trans_date):
        self.trans_date = trans_date
        self.trans_value_record = self.value_record[trans_date]
        self.trans_ret_record = util.value_2_ret(self.trans_value_record)


# Combine the strategies to describe their performance together
class CombineStrategy(DescribeStrategy):
    def __init__(self, Strategies, profit_merge):
        super().__init__()
        self.Strategies = Strategies
        self.profit_merge = profit_merge
        # Check input
        assert isinstance(self.profit_merge, bool)

    def record_merge(self):
        # Set the transactions days for this CombinaStrategy
        # Especially when there is any index strategy in the combination
        for Strategy in self.Strategies:
            if len(Strategy.trans_date) > 0:
                self.trans_date = Strategy.trans_date
                break
        if len(self.trans_date) > 0:
            for Strategy in self.Strategies:
                # If a strategy is an Index Strategy, then we need to set its trans_dates in advance
                if isinstance(Strategy, IndexStrategy):
                    Strategy.set_trans_dates(self.trans_date)

        if self.profit_merge:
            # Merge the strategies through profits
            # This is trading mode
            res, res_trans = [], []
            for Strategy in self.Strategies:
                res.append(Strategy.value_record)
                res_trans.append(Strategy.trans_value_record)

            self.value_record = np.mean(np.array(res), axis=0)
            self.trans_value_record = np.mean(np.array(res_trans), axis=0)
            self.convert_ret()

        else:
            # Merge the strategies through returns
            # This is index mode
            res, res_trans = [], []
            for Strategy in self.Strategies:
                res.append(Strategy.ret_record)
                res_trans.append(Strategy.trans_ret_record)

            self.ret_record = np.nanmean(np.array(res), axis=0)
            self.value_record = util.ret_2_value(self.ret_record)

            self.trans_ret_record = np.nanmean(np.array(res_trans), axis=0)
            self.trans_value_record = util.ret_2_value(self.trans_ret_record)


def Simulate_Strategy(Strats, trade_freq, CN_data, HK_data, AH_multiple):
    days, tickers = CN_data.shape
    if isinstance(Strats, CombineStrategy):
        Strats_list = Strats.Strategies
    else:
        Strats_list = [Strats]

    halted = [False]*len(Strats_list)

    for i in range(days):
        CN_data_info, HK_data_info, AH_info = CN_data[:i+1], HK_data[:i+1], AH_multiple[:i+1]
        CN_prices, HK_prices = CN_data[i], HK_data[i]

        for num in range(len(Strats_list)):
            Strat = Strats_list[num]

            # Time to decide transactions
            if halted[num] or i % trade_freq == 0:
                selects = Strat.stock_select(CN_data_info, HK_data_info, AH_info)
                halted[num] = (selects == 'keep')
                Strat.transaction(selects, CN_prices, HK_prices, i)
            else:
                Strat.hold_profit(CN_prices, HK_prices)

    for Strat in Strats_list:
        Strat.convert_ret()

    if isinstance(Strats, CombineStrategy):
        Strats.record_merge()
