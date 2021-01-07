import numpy as np
import matplotlib.pyplot as plt


def ret_2_value(rets):
    return np.nancumprod(rets+1)


def value_2_ret(value):
    add_value = np.insert(value, 0, 1)
    profit_chg = np.diff(add_value)
    return profit_chg / add_value[:-1]


def dict_2_array(dict):
    return np.array(list(dict.keys())).astype(int), np.array(list(dict.values()))


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


def get_beta_alpha(strat, index_strat, strat_name, index_name, neutral):
    strat_ret = strat.ret_record
    index_ret = index_strat.ret_record
    if neutral:
        index_ret = -index_ret
    if hasattr(strat, 'MA_window'):
        strat_ret = strat_ret[strat.MA_window:]
        index_ret = index_ret[strat.MA_window:]
    valid_loca = np.where(~np.isnan(index_ret) & ~(np.isnan(strat_ret)))
    cut_strat_ret, cut_index_ret = strat_ret[valid_loca], index_ret[valid_loca]

    polys = np.polyfit(cut_index_ret, cut_strat_ret, 1)
    xp = np.linspace(np.min(cut_index_ret), np.max(cut_index_ret), 200)
    p = np.poly1d(polys)
    plt.plot(xp, p(xp), color='r')
    plt.scatter(cut_index_ret, cut_strat_ret)
    plt.title('Beta %.2f Alpha %.5f of %s -- %s' % (round(polys[0],2), round(polys[1],5), strat_name, index_name))
    plt.show()


def get_weights(cur, info):
    pass
