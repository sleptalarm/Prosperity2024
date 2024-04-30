import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import numpy as np
import pandas as pd
import collections
from math import erf, floor



class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(
                state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


class Trader:

    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'ROSES': 0, 'STRAWBERRIES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
    pos_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'ROSES': 60, 'STRAWBERRIES': 350, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    starfruit = []
    chocolate = []
    chocolate_ema = []
    orchids = []
    strawberries = []
    strawberries_ema = []
    roses = []
    roses_ema = []
    prev_price = 2
    basket_std = 160


    def cal_actual_price(self, S, time):
        vol = 0.0101193
        logger.print(f"Time: {time}")
        T = 246 - time / 1000000
        r = 0
        K = 10000
        d1 = (np.log(S/K) + (r+vol**2/2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)

        normcdf = lambda x: (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        N1 = normcdf(d1)
        N2 = normcdf(d2)

        px = S*N1 - K*np.exp((-r)*T)*N2

        return px, N1


    def add_order(self, product, order_depth, orders, price):
        orders[product] = []

        if len(order_depth.sell_orders) != 0:
            count1 = 0
            tmp_list1 = list(order_depth.sell_orders.items())
            for i in range(len(tmp_list1)):
                price, amount = tmp_list1[i]
                if int(price) <= price:
                    limit = self.pos_limit[product] - \
                        self.position[product]
                    count1 += min(-amount, limit)
                    orders[product].append(
                        Order(product, price, min(-amount, limit)))
                else:
                    break

        if len(order_depth.buy_orders) != 0:
            count2 = 0
            tmp_list2 = list(order_depth.buy_orders.items())
            for i in range(len(tmp_list2)):
                price, amount = tmp_list2[i]
                if int(price) >= price:
                    limit = self.pos_limit[product] + \
                        self.position[product]
                    count2 += max(-amount, -limit)
                    orders[product].append(
                        Order(product, price, max(-amount, -limit)))
                else:
                    break

        if tmp_list1[0][0] - 1 > 10000:
            orders[product].append(Order(
                product, tmp_list1[0][0] - 1, -(self.pos_limit[product] + self.position[product] + count2)))

        if tmp_list2[0][0] + 1 < 10000:
            orders[product].append(Order(
                product, tmp_list2[0][0] + 1, self.pos_limit[product] - self.position[product] - count1))
            
        return orders

    def compute_orders_basket(self, order_depth):

        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, mid_price= {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(order_depth[p].sell_orders.items())
            obuy[p] = collections.OrderedDict(order_depth[p].buy_orders.items())

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2

        for p in prods:
            buy_list = list(osell[p].items())
            sell_list = list(obuy[p].items())
            buy_sum = 0
            buy_volume = 0
            for i in range(len(buy_list)):
                buy_sum += buy_list[i][0] * abs(buy_list[i][1])
                buy_volume += abs(buy_list[i][1])
            sell_sum = 0
            sell_volume = 0
            for i in range(len(sell_list)):
                sell_sum += sell_list[i][0] * abs(sell_list[i][1])
                sell_volume += abs(sell_list[i][1])
            mid_price[p] = (buy_sum + sell_sum) / (buy_volume + sell_volume)

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - mid_price['STRAWBERRIES'] * 6 - 380
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - mid_price['STRAWBERRIES'] * 6 - 380
        logger.print(f"Price diff {res_buy}")
        trade_at = 30


        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.pos_limit['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
            #     for price, val in obuy['GIFT_BASKET'].items():
            #         if val > vol:
            #             orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -val))
            #             vol -= val
            #             # break
            #         else:
            #             orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -vol))
            #             break
                # orders['GIFT_BASKET'].append(Order('GIFT_BASKET', next(reversed(obuy['GIFT_BASKET'])), -vol)) 
                final_val = (res_sell - trade_at)
                exp_growth = 1 - np.exp(-0.03 * final_val)
                result = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', max(next(reversed(obuy['GIFT_BASKET'])),floor(mid_price['GIFT_BASKET'] - 30)), floor(-result * vol)))
                # orders['GIFT_BASKET'].append(Order('GIFT_BASKET', max(next(reversed(obuy['GIFT_BASKET'])),round(mid_price['GIFT_BASKET'] - 30)), -vol)) 
        elif res_buy < -trade_at:
            vol = self.pos_limit['GIFT_BASKET'] - self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                # for price, val in osell['GIFT_BASKET'].items():
                #     if val > vol:
                #         orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, val))
                #         vol -= val
                #         # break
                #     else:
                #         orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, vol))
                #         break
                # orders['GIFT_BASKET'].append(Order('GIFT_BASKET', next(reversed(osell['GIFT_BASKET'])), vol))
                final_val = trade_at - res_buy
                exp_growth = 1 - np.exp(-0.03 * final_val)
                result = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(next(reversed(osell['GIFT_BASKET'])), floor(mid_price['GIFT_BASKET'] + 30)), floor(result * vol)))
                # orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(next(reversed(osell['GIFT_BASKET'])), round(mid_price['GIFT_BASKET'] + 30)), vol))


        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        for key in state.own_trades.keys():
            logger.print(f"Own Trades: {key}, {state.own_trades[key]}")

        for key in self.position.keys():
            self.position[key] = 0
        for key, val in state.position.items():
            self.position[key] = val

        orders = {}
        conversions = 0
        trader_data = ""
        time = int(state.timestamp)

        for product, order_depth in state.order_depths.items():
            if product == 'AMETHYSTS':
                orders[product] = []

                acceptable_price_buy = 10000
                acceptable_price_sell = 10000

                if len(order_depth.sell_orders) != 0:
                    count1 = 0
                    tmp_list1 = list(order_depth.sell_orders.items())
                    for i in range(len(tmp_list1)):
                        price, amount = tmp_list1[i]
                        if int(price) <= acceptable_price_buy:
                            limit = self.pos_limit[product] - \
                                self.position[product]
                            count1 += min(-amount, limit)
                            orders[product].append(
                                Order(product, price, min(-amount, limit)))
                        else:
                            break

                if len(order_depth.buy_orders) != 0:
                    count2 = 0
                    tmp_list2 = list(order_depth.buy_orders.items())
                    for i in range(len(tmp_list2)):
                        price, amount = tmp_list2[i]
                        if int(price) >= acceptable_price_sell:
                            limit = self.pos_limit[product] + \
                                self.position[product]
                            count2 += max(-amount, -limit)
                            orders[product].append(
                                Order(product, price, max(-amount, -limit)))
                        else:
                            break

                if tmp_list1[0][0] - 1 > 10000:
                    orders[product].append(Order(
                        product, tmp_list1[0][0] - 1, -(self.pos_limit[product] + self.position[product] + count2)))

                if tmp_list2[0][0] + 1 < 10000:
                    orders[product].append(Order(
                        product, tmp_list2[0][0] + 1, self.pos_limit[product] - self.position[product] - count1))

            elif product == 'STARFRUIT':
                orders[product] = []

                # predict price coefficient
                Coefficients = [-0.2056737588652483, -0.05673758865248235, 0.09219858156028365,
                                0.2411347517730496, 0.3900709219858156, 0.5390070921985816]

                buy_list = list(order_depth.sell_orders.items())
                sell_list = list(order_depth.buy_orders.items())
                buy_sum = 0
                buy_volume = 0
                for i in range(len(buy_list)):
                    buy_sum += buy_list[i][0] * abs(buy_list[i][1])
                    buy_volume += abs(buy_list[i][1])
                sell_sum = 0
                sell_volume = 0
                for i in range(len(sell_list)):
                    sell_sum += sell_list[i][0] * abs(sell_list[i][1])
                    sell_volume += abs(sell_list[i][1])
                midprice = (buy_sum + sell_sum) / (buy_volume + sell_volume)

                # add more data if data is less than 6
                if len(self.starfruit) < 6:
                    self.starfruit.append(midprice)
                else:
                    # expected price
                    self.starfruit.append(midprice)
                    self.starfruit.pop(0)
                    price = round(
                        np.dot(np.transpose(Coefficients), np.array(self.starfruit)))

                    cpos = self.position[product]
                    for ask, vol in buy_list:
                        if ask <= price - 0.05 * self.position[product] - 0.2 and cpos < self.pos_limit[product]:
                            order_for = min(-vol,
                                            self.pos_limit[product] - cpos)
                            cpos += order_for
                            assert (order_for >= 0)
                            orders[product].append(
                                Order(product, ask, order_for))

                    bid_pr_tmp, _ = max(sell_list, key=lambda x: x[1])
                    bid_pr = min(bid_pr_tmp + 1, price)

                    orders[product].append(
                        Order(product, bid_pr, self.pos_limit[product] - cpos))

                    cpos = self.position[product]
                    for bid, vol in sell_list:
                        if bid >= price - 0.05 * self.position[product] + 0.2 and cpos > -self.pos_limit[product]:
                            order_for = max(-vol, -
                                            self.pos_limit[product]-cpos)
                            cpos += order_for
                            assert (order_for <= 0)
                            orders[product].append(
                                Order(product, bid, order_for))

                    sell_pr_tmp, _ = min(buy_list, key=lambda x: x[1])
                    sell_pr = max(sell_pr_tmp - 1, price + 1)

                    orders[product].append(
                        Order(product, sell_pr, -self.pos_limit[product]-cpos))

            elif product == 'ORCHIDS':
                conversions = -self.position[product]
                orders[product] = []
                
                # diff_price = min(max(1, self.prev_price + (conversions - 50) * 0.006),2)
                
                
                ob_info = state.observations.conversionObservations[product]
                storage_cost = 0.1
                
                
                final_bid = ob_info.bidPrice - ob_info.transportFees - ob_info.exportTariff - storage_cost
                final_ask = ob_info.askPrice + ob_info.transportFees + ob_info.importTariff
                cpos = self.position[product]
                diff_price = max(1, self.prev_price + (conversions - 50) * 0.001)
                self.prev_price = diff_price
                logger.print(
                    f"Diff Price: {diff_price}")
                
                orders[product].append(Order(product, round(final_ask + diff_price), -self.pos_limit[product] -cpos - conversions))
                orders[product].append(Order(product, round(final_bid - diff_price), self.pos_limit[product] - cpos - conversions))

                # if self.position[product] < 0:
                #     for ask, vol in order_depth.sell_orders.items():
                #         if ask < final_ask:
                #             order_for = min(-vol,
                #                             self.pos_limit[product] - cpos)
                #             cpos += order_for
                #             if order_for > 0 :
                #                 orders[product].append(
                #                     Order(product, ask, order_for))
                
                # elif self.position[product] > 0:
                #     for bid, vol in order_depth.buy_orders.items():
                #         if bid > final_bid:
                #             order_for = max(-vol, -
                #                             self.pos_limit[product]-cpos)
                #             cpos += order_for
                #             if order_for < 0:
                #                 orders[product].append(
                #                     Order(product, bid, order_for))   
                # # conversions = -cpos

                # orders[product].append(Order(product, round(final_ask + 2.06), -self.pos_limit[product] - cpos - conversions))
                # orders[product].append(Order(product, round(final_bid - 2.06), self.pos_limit[product] - cpos - conversions))

                
            

            elif product == 'COCONUT':
                orders[product] = []

                if len(order_depth.sell_orders) != 0:
                    can_buy_coconut = True
                    count1 = 0
                    buy_coconut = list(order_depth.sell_orders.items())
                    buy_sum = 0
                    buy_volume = 0
                    for i in range(len(buy_coconut)):
                        buy_sum += buy_coconut[i][0] * abs(buy_coconut[i][1])
                        buy_volume += abs(buy_coconut[i][1])
                else:
                    can_buy_coconut = False

                if len(order_depth.buy_orders) != 0:
                    can_sell_coconut = True
                    count2 = 0
                    sell_coconut = list(order_depth.buy_orders.items())
                    sell_sum = 0
                    sell_volume = 0
                    for i in range(len(sell_coconut)):
                        sell_sum += sell_coconut[i][0] * abs(sell_coconut[i][1])
                        sell_volume += abs(sell_coconut[i][1])
                    midprice_coconut = (buy_sum + sell_sum) / (buy_volume + sell_volume)
                else:
                    can_sell_coconut = False
                
                

            elif product == 'COCONUT_COUPON':
                orders[product] = []

                if len(order_depth.sell_orders) != 0:
                    can_buy_coconut_coupon = True
                    buy_coconut_coupon = list(order_depth.sell_orders.items())
                    buy_sum = 0
                    buy_volume = 0
                    for i in range(len(buy_coconut_coupon)):
                        buy_sum += buy_coconut_coupon[i][0] * abs(buy_coconut_coupon[i][1])
                        buy_volume += abs(buy_coconut_coupon[i][1])
                else:
                    can_buy_coconut_coupon = False

                if len(order_depth.buy_orders) != 0:
                    can_sell_coconut_coupon = True
                    sell_coconut_coupon = list(order_depth.buy_orders.items())
                    sell_sum = 0
                    sell_volume = 0
                    for i in range(len(sell_coconut_coupon)):
                        sell_sum += sell_coconut_coupon[i][0] * abs(sell_coconut_coupon[i][1])
                        sell_volume += abs(sell_coconut_coupon[i][1])
                else:
                    can_sell_coconut_coupon = False
                
                
                midprice_coconut_coupon = (buy_sum + sell_sum) / (buy_volume + sell_volume)

        fair_price_coconut_coupon, coef = self.cal_actual_price(S = midprice_coconut, time = time)


        # buy coconut coupon, sell coconut
        if fair_price_coconut_coupon > midprice_coconut_coupon:
            num_buy_coconut_coupon = floor(min(self.pos_limit['COCONUT_COUPON'] - self.position['COCONUT_COUPON'], (fair_price_coconut_coupon - midprice_coconut_coupon) * 15))
            num_sell_coconut = floor(max(-self.pos_limit['COCONUT'] - self.position['COCONUT'], -num_buy_coconut_coupon * coef))
            # orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', floor(midprice_coconut_coupon), num_buy_coconut_coupon))
            # orders['COCONUT'].append(Order('COCONUT', floor(midprice_coconut) + 1, num_sell_coconut))
            if can_buy_coconut_coupon:
                for price, vol in buy_coconut_coupon:
                    order_for = min(-vol, num_buy_coconut_coupon)
                    num_buy_coconut_coupon -= order_for
                    if order_for > 0:
                        orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', price, order_for))
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', buy_coconut_coupon[0][0] if can_buy_coconut_coupon else round(midprice_coconut_coupon), num_buy_coconut_coupon))
            if can_sell_coconut:
                for price, vol in sell_coconut:
                    order_for = max(-vol, num_sell_coconut)
                    num_sell_coconut -= order_for
                    if order_for < 0:
                        orders['COCONUT'].append(Order('COCONUT', price, order_for))
            orders['COCONUT'].append(Order('COCONUT', sell_coconut[0][0] if can_sell_coconut else round(midprice_coconut), num_sell_coconut))

        # sell coconut coupon, buy coconut
        if fair_price_coconut_coupon < midprice_coconut_coupon - 5:
            num_sell_coconut_coupon = floor(max(-self.pos_limit['COCONUT_COUPON'] - self.position['COCONUT_COUPON'], (fair_price_coconut_coupon - midprice_coconut_coupon) * 15))
            num_buy_coconut = floor(min(self.pos_limit['COCONUT'] - self.position['COCONUT'], round(-num_sell_coconut_coupon * coef)))
            # orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', floor(midprice_coconut), num_sell_coconut_coupon))
            # orders['COCONUT'].append(Order('COCONUT', floor(midprice_coconut_coupon) + 1, num_buy_coconut))
            if can_sell_coconut_coupon:
                for price, vol in sell_coconut_coupon:
                    order_for = max(-vol, num_sell_coconut_coupon)
                    num_sell_coconut_coupon -= order_for
                    if order_for < 0:
                        orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', price, order_for))
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', sell_coconut_coupon[0][0] if can_sell_coconut_coupon else round(midprice_coconut_coupon), num_sell_coconut_coupon))
            if can_buy_coconut:
                for price, vol in buy_coconut:
                    order_for = min(-vol, num_buy_coconut)
                    num_buy_coconut -= order_for
                    if order_for > 0:
                        orders['COCONUT'].append(Order('COCONUT', price, order_for))
            orders['COCONUT'].append(Order('COCONUT', buy_coconut[0][0] if can_buy_coconut else round(midprice_coconut), num_buy_coconut))

        # logger.print(f"fair_price_coconut_coupon: {fair_price_coconut_coupon}") 
        # logger.print(f"diff: {fair_price_coconut_coupon - midprice_coconut_coupon}")
        # logger.print(f"coef: {coef}")

        # orders_round3 = self.compute_orders_basket(state.order_depths)
        # orders['GIFT_BASKET'] = orders_round3['GIFT_BASKET']


        

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data