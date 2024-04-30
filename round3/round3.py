import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import numpy as np
import pandas as pd
import collections


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

    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'ROSES': 0, 'STRAWBERRIES': 0, 'GIFT_BASKET': 0}
    pos_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'ROSES': 60, 'STRAWBERRIES': 350, 'GIFT_BASKET': 60}
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
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', max(next(reversed(obuy['GIFT_BASKET'])),round(mid_price['GIFT_BASKET'] - 30)), round(-result * vol)))
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
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(next(reversed(osell['GIFT_BASKET'])), round(mid_price['GIFT_BASKET'] + 30)), round(result * vol)))
                # orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(next(reversed(osell['GIFT_BASKET'])), round(mid_price['GIFT_BASKET'] + 30)), vol))


        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        for key in self.position.keys():
            self.position[key] = 0
        for key, val in state.position.items():
            self.position[key] = val

        orders = {}
        conversions = 0
        trader_data = ""

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
                diff_price = max(1.02, self.prev_price + (conversions - 50) * 0.001)
                self.prev_price = diff_price
                logger.print(
                    f"Diff Price: {diff_price}")

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

                orders[product].append(Order(product, round(final_ask + diff_price), -self.pos_limit[product] -cpos - conversions))
                orders[product].append(Order(product, round(final_bid - diff_price), self.pos_limit[product] - cpos - conversions))
                
            # elif product == 'CHOCOLATE':
            #     orders[product] = []

            #     # predict price coefficient
            #     Coefficients = [-0.2056737588652483, -0.05673758865248235, 0.09219858156028365,
            #                     0.2411347517730496, 0.3900709219858156, 0.5390070921985816]

            #     buy_list = list(order_depth.sell_orders.items())
            #     sell_list = list(order_depth.buy_orders.items())
            #     buy_sum = 0
            #     buy_volume = 0
            #     for i in range(len(buy_list)):
            #         buy_sum += buy_list[i][0] * abs(buy_list[i][1])
            #         buy_volume += abs(buy_list[i][1])
            #     sell_sum = 0
            #     sell_volume = 0
            #     for i in range(len(sell_list)):
            #         sell_sum += sell_list[i][0] * abs(sell_list[i][1])
            #         sell_volume += abs(sell_list[i][1])
            #     midprice = (buy_sum + sell_sum) / (buy_volume + sell_volume)

            #     # add more data if data is less than 6
            #     if len(self.chocolate_ema) < 20:
            #         self.chocolate.append(midprice)
            #         self.chocolate_ema.append(midprice)
            #     else:
            #         # expected price
            #         self.chocolate.append(midprice)
            #         self.chocolate_ema.append(midprice)
            #         self.chocolate.pop(0)
            #         # price = round(
            #         #     np.dot(np.transpose(Coefficients), np.array(self.chocolate)))
                    
            #         price_list = pd.Series(self.chocolate_ema)
            #         price1 = price_list.rolling(window=5).mean()
            #         price2 = price_list.rolling(window=20).mean()
            #         buy_signals = (price1.iloc[-1] > price2.iloc[-1]) and (price1.iloc[-2] <= price2.iloc[-2])
            #         sell_signals = (price1.iloc[-1] < price2.iloc[-1]) and (price1.iloc[-2] >= price2.iloc[-2])
            #         logger.print(f"Buy Signals: {buy_signals}, sell signals: {sell_signals}")
            #         if buy_signals:
            #             cpos = self.position[product]
            #             if self.pos_limit[product] - cpos != 0:
            #                 orders[product].append(Order(product, buy_list[0][0], self.pos_limit[product] - cpos))
            #             # for ask, vol in buy_list:
            #             #     order_for = min(-vol,
            #             #                         self.pos_limit[product] - cpos)
            #             #     cpos += order_for
            #             #     assert (order_for >= 0)
            #             #     orders[product].append(
            #             #         Order(product, ask, order_for))
            #         # price_list = price_list.ewm(com=0.3).mean()
            #         # price = price_list.iloc[-1]

            #         # cpos = self.position[product]
            #         # for ask, vol in buy_list:
            #         #     if (ask <= price - 1) and cpos < self.pos_limit[product]:
            #         #         order_for = min(-vol,
            #         #                         self.pos_limit[product] - cpos)
            #         #         cpos += order_for
            #         #         assert (order_for >= 0)
            #         #         orders[product].append(
            #         #             Order(product, ask, order_for))

            #         # bid_pr_tmp, _ = max(sell_list, key=lambda x: x[1])
            #         # bid_pr = min(bid_pr_tmp + 1, price - 1)
                    
            #         # if self.pos_limit[product] - cpos != 0:
            #         #     orders[product].append(Order(product, round(bid_pr), self.pos_limit[product] - cpos))
                    
            #         if sell_signals:
            #             cpos = self.position[product]
            #             if -self.pos_limit[product]-cpos != 0:
            #                 orders[product].append(Order(product, sell_list[0][0], -self.pos_limit[product]-cpos))
            #             # for bid, vol in sell_list:
            #             #     order_for = max(-vol, -self.pos_limit[product]-cpos)
            #             #     cpos += order_for
            #             #     assert (order_for <= 0)
            #             #     orders[product].append(
            #             #         Order(product, bid, order_for))
            #         # cpos = self.position[product]
            #         # for bid, vol in sell_list:
            #         #     if (bid >= price + 1) and cpos > -self.pos_limit[product]:
            #         #         order_for = max(-vol, -
            #         #                         self.pos_limit[product]-cpos)
            #         #         cpos += order_for
            #         #         assert (order_for <= 0)
            #         #         orders[product].append(
            #         #             Order(product, bid, order_for))

            #         # sell_pr_tmp, _ = min(buy_list, key=lambda x: x[1])
            #         # sell_pr = max(sell_pr_tmp - 1, price + 1)

            #         # if -self.pos_limit[product]-cpos != 0:
            #         #     orders[product].append(Order(product, round(sell_pr), -self.pos_limit[product]-cpos))
                    
            # elif product == 'STRAWBERRIES':
            #     orders[product] = []

            #     # predict price coefficient
            #     Coefficients = [-0.2056737588652483, -0.05673758865248235, 0.09219858156028365,
            #                     0.2411347517730496, 0.3900709219858156, 0.5390070921985816]

            #     buy_list = list(order_depth.sell_orders.items())
            #     sell_list = list(order_depth.buy_orders.items())
            #     buy_sum = 0
            #     buy_volume = 0
            #     for i in range(len(buy_list)):
            #         buy_sum += buy_list[i][0] * abs(buy_list[i][1])
            #         buy_volume += abs(buy_list[i][1])
            #     sell_sum = 0
            #     sell_volume = 0
            #     for i in range(len(sell_list)):
            #         sell_sum += sell_list[i][0] * abs(sell_list[i][1])
            #         sell_volume += abs(sell_list[i][1])
            #     midprice = (buy_sum + sell_sum) / (buy_volume + sell_volume)

            #     # add more data if data is less than 6
            #     if len(self.strawberries) < 6:
            #         self.strawberries.append(midprice)
            #         self.strawberries_ema.append(midprice)
            #     else:
            #         # expected price
            #         self.strawberries.append(midprice)
            #         self.strawberries.pop(0)
            #         self.strawberries_ema.append(midprice)
            #         price = round(
            #             np.dot(np.transpose(Coefficients), np.array(self.strawberries)))

            #         cpos = self.position[product]
            #         for ask, vol in buy_list:
            #             if (ask <= price - 1) and cpos < self.pos_limit[product]:
            #                 order_for = min(-vol,
            #                                 self.pos_limit[product] - cpos)
            #                 cpos += order_for
            #                 assert (order_for >= 0)
            #                 orders[product].append(
            #                     Order(product, ask, order_for))

            #         bid_pr_tmp, _ = max(sell_list, key=lambda x: x[1])
            #         bid_pr = min(bid_pr_tmp + 1, price - 1)

            #         # if self.pos_limit[product] - cpos != 0:
            #         #     orders[product].append(Order(product, round(bid_pr), self.pos_limit[product] - cpos))

            #         cpos = self.position[product]
            #         for bid, vol in sell_list:
            #             if (bid >= price + 1) and cpos > -self.pos_limit[product]:
            #                 order_for = max(-vol, -
            #                                 self.pos_limit[product]-cpos)
            #                 cpos += order_for
            #                 assert (order_for <= 0)
            #                 orders[product].append(
            #                     Order(product, bid, order_for))

            #         sell_pr_tmp, _ = min(buy_list, key=lambda x: x[1])
            #         sell_pr = max(sell_pr_tmp - 1, price + 1)

            #         # if -self.pos_limit[product]-cpos != 0:
            #         #     orders[product].append(Order(product, round(sell_pr), -self.pos_limit[product]-cpos))
                    
            # elif product == 'ROSES':
            #     orders[product] = []

            #     # predict price coefficient
            #     Coefficients = [-0.2056737588652483, -0.05673758865248235, 0.09219858156028365,
            #                     0.2411347517730496, 0.3900709219858156, 0.5390070921985816]

            #     buy_list = list(order_depth.sell_orders.items())
            #     sell_list = list(order_depth.buy_orders.items())
            #     buy_sum = 0
            #     buy_volume = 0
            #     for i in range(len(buy_list)):
            #         buy_sum += buy_list[i][0] * abs(buy_list[i][1])
            #         buy_volume += abs(buy_list[i][1])
            #     sell_sum = 0
            #     sell_volume = 0
            #     for i in range(len(sell_list)):
            #         sell_sum += sell_list[i][0] * abs(sell_list[i][1])
            #         sell_volume += abs(sell_list[i][1])
            #     midprice = (buy_sum + sell_sum) / (buy_volume + sell_volume)

            #     # add more data if data is less than 6
            #     if len(self.roses) < 6:
            #         self.roses.append(midprice)
            #         self.roses_ema.append(midprice)
            #     else:
            #         # expected price
            #         self.roses.append(midprice)
            #         self.roses.pop(0)
            #         self.roses_ema.append(midprice)
            #         price = round(
            #             np.dot(np.transpose(Coefficients), np.array(self.roses)))
            #         # price_list = pd.Series(self.roses_ema)
            #         # price_list = price_list.ewm(com=0.25).mean()
            #         # price = price_list.iloc[-1]

            #         cpos = self.position[product]
            #         for ask, vol in buy_list:
            #             if (ask <= price - 0.2) and cpos < self.pos_limit[product]:
            #                 order_for = min(-vol,
            #                                 self.pos_limit[product] - cpos)
            #                 cpos += order_for
            #                 assert (order_for >= 0)
            #                 orders[product].append(
            #                     Order(product, ask, order_for))

            #         bid_pr_tmp, _ = max(sell_list, key=lambda x: x[1])
            #         bid_pr = min(bid_pr_tmp + 1, price - 1)

            #         # if self.pos_limit[product] - cpos != 0:
            #         #     orders[product].append(Order(product, round(bid_pr), self.pos_limit[product] - cpos))

            #         cpos = self.position[product]
            #         for bid, vol in sell_list:
            #             if (bid >= price + 0.2) and cpos > -self.pos_limit[product]:
            #                 order_for = max(-vol, -
            #                                 self.pos_limit[product]-cpos)
            #                 cpos += order_for
            #                 assert (order_for <= 0)
            #                 orders[product].append(
            #                     Order(product, bid, order_for))

            #         sell_pr_tmp, _ = min(buy_list, key=lambda x: x[1])
            #         sell_pr = max(sell_pr_tmp - 1, price + 1)

            #         # if -self.pos_limit[product]-cpos != 0:
            #         #     orders[product].append(Order(product, round(sell_pr), -self.pos_limit[product]-cpos))
                
        orders_round3 = self.compute_orders_basket(state.order_depths)
        orders['GIFT_BASKET'] = orders_round3['GIFT_BASKET']
        # orders['CHOCOLATE'] = orders_round3['CHOCOLATE']
        # orders['ROSES'] = orders_round3['ROSES']
        # orders['STRAWBERRIES'] = orders_round3['STRAWBERRIES']

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data