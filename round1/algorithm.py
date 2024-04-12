import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import numpy as np
from collections import Counter


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

    position = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    pos_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    starfruit = []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        for key in self.position.keys():
            self.position[key] = 0
        for key, val in state.position.items():
            logger.print(f"Position for {key}: {val}")
            self.position[key] = val

        orders = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            print(product)
            if product == 'AMETHYSTS':
                orders[product] = []

                acceptable_price_buy = 10000
                acceptable_price_sell = 10000

                logger.print(
                    f"Acceptable price for {product}: {acceptable_price_buy}")
                logger.print(
                    f"Acceptable price for {product}: {acceptable_price_sell}")

                if len(order_depth.sell_orders) != 0:
                    count1 = 0
                    tmp_list1 = list(order_depth.sell_orders.items())
                    for i in range(len(tmp_list1)):
                        price, amount = tmp_list1[i]
                        if int(price) <= acceptable_price_buy:
                            logger.print("BUY", str(-amount) + "x", price)
                            limit = self.pos_limit[product] - \
                                self.position[product]
                            if self.position[product] == 20:
                                logger.print(f"Can't buy {product}")
                            count1 += min(-amount, limit)
                            logger.print(f"count1: {count1}")
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
                            logger.print("SELL", str(amount) + "x", price)
                            limit = self.pos_limit[product] + \
                                self.position[product]
                            if self.position[product] == -20:
                                logger.print(f"Can't Sell {product}")
                            count2 += max(-amount, -limit)
                            logger.print(f"count2: {count2}")
                            orders[product].append(
                                Order(product, price, max(-amount, -limit)))
                        else:
                            break

                logger.print("sell list", tmp_list1)
                if tmp_list1[0][0] - 1 > 10000:
                    orders[product].append(Order(
                        product, tmp_list1[0][0] - 1, -(self.pos_limit[product] + self.position[product] + count2)))
                logger.print("buy list", tmp_list2)
                if tmp_list2[0][0] + 1 < 10000:
                    orders[product].append(Order(
                        product, tmp_list2[0][0] + 1, self.pos_limit[product] - self.position[product] - count1))

            elif product == 'STARFRUIT':
                orders[product] = []
                if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                    logger.print("No orders for STARFRUIT")

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
                    logger.print(f"Predicted price for STARFRUIT: {price}")

                    cpos = self.position[product]

                    for ask, vol in buy_list:
                        if (ask <= price - 0.05 * self.position[product] - 0.2) and cpos < self.pos_limit[product]:
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
                    logger.print(
                        f"buy added {product}, {bid_pr}, {self.pos_limit[product] - cpos}")

                    cpos = self.position[product]
                    for bid, vol in sell_list:
                        if (bid >= price - 0.05 * self.position[product] + 0.2) and cpos > -self.pos_limit[product]:
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
                    logger.print(
                        f"sell added {product}, {sell_pr}, {cpos}, {-self.pos_limit[product]-cpos}")

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
