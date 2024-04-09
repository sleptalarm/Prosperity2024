import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import numpy as np

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
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

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

    position = {'AMETHYSTS' : 0, 'STARFRUIT' : 0}
    pos_limit = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    # discount = 0.02
    discount = 0.98
    discount_inventory = 16

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        for key, val in state.position.items():
            self.position[key] = val
        orders = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            if product == 'AMETHYSTS':
                orders[product] = []

                acceptable_price_buy = 9998
                acceptable_price_sell = 10002
                best_price_buy = 9996
                best_price_sell = 10004

                
                logger.print(f"Acceptable price for {product}: {acceptable_price_buy}")
                logger.print(f"Acceptable price for {product}: {acceptable_price_sell}")

                if len(order_depth.sell_orders) != 0:
                    tmp_list = list(order_depth.sell_orders.items())
                    for i in range(len(tmp_list)):
                        price, amount = tmp_list[i]
                        if int(price) <= acceptable_price_buy:
                            logger.print("BUY", str(-amount) + "x", price)
                            limit = self.pos_limit[product] - self.position[product]
                            if self.position[product] == 20:
                                logger.print(f"Can't buy {product}")
                            # orders[product].append(Order(product, price, -amount))
                            adjust = max(self.position[product]/ self.discount_inventory,0) 
                            amount_new = round(amount * np.power(self.discount,max(price - best_price_buy + adjust, 0)))
                            # amount_new = round(amount * (1 - self.discount * max(price - best_price_buy + adjust, 0)))
                            orders[product].append(Order(product, price, max(-amount_new, -limit)))
                        else:
                            break
                    
                    # best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    # if int(best_ask) < acceptable_price_buy:
                    #     logger.print("BUY", str(-best_ask_amount) + "x", best_ask)
                    #     limit = self.pos_limit[product] - self.position[product]
                    #     # orders[product].append(Order(product, best_ask, -best_ask_amount))
                    #     orders[product].append(Order(product, best_ask, max(-best_ask_amount, -limit)))

                if len(order_depth.buy_orders) != 0:
                    tmp_list = list(order_depth.buy_orders.items())
                    for i in range(len(tmp_list)):
                        price, amount = tmp_list[i]
                        if int(price) >= acceptable_price_sell:
                            logger.print("SELL", str(amount) + "x", price)
                            limit = self.pos_limit[product] + self.position[product]
                            if self.position[product] == -20:
                                logger.print(f"Can't Sell {product}")
                            # orders[product].append(Order(product, price, amount))
                            adjust = max(-self.position[product]/ self.discount_inventory,0) 
                            amount_new = round(amount * np.power(self.discount,max(best_price_sell - price + adjust, 0)))
                            # amount_new = round(amount * (1 - self.discount * max(best_price_sell - price + adjust, 0)))
                            orders[product].append(Order(product, price, max(-amount, -limit)))
                        else:
                            break

                    # best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    # if int(best_bid) > acceptable_price_sell:
                    #     logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                    #     limit = self.pos_limit[product] + self.position[product]
                    #     # orders[product].append(Order(product, best_bid, -best_bid_amount))
                    #     orders[product].append(Order(product, best_bid, max(-best_bid_amount, -limit)))

            logger.flush(state, orders, conversions, trader_data)
            return orders, conversions, trader_data