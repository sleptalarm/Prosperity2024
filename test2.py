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
    discount_buy = 0.98
    discount_sell = 0.98
    discount_inventory_buy = 16
    discount_inventory_sell = 10
    starfruit = []

    def compute_slope(self, y_values):
        # Assuming x-values are sequential integers starting from 0
        x_values = np.arange(len(y_values))
        
        # Number of observations
        N = len(x_values)
        # Sum of x-values
        sum_x = np.sum(x_values)
        # Sum of y-values
        sum_y = np.sum(y_values)
        # Sum of products of x and y
        sum_xy = np.sum(x_values * y_values)
        # Sum of x squared
        sum_x_squared = np.sum(x_values**2)
        
        # Calculating the slope
        slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x**2)
        
        return slope

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        for key, val in state.position.items():
            self.position[key] = val
        orders = {}
        conversions = 0
        trader_data = ""

        # values = list(state.order_depths['STARFRUIT'].sell_orders.items())
        # if len(values) != 0:
        #     logger.print("True, ", len(values))
        for product, order_depth in state.order_depths.items():
            print(product)
            if product == 'AMETHYSTS':

                orders[product] = []

                acceptable_price_buy = 9999
                acceptable_price_sell = 10001
                best_price_buy = 9996
                best_price_sell = 10004

                if len(order_depth.sell_orders) != 0:
                    tmp_list = list(order_depth.sell_orders.items())
                    for i in range(len(tmp_list)):
                        price, amount = tmp_list[i]
                        if int(price) <= acceptable_price_buy:
                            logger.print("BUY", str(-amount) + "x", price)
                            limit = self.pos_limit[product] - self.position[product]
                            if self.position[product] == 20:
                                logger.print(f"Can't buy {product}")
                            adjust = max(self.position[product]/ self.discount_inventory_buy,0) 
                            amount_new = price - best_price_buy
                            orders[product].append(Order(product, price, max(-amount, -limit)))
                    sort = sorted(tmp_list)
                    if sort[0][0] + 1 >= 10000:
                        orders[product].append(Order(product, sort[0][0] + 1, self.pos_limit[product] + self.position[product]))
                    

                if len(order_depth.buy_orders) != 0:
                    tmp_list = list(order_depth.buy_orders.items())
                    for i in range(len(tmp_list)):
                        price, amount = tmp_list[i]
                        if int(price) >= acceptable_price_sell:
                            logger.print("SELL", str(amount) + "x", price)
                            limit = self.pos_limit[product] + self.position[product]
                            if self.position[product] == -20:
                                logger.print(f"Can't Sell {product}")
                            adjust = max(-self.position[product]/ self.discount_inventory_sell,0) 
                            amount_new = best_price_sell - price
                            orders[product].append(Order(product, price, max(-amount, -limit)))
                    sort = sorted(tmp_list)
                    if sort[0][0] - 1 <= 10000:
                        orders[product].append(Order(product, sort[0][0] - 1, self.pos_limit[product] - self.position[product]))
                # orders[product] = []

                # acceptable_price_buy = 9998
                # acceptable_price_sell = 10002
                # best_price_buy = 9996
                # best_price_sell = 10004

                # if len(order_depth.sell_orders) != 0:
                #     tmp_list = list(order_depth.sell_orders.items())
                #     for i in range(len(tmp_list)):
                #         price, amount = tmp_list[i]
                #         if int(price) <= acceptable_price_buy:
                #             logger.print("BUY", str(-amount) + "x", price)
                #             limit = self.pos_limit[product] - self.position[product]
                #             if self.position[product] == 20:
                #                 logger.print(f"Can't buy {product}")
                #             adjust = max(self.position[product]/ self.discount_inventory_buy,0) 
                #             amount_new = round(amount * np.power(self.discount_buy,max(price - best_price_buy + adjust, 0)))
                #             orders[product].append(Order(product, price, max(-amount_new, -limit)))
                    

                # if len(order_depth.buy_orders) != 0:
                #     tmp_list = list(order_depth.buy_orders.items())
                #     for i in range(len(tmp_list)):
                #         price, amount = tmp_list[i]
                #         if int(price) >= acceptable_price_sell:
                #             logger.print("SELL", str(amount) + "x", price)
                #             limit = self.pos_limit[product] + self.position[product]
                #             if self.position[product] == -20:
                #                 logger.print(f"Can't Sell {product}")
                #             adjust = max(-self.position[product]/ self.discount_inventory_sell,0) 
                #             amount_new = round(amount * np.power(self.discount_sell,max(best_price_sell - price + adjust, 0)))
                #             orders[product].append(Order(product, price, max(-amount_new, -limit)))

            elif product == 'STARFRUIT':
                orders[product] = []
                if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                    logger.print("No orders for STARFRUIT")
                # predict price coefficient                
                buy_list = list(order_depth.sell_orders.items())
                sell_list = list(order_depth.buy_orders.items())
                midprice = (buy_list[0][0] + sell_list[0][0]) / 2

                # add more data if data is less than 6
                if len(self.starfruit) < 15:
                    self.starfruit.append(midprice)
                else:
                    # expected price
                    # price = round(np.dot(np.transpose(Coefficients),np.array(self.starfruit)) + Intercept)
                    self.starfruit.append(midprice)
                    self.starfruit.pop(0)
                    predict_price = round(self.Reggression())
                    logger.print(f"Predicted price for STARFRUIT: {predict_price}")
                    # +1 -1 to make profit
                    buy_price = predict_price
                    sell_price = predict_price + 1            

                    tot_vol = 0
                    mxvol = -1

                    for ask, vol in buy_list:
                        vol *= -1
                        tot_vol += vol
                        if tot_vol > mxvol:
                            mxvol = vol
                            best_sell_pr = ask
                    for ask, vol in sell_list:
                        tot_vol += vol
                        if tot_vol > mxvol:
                            mxvol = vol
                            best_buy_pr = ask      

                    cpos = self.position[product]

                    # conditions to buy
                    for ask, vol in buy_list:
                        if (ask <= buy_price) and cpos < self.pos_limit[product]:
                            order_for = min(-vol, self.pos_limit[product] - cpos)
                            cpos += order_for
                            assert(order_for >= 0)
                            orders[product].append(Order(product, ask, order_for))    

                    best_buy_pr += 1
                    best_sell_pr -= 1
                    bid_pr = min(best_buy_pr, buy_price)
                    sell_pr = max(best_sell_pr, sell_price)

                    if cpos < self.pos_limit[product]:
                        num = self.pos_limit[product] - cpos
                        orders[product].append(Order(product, bid_pr, num))  
                    
                    cpos = self.position[product]

                    # conditions to sell
                    for bid, vol in sell_list:
                        if (bid >= sell_price) and cpos > -self.pos_limit[product]:
                            order_for = max(-vol, -self.pos_limit[product]-cpos)
                            cpos += order_for
                            assert(order_for <= 0)
                            orders[product].append(Order(product, bid, order_for))

                    if cpos > -self.pos_limit[product]:
                        num = -self.pos_limit[product]-cpos
                        orders[product].append(Order(product, sell_pr, num))
                        cpos += num
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def Reggression(self):
        price = np.array(self.starfruit)
        x = np.arange(len(self.starfruit))
        linear = np.polyfit(x, price, 1)
        quadratic = np.polyfit(x, price, 2)
        predict_linear = []
        predict_quadratic = []
        linear_coefficient = np.poly1d(linear, r=False)
        quadratic_coefficient = np.poly1d(quadratic, r=False)
        for i in range(len(self.starfruit)):
            predict_linear.append(linear_coefficient(i))
            predict_quadratic.append(quadratic_coefficient(i))
        predict_linear = np.array(predict_linear)
        predict_quadratic = np.array(predict_quadratic)
        r2_linear = 1 - np.sum((predict_linear - price)**2) / np.sum((price - np.mean(price))**2)
        r2_quadratic = 1 - np.sum((predict_quadratic - price)**2) / np.sum((price - np.mean(price))**2)
        if r2_linear > r2_quadratic:
            return linear_coefficient(len(self.starfruit))
        else:
            return quadratic_coefficient(len(self.starfruit))



    
"""
                # predict price coefficient
                Coefficients = [0.09429971,0.08763816,0.12571242,0.14692366,0.22735297,0.31538848]
                Intercept = 13.581461215173476
                
                buy_list = list(order_depth.sell_orders.items())
                sell_list = list(order_depth.buy_orders.items())
                midprice = 5000

            
                # add more data if data is less than 6
                if len(self.starfruit) < 6:
                    self.starfruit.append(midprice)
                else:
                    # expected price
                    price = round(np.dot(np.transpose(Coefficients),np.array(self.starfruit)) + Intercept)
                    logger.print(f"Predicted price for STARFRUIT: {price}")

                    # +1 -1 to make profit
                    buy_price = price - 1
                    sell_price = price + 1
                    self.starfruit.append(midprice)
                    self.starfruit.pop(0)
                    tot_vol = 0
                    mxvol = -1

                    for ask, vol in buy_list:
                        vol *= -1
                        tot_vol += vol
                        if tot_vol > mxvol:
                            mxvol = vol
                            best_sell_pr = ask
                    for ask, vol in sell_list:
                        tot_vol += vol
                        if tot_vol > mxvol:
                            mxvol = vol
                            best_buy_pr = ask      

                    cpos = self.position[product]

                    # conditions to buy
                    for ask, vol in buy_list:
                        if ((ask <= buy_price) or ((self.position[product]<0) and (ask == buy_price+1))) and cpos < self.pos_limit[product]:
                            order_for = min(-vol, self.pos_limit[product] - cpos)
                            cpos += order_for
                            assert(order_for >= 0)
                            orders[product].append(Order(product, ask, order_for))    

                    best_buy_pr += 1
                    best_sell_pr -= 1
                    bid_pr = min(best_buy_pr, buy_price)
                    sell_pr = max(best_sell_pr, sell_price)

                    if cpos < self.pos_limit[product]:
                        num = self.pos_limit[product] - cpos
                        orders[product].append(Order(product, bid_pr, num))  


                    
                    cpos = self.position[product]

                    # conditions to sell
                    for bid, vol in sell_list:
                        if ((bid >= sell_price) or ((self.position[product]>0) and (bid+1 == sell_price))) and cpos > -self.pos_limit[product]:
                            order_for = max(-vol, -self.pos_limit[product]-cpos)
                            cpos += order_for
                            assert(order_for <= 0)
                            orders[product].append(Order(product, bid, order_for))

                    if cpos > -self.pos_limit[product]:
                        num = -self.pos_limit[product]-cpos
                        orders[product].append(Order(product, sell_pr, num))
                        cpos += num
"""
                    # state.traderData = " ".join(historical_price)
                    # if slope1 > 0.1 and slope2 > 0.1:
                    #     for i in range(1):
                    #         price, amount = buy_list[i]
                    #         limit = self.pos_limit[product] - self.position[product]
                    #         orders[product].append(Order(product, price, max(-amount, -limit)))
                    #         logger.print(f"Bought STARFRUIT at {price} for {max(-amount, -limit)} with slope {slope1} and timestamp {state.timestamp}") 
                    #         self.position[product] -= max(-amount, -limit)
                    # elif slope1 < -0.1 and slope2 < -0.1:
                    #     for i in range(1):
                    #         price, amount = sell_list[i]
                    #         limit = self.pos_limit[product] + self.position[product]
                    #         orders[product].append(Order(product, price, max(-amount, -limit)))
                    #         logger.print(f"Sold STARFRUIT at {price} for {max(-amount, -limit)} with slope {slope1} and timestamp {state.timestamp}") 
                    #         self.position[product] += max(-amount, -limit)

                # buy_list = list(order_depth.sell_orders.items())
                # sell_list = list(order_depth.buy_orders.items())
                # midprice = (buy_list[0][0] + sell_list[0][0]) / 2
                # # historical_price = [float(x) for x in state.traderData.split()]
                # if len(self.starfruit) < 11:
                #     self.starfruit.append(midprice)
                # else:
                #     slope1 = self.compute_slope(self.starfruit[:-1])
                #     slope2 = self.compute_slope(self.starfruit[1:])
                #     self.starfruit.append(midprice)
                #     self.starfruit.pop(0)
                #     # state.traderData = " ".join(historical_price)
                #     if slope1 > 0.1 and slope2 > 0.1:
                #         for i in range(1):
                #             price, amount = buy_list[i]
                #             limit = self.pos_limit[product] - self.position[product]
                #             orders[product].append(Order(product, price, max(-amount, -limit)))
                #             logger.print(f"Bought STARFRUIT at {price} for {max(-amount, -limit)} with slope {slope1} and timestamp {state.timestamp}") 
                #             self.position[product] -= max(-amount, -limit)
                #     elif slope1 < -0.1 and slope2 < -0.1:
                #         for i in range(1):
                #             price, amount = sell_list[i]
                #             limit = self.pos_limit[product] + self.position[product]
                #             orders[product].append(Order(product, price, max(-amount, -limit)))
                #             logger.print(f"Sold STARFRUIT at {price} for {max(-amount, -limit)} with slope {slope1} and timestamp {state.timestamp}") 
                #             self.position[product] += max(-amount, -limit)

