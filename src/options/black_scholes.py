# File: src/options/black_scholes.py

from numpy import exp, sqrt, log
from scipy.stats import norm


class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
        ) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        self.call_price = current_price * norm.cdf(d1) - (
            strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
        )
        self.put_price = (
            strike * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_delta = norm.cdf(d1)
        self.put_delta = self.call_delta - 1

        self.call_gamma = norm.pdf(d1) / (
            current_price * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return self.call_price, self.put_price