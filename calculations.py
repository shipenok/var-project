import math
import numpy as np
from typing import List, Dict, Tuple

class StatisticsCalculator:

    @staticmethod
    def calculate_mean(numbers: List[float]) -> float:
        return sum(numbers) / len(numbers)
    
    @staticmethod
    def calculate_variance(numbers: List[float]) -> float:
        avg = StatisticsCalculator.calculate_mean(numbers)
        squared_deviations = [(x - avg)**2 for x in numbers]
        return sum(squared_deviations) / len(numbers)
    
    @staticmethod
    def calculate_std(numbers: List[float]) -> float:
        variance = StatisticsCalculator.calculate_variance(numbers)
        return math.sqrt(variance)
    
    @staticmethod
    def calculate_covariation(x: List[float], y: List[float]) -> float:
        
        n = len(x)
        mean_x = StatisticsCalculator.calculate_mean(x)
        mean_y = StatisticsCalculator.calculate_mean(y)
        covariation = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        return covariation
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        
        n = len(x)
        mean_x = StatisticsCalculator.calculate_mean(x)
        mean_y = StatisticsCalculator.calculate_mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x)**2 for i in range(n))
        denominator_y = sum((y[i] - mean_y)**2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0
            
        return numerator / math.sqrt(denominator_x * denominator_y)


class ProfitCalculator:
    
    @staticmethod
    def calculate_profit(old_price: float, new_price: float) -> float:
        return (new_price - old_price) / old_price
    
    @staticmethod
    def calculate_profits_arr(prices: List[float]) -> List[float]:

        profits = []
        for i in range(1, len(prices)):
            old_price = prices[i - 1]
            new_price = prices[i]
            profit = ProfitCalculator.calculate_profit(old_price, new_price)
            profits.append(profit)
        return profits
    
    @staticmethod
    def calculate_log_profits(prices: List[float]) -> List[float]:
        log_profits = []
        for i in range(1, len(prices)):
            log_profit = math.log(prices[i] / prices[i-1])
            log_profits.append(log_profit)
        return log_profits


class VaRCalculator:

    def __init__(self, confidence_level: float = 0.99, time_horizon: int = 7):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.alpha = 1 - confidence_level
        
        self.z_scores = {
            0.50: 0,
            0.51: -0.025,
            0.52: -0.05,
            0.53: -0.075,
            0.54: -0.1,
            0.55: -0.126,
            0.56: -0.151,
            0.57: -0.176,
            0.58: -0.202,
            0.59: -0.228,
            0.60: -0.253,
            0.61: -0.279,
            0.62: -0.305,
            0.63: -0.332,
            0.64: -0.358,
            0.65: -0.385,
            0.66: -0.412,
            0.67: -0.44,
            0.68: -0.468,
            0.69: -0.496,
            0.70: -0.524,
            0.71: -0.553,
            0.72: -0.583,
            0.73: -0.613,
            0.74: -0.643,
            0.75: -0.674,
            0.76: -0.706,
            0.77: -0.739,
            0.78: -0.772,
            0.79: -0.806,
            0.80: -0.842,
            0.81: -0.878,
            0.82: -0.915,
            0.83: -0.954,
            0.84: -0.994,
            0.85: -1.036,
            0.86: -1.08,
            0.87: -1.126,
            0.88: -1.175,
            0.89: -1.227,
            0.90: -1.282,
            0.91: -1.341,
            0.92: -1.405,
            0.93: -1.476,
            0.94: -1.555,
            0.95: -1.645,
            0.96: -1.751,
            0.97: -1.881,
            0.98: -2.054,
            0.99: -2.326,
            0.991: -2.366,
            0.992: -2.409,
            0.993: -2.457,
            0.994: -2.512,
            0.995: -2.576,
            0.996: -2.652,
            0.997: -2.748,
            0.998: -2.878,
            0.999: -3.09
        }

    def get_z_score(self, confidence_level: float) -> float:

        if confidence_level in self.z_scores:
            return self.z_scores[confidence_level]

        min_distance = 1.0
        closest_confidence = None
        
        for conf_level in self.z_scores.keys():
            distance = abs(confidence_level - conf_level)
            if distance < min_distance:
                min_distance = distance
                closest_confidence = conf_level
        
        return self.z_scores[closest_confidence]
    
    def parametric_var(self, profits: List[float], portfolio_value: float) -> Dict[str, float]:
        
        mean_return = StatisticsCalculator.calculate_mean(profits)
        std_return = StatisticsCalculator.calculate_std(profits)
        
        z_score = self.get_z_score(self.confidence_level)
        
        time_scaling_factor = math.sqrt(self.time_horizon)
        time_scaled_mean = mean_return * self.time_horizon
        time_scaled_std = std_return * time_scaling_factor
        
        var_percentage = time_scaled_mean + z_score * time_scaled_std
        var_absolute = abs(var_percentage * portfolio_value)
        
        return {
            'var_percentage': var_percentage,
            'var_absolute': var_absolute,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'mean_return': mean_return,
            'std_return': std_return,
            'z_score': z_score,
            'time_scaled_mean': time_scaled_mean,
            'time_scaled_std': time_scaled_std,
            'method': 'parametric'
        }
    
    def historical_var(self, profits: List[float], portfolio_value: float) -> Dict[str, float]:
        
        time_scaling_factor = math.sqrt(self.time_horizon)
        scaled_profits = [r * time_scaling_factor for r in profits]
        sorted_profits = sorted(scaled_profits)
        
        exact_index = len(sorted_profits) * self.alpha
        index = int(exact_index)

        var_percentage = sorted_profits[index]
        var_absolute = abs(var_percentage * portfolio_value)
        
        return {
            'var_percentage': var_percentage,
            'var_absolute': var_absolute,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'method': 'historical',
            'quantile_index': index,
            'exact_quantile_index': exact_index,
            'total_observations': len(profits),
            'time_scaling_factor': time_scaling_factor
        }
    
class PortfolioVaRCalculator:

    def __init__(self, confidence_level: float = 0.99, time_horizon: int = 7):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.alpha = 1 - confidence_level
        self.var_calculator = VaRCalculator(confidence_level, time_horizon)
    
    def calculate_portfolio_statistics(self, profits_data: Dict[str, List[float]], 
                                      weights: Dict[str, float]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        
        assets = list(profits_data.keys())
        n_assets = len(assets)
        
        weights_array = np.array([weights[asset] for asset in assets])
        
        means = np.array([StatisticsCalculator.calculate_mean(profits_data[asset]) for asset in assets])
        
        covariance_matrix = np.zeros((n_assets, n_assets))
        
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                covariance_matrix[i, j] = StatisticsCalculator.calculate_covariation(
                    profits_data[asset_i], profits_data[asset_j])
        
        portfolio_mean = np.dot(weights_array, means)
        
        portfolio_variance = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
        portfolio_std = math.sqrt(portfolio_variance)
        
        return portfolio_mean, portfolio_std, weights_array, covariance_matrix

    def parametric_portfolio_var(self, returns_data: Dict[str, List[float]], 
                                weights: Dict[str, float], 
                                portfolio_value: float) -> Dict[str, float]:

        portfolio_mean, portfolio_std, weights_array, cov_matrix = self.calculate_portfolio_statistics(
            returns_data, weights)
        
        time_scaling_factor = math.sqrt(self.time_horizon)
        time_scaled_mean = portfolio_mean * self.time_horizon
        time_scaled_std = portfolio_std * time_scaling_factor
        
        z_score = self.var_calculator.get_z_score(self.confidence_level)
        
        var_percentage = time_scaled_mean + z_score * time_scaled_std
        var_absolute = abs(var_percentage * portfolio_value)
        
        return {
            'var_percentage': var_percentage,
            'var_absolute': var_absolute,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std,
            'time_scaled_mean': time_scaled_mean,
            'time_scaled_std': time_scaled_std,
            'z_score': z_score,
            'method': 'parametric',
            'weights': dict(zip(returns_data.keys(), weights_array)),
            'covariance_matrix': cov_matrix.tolist()
        }
    
    def historical_portfolio_var(self, profits_data: Dict[str, List[float]], 
                                weights: Dict[str, float], 
                                portfolio_value: float) -> Dict[str, float]:
        
        assets = list(profits_data.keys())
        n_periods = len(profits_data[assets[0]])
        
        portfolio_profits = []
        for t in range(n_periods):
            period_profit = sum(weights[asset] * profits_data[asset][t] for asset in assets)
            portfolio_profits.append(period_profit)
        
        result = self.var_calculator.historical_var(portfolio_profits, portfolio_value)
        
        result['weights'] = weights
        result['portfolio_profits_count'] = len(portfolio_profits)
        
        return result