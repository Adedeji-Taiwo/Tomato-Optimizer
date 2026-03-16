import numpy as np
import pandas as pd

np.random.seed(42)

# Scenarios: three states of nature (e.g., good, avg, bad rainfall)
scenarios = ['Good', 'Average', 'Bad']
probabilities = [0.3, 0.5, 0.2]

# Tomato yield (tons per hectare) under each scenario
yield_per_ha = {'Good': 20, 'Average': 15, 'Bad': 10}

# Quality: brix (%) – higher is better for paste
brix = {'Good': 6.0, 'Average': 5.0, 'Bad': 4.0}

# Spoilage rate (%) during transport/storage
spoilage = {'Good': 5, 'Average': 10, 'Bad': 20}

# Total available tomatoes from contracted farms (tons)
total_ha = 100
available_tomatoes = {
    s: yield_per_ha[s] * total_ha * (1 - spoilage[s] / 100) for s in scenarios
}

# Product prices (₦ per ton)
base_prices = {'Fresh': 120000, 'Paste': 250000, 'Dried': 400000}
price_variation = 0.2

rng = np.random.default_rng(42)
market_price = {}
for s in scenarios:
    market_price[s] = {}
    for prod in base_prices.keys():
        multiplier = 1 + price_variation * (2 * rng.random() - 1)
        market_price[s][prod] = base_prices[prod] * multiplier

# Processing requirements and costs
conversion = {'Fresh': 1.0, 'Paste': 5.0, 'Dried': 8.0}
processing_cost = {'Fresh': 5000, 'Paste': 50000, 'Dried': 80000}

# Capacity limits (tons of final product per week)
capacity = {'Fresh': 100, 'Paste': 50, 'Dried': 20}

tomato_data = {
    'scenarios': scenarios,
    'probabilities': probabilities,
    'available': available_tomatoes,
    'brix': brix,
    'spoilage': spoilage,
    'prices': market_price,
    'conversion': conversion,
    'proc_cost': processing_cost,
    'capacity': capacity,
}
