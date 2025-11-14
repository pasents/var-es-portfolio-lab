# config.py

# Portfolio weights in order: [BTC_EUR, GOLD_EUR, IWDA_EUR]
WEIGHTS = [0.2, 0.2, 0.6]

# Confidence levels for VaR/ES
CONF_LEVELS = [0.95, 0.99, 0.995]

# Horizon (days)
HORIZON_DAYS = 1

# Sample date range
START_DATE = "2018-01-01"
END_DATE = None   # None = up to today
