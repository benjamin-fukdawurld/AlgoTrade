# Import finta to get custom indicator
from gym_anytrading.envs import ForexEnv


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[
        start:end]
    return prices, signal_features


class CustomEnv(ForexEnv):
    _process_data = add_signals
