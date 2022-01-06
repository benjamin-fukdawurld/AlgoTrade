# Import Gym stuffs (RL environment)
from matplotlib import pyplot as plt

from agent import Agent
from custom_env import CustomEnv
from data_loader import CSVDataProvider

data_provider = CSVDataProvider('btc_data.csv')

# Import market data
market_data = data_provider.data('2015-1-1', '2022-2-2')

agent = Agent(market_data, (2220, 2300), 20)
agent.train()
agent.save()

# agent = Agent.load(market_data, (2020, 2100), 20)

env = CustomEnv(df=market_data, frame_bound=(2320, 2500), window_size=20)
agent.run(env)

plt.figure(figsize=(52, 10))
plt.cla()
env.render_all()
plt.show()
