'''
Yongcheng Ding

'''
from tensorforce.agents import TRPOAgent
from tensorforce.agents import PPOAgent
from tensorforce.agents import DuelingDQNAgent
from tensorforce.agents import DQNAgent
import RELAXENV
import numpy as np
from tensorforce.execution import Runner
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

WEAKMEASUREMENT=RELAXENV.WEAKMEASUREMENT()#prepare the environment

#hidden layer
network_spec=[
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        ]

np.random.seed(1)

agent=PPOAgent(
        states=WEAKMEASUREMENT.states(),
        actions=WEAKMEASUREMENT.actions(),
        network=network_spec,
        max_episode_timesteps=WEAKMEASUREMENT.max_episode_timesteps(),
        learning_rate = 1e-3,
        #for pretrain 3*1e-3, for fine tuningh 1e-3
        batch_size=20,
        saver=dict(directory='best-model')
        )


agent.initialize()
runner=Runner(agent=agent,
              environment=WEAKMEASUREMENT,
              save_best_agent=True,
              evaluation_environment=WEAKMEASUREMENT
              )

runner.run(num_episodes=8000)