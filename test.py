import robel
import gym
import time

# Create a simulation environment for the D'Claw turn task.
env = gym.make('DClawTurnFixedT6-v0')

done = False
obs = env.reset()

#while not done:
while 1:
    print(obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time.sleep(0.05)
    env.render()

