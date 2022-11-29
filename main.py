from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000         # Number of steps to fill the replay memory before training
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

PR_SCALE = 0.7

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device) # +1 for the next state (?)

#### Training ####
obs_queue: deque = deque(maxlen=5)  # The observation queue, which is used to make up a state
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b") # The progress bar
for step in progressive:
    if done:
        # If the episode is done, reset the environment and the observation queue
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()    # Make up a state
    action = agent.run(state, training)                     # Run the agent to get an action
    obs, reward, done = env.step(action)                    # Perform the action in the environment and get the observation frames and the reward
    obs_queue.append(obs)                                   # Append the observation to the queue
    memory.push(env.make_folded_state(obs_queue), action, reward, done) # Push the state, action, reward, and done flag to the replay memory

    if step % POLICY_UPDATE == 0 and training:
        # For every POLICY_UPDATE steps, update the policy network by calling the agent to learn
        # The agent will then sample a batch of transitions from the replay memory
        # It generates Qopt and Q'opt from the policy and target networks respectively
        # And then it updates the policy network using the loss function
        agent.learn(memory, BATCH_SIZE, PR_SCALE)

    if step % TARGET_UPDATE == 0:
        # For every TARGET_UPDATE steps, update the target network by copying the policy network
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        # For every EVALUATE_FREQ steps, evaluate the agent
        # The agent will run the game for a few episodes and return the average reward
        # The agent will also return the captured frames for visualization
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
