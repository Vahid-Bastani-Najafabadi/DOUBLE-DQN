# main.py
# Single‐file Double DQN implementation for the SkullTrophyRLEnv environment.
# ------------------------------------------------------------

# --- Imports ---
from torchvision import transforms               # image preprocessing
from PIL import Image                            # for PIL image conversion
import torch                                     # main PyTorch library
from RLSkullTrophy.core import SkullTrophyRLEnv       # custom environment
import torch.nn as nn                            # neural network modules
import torch.nn.functional as F                  # activation functions, etc.
from torchinfo import summary                    # model summary printer
import torch.optim as optim                      # optimizers
import numpy as np                               # numerical computing
import copy                                      # for deep copying the network
import random                                    # random seeding and sampling
import matplotlib.pyplot as plt                  # plotting training curves

# ------------------------------------------------------------
# --- Q-Network Definition ---
class DEEPQ_net(nn.Module):
    """
    Convolutional Q-network with:
    - 3 conv layers (32→64→64), two max-pools
    - Flatten → FC(1024) → output n_actions
    """
    def __init__(self, input_channels, n_actions):
        super(DEEPQ_net, self).__init__()
        # Convolution + pooling layers
        self.conv1   = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2   = nn.Conv2d(32, 64,          kernel_size=5, stride=2, padding=1)
        self.conv3   = nn.Conv2d(64, 64,          kernel_size=3)
        self.mxp1    = nn.MaxPool2d(kernel_size=2)
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(64 * 8 * 8, 1024)
        self.fc2     = nn.Linear(1024, n_actions)

    def forward(self, x):
        # Pass through conv + relu + pool
        x = self.mxp1(F.relu(self.conv1(x)))
        x = self.mxp1(F.relu(self.conv2(x)))
        # Final conv, then flatten to vector
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        # FC layers to output Q-values
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------------------------------------
# --- Device Setup & Model Summary ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)
model = DEEPQ_net(input_channels=3, n_actions=6).to(device)
# Print a summary of layers & output shapes for input size (1,3,84,84)
summary(model, input_size=(1, 3, 84, 84))

# ------------------------------------------------------------
# --- State Preprocessing Pipeline ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),           # convert NumPy array to PIL Image
    transforms.Resize((84, 84)),       # resize to 84×84
    transforms.ToTensor()              # convert back to PyTorch tensor (C×H×W, normalized [0,1])
])

# ------------------------------------------------------------
# --- Replay Buffer Definition ---
class replaybuffer:
    """Simple FIFO replay buffer with fixed capacity."""
    def __init__(self, cap):
        self.buffer   = []     # list of (state, action, reward, next_state, done)
        self.capacity = cap

    def push(self, state, action, reward, next_state, flag):
        """Add a transition and evict oldest if over capacity."""
        self.buffer.append((state, action, reward, next_state, flag))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, flags = zip(*batch)
        return (
            np.stack(states),                                        # (batch, C, H, W)
            np.array(actions),                                       # (batch,)
            np.array(rewards, dtype=np.float32),                     # (batch,)
            np.stack(next_states),                                   # (batch, C, H, W)
            np.array(flags, dtype=np.uint8)                          # (batch,)
        )

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

# ------------------------------------------------------------
# --- Hyperparameters ---
batch_size         = 64         # number of samples per training update
discount_factor    = 0.995      # γ in Bellman equation
target_update      = 1000       # how often to copy weights to target network
epsilon_start      = 1.0        # initial exploration rate
epsilon_final      = 0.01       # final exploration rate
epsilon_decay      = 50000      # over how many steps to decay ε

# Function to compute ε given the current global step
def curr_epsilon(step):
    """Linearly decay epsilon from start to final over epsilon_decay steps."""
    exploration = epsilon_final
    exploitation = epsilon_start - epsilon_final
    saturation = max(0, (epsilon_decay - step) / epsilon_decay)
    return exploration + exploitation * saturation

# ------------------------------------------------------------
# --- Environment, Networks, Buffer & Optimizer Setup ---
random.seed(42)                                    # fix random seed
env = SkullTrophyRLEnv()                            # instantiate custom env
model = DEEPQ_net(3, 6).to(device)                  # policy network
target_model = copy.deepcopy(model).to(device)      # target network
target_model.eval()                                 # set target to eval mode
replay_buffer = replaybuffer(cap=50000)             # experience buffer

criterion = nn.MSELoss()                            # loss between Q_pred & Q_target
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------------------------------------
# --- Training Loop Variables ---
num_episodes       = 6000     # total episodes to run
global_step        = 0        # total environment steps taken
max_episode_length = 100      # max steps per episode
warmup_steps       = 500      # no learning until buffer has this many steps

rewards_list       = []       # store reward per episode
epsiode_length     = []       # store length per episode

# ------------------------------------------------------------
# --- Main Training Loop ---
for epispde in range(num_episodes):
    # 1) Sample a valid initial state from env
    while True:
        s0 = random.randint(0, 18)
        state, init_rew, done = env.beginEpisode(s0)
        # require done==True but not init_rew==1
        if done and init_rew != 1:
            break

    state = preprocess(state)        # preprocess raw image to tensor
    episode_reward = 0.0
    step_count = 0

    # 2) Step through the episode
    while done and step_count < max_episode_length:
        global_step += 1
        step_count += 1

        # ε-greedy action selection
        epsilon = curr_epsilon(global_step)
        if random.random() < epsilon:
            action = random.randint(0, 5)
        else:
            q_val = model(state.unsqueeze(0).to(device))
            action = q_val.argmax(dim=1).item()

        # 3) Execute action in the env
        next_state, reward, done = env.step(action)
        episode_reward += reward
        next_state = preprocess(next_state)

        # 4) Store transition
        replay_buffer.push(
            state.cpu().numpy(), action, reward,
            next_state.cpu().numpy(), done
        )
        state = next_state

        # 5) Learning step once warmup is over
        if global_step >= warmup_steps and len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert to torch tensors
            states      = torch.tensor(states).to(device)
            actions     = torch.tensor(actions).unsqueeze(1).to(device)
            rewards_t   = torch.tensor(rewards).to(device)
            next_states = torch.tensor(next_states).to(device)
            dones       = torch.tensor(dones).unsqueeze(1).to(device)

            # Compute current Q predictions
            q_pred = model(states).gather(1, actions)
            # Compute target Q values with target network
            with torch.no_grad():
                q_next   = target_model(next_states).max(dim=1, keepdim=True)[0]
                q_target = rewards_t + discount_factor * q_next * dones

            # Backpropagate loss
            loss = criterion(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 6) Periodically update target network
        if global_step % target_update == 0:
            target_model.load_state_dict(model.state_dict())

    # End of episode bookkeeping
    rewards_list.append(episode_reward)
    epsiode_length.append(step_count)
    if (epispde + 1) % 100 == 0:
        print(f"Episode {epispde+1}, Total Reward: {episode_reward}, "
              f"Steps: {step_count}, ε: {epsilon:.3f}")

# ------------------------------------------------------------
# --- Training Metrics & Plotting ---
last2000 = np.array(rewards_list[-1200:], dtype=float)
correct_eps = np.sum(last2000 > 0)
print(f"Train Accuracy (last 1200 eps > 0): {correct_eps/1200:.4f}")

# Moving average of rewards
returns = np.array(rewards_list, dtype=float)
window = 100
ma = np.convolve(returns, np.ones(window)/window, mode='valid')

plt.plot(returns, alpha=0.3, label='Rewards')
plt.plot(np.arange(window-1, len(returns)), ma, label=f'{window}-MA')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# ------------------------------------------------------------
# --- Greedy‐Policy Evaluation ---
def select_action_greedy(state_tensor):
    """Select action with highest Q-value (no exploration)."""
    with torch.no_grad():
        q = model(state_tensor.to(device))
        return q.argmax(dim=1).item()

model.eval()  # set policy network to eval mode

num_eval_eps = 1000
max_episode_length = 100
returns, lengths, trupy_flags = [], [], []
successes = 0

for ep in range(num_eval_eps):
    raw, _, _ = env.beginEpisode(s0=17)
    state = preprocess(raw).unsqueeze(0).to(device)
    running = True
    ep_r, steps = 0.0, 0

    while running and steps < max_episode_length:
        steps += 1
        action = select_action_greedy(state)
        raw_next, reward, running = env.step(action)
        ep_r += reward
        state = preprocess(raw_next).unsqueeze(0).to(device)

    if reward > 0:
        successes += 1
        trupy_flags.append(ep)

    returns.append(ep_r)
    lengths.append(steps)

# Print evaluation stats
mean_return  = np.mean(returns)
std_return   = np.std(returns)
mean_length  = np.mean(lengths)
success_rate = successes / num_eval_eps * 100

print(f"Eval over {num_eval_eps} greedy episodes:")
print(f"Avg Rewards: {mean_return:.2f} ± {std_return:.2f}")
print(f"Avg Num of Actions: {mean_length:.1f}")
print(f"Success Rate: {success_rate:.1f}%")

# ------------------------------------------------------------
# --- Success Steps Histogram ---
lengths2 = np.array(lengths)
trup = np.array(trupy_flags)
successes_steps = lengths2[trup]

print(f"Average Steps (successful): {np.mean(successes_steps):.2f}")
print(f"Std Steps (successful): {np.std(successes_steps):.2f}")

bins = np.arange(min(successes_steps), max(successes_steps)+2)
plt.figure(figsize=(16, 8))
plt.hist(successes_steps, bins=bins, align='left', edgecolor='black', rwidth=0.8)
plt.xticks(np.arange(min(successes_steps)-1, max(successes_steps)+1))
plt.xlabel("Number of Steps")
plt.ylabel("Frequency")
plt.title("Histogram of Steps in Successful Episodes")
plt.show()

# ------------------------------------------------------------
# --- Visualizing Final Episode ---
raw, _, _ = env.beginEpisode(s0=17)
plt.imshow(raw)
plt.title("Starting Observation")
plt.show()

model.eval()
done = True
step = 0
rewards = 0
while done and step < 20:
    step += 1
    state = preprocess(raw).unsqueeze(0).to(device)
    action = select_action_greedy(state)
    raw_next, reward, done = env.step(action)
    rewards += reward
    print(f"Reward at step {step}: {rewards}")
    plt.imshow(raw_next)
    plt.title(f"Step {step} Observation")
    plt.show()
    raw = raw_next
