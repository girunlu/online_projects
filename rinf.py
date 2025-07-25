import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(9, 32)
        self.fc3 = nn.Linear(32, 9)

    def forward(self, x):  # x: state (9 values)
        x = F.relu(self.fc(x))          # raw scores
        logits = self.fc3(x)
        return logits
    
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0 = empty, 1 = agent, -1 = opponent
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.copy().astype(np.float32)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # cols
            [0,4,8], [2,4,6]            # diagonals
        ]
        for line in lines:
            total = sum(self.board[i] for i in line)
            if total == 3:
                return 1
            if total == -3:
                return -1
        if 0 not in self.board:
            return 0  # draw
        return None  # game ongoing

    def step_random(self, action):
        if self.done or self.board[action] != 0:
            raise ValueError('invalid move')

        self.board[action] = 1  # agent move
        result = self.check_winner()
        if result is not None:
            self.done = True
            self.winner = result
            return self.get_state(), self._reward(result), self.done

        # Opponent move (random)
        opp_actions = self.available_actions()
        if opp_actions:
            opp_move = random.choice(opp_actions)
            self.board[opp_move] = -1
            result = self.check_winner()
            if result is not None:
                self.done = True
                self.winner = result
                return self.get_state(), self._reward(result), self.done
        return self.get_state(), 0, self.done

    def step_opp(self, action, player):
        if self.done or self.board[action] != 0:
            raise ValueError('invalid move')
        
        self.board[action] = player  # agent move
        result = self.check_winner()
        if result is not None:
            self.done = True
            self.winner = result
            return self.get_state(), self._reward(result), self.done
        return self.get_state(), 0, self.done

    def _reward(self, result):
        if result == 1:
            return 1   # win
        elif result == -1:
            return -1  # lose
        else:
            return 0.7   # draw

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(3):
            row = [symbols[self.board[j]] for j in range(i*3, (i+1)*3)]
            print(' '.join(row))
        print()

# Compute returns G_t (delayed reward, only final)
def compute_returns(final_reward, length, gamma=0.99):
    return [final_reward * (gamma ** (length - t - 1)) for t in range(length)]

# Initialize
env = TicTacToe()
model = PolicyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.97
baseline_buffer = deque(maxlen=100)

win_count = 0
loss_count = 0
draw_count = 0
history = []  # stores (win%, draw%, loss%) over time

# Training loop
for episode in range(6000):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    log_probs = []
    states = []
    actions = []

    while True:
        logits = model(state)
        mask = torch.tensor([1 if i in env.available_actions() else 0 for i in range(9)])
        masked_logits = logits + (mask + 1e-10).log()  # mask invalid actions
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        actions.append(action.item())
        states.append(state)
        try:
            next_state, reward, done = env.step_random(action.item())
        except:
            continue
        state = torch.tensor(next_state, dtype=torch.float32)
        
        if done:
            break
        
    if reward == 1:
        win_count += 1
    elif reward == -1:
        loss_count += 1
    else:
        draw_count += 1

    # Compute returns (same reward for all moves, discounted)
    returns = compute_returns(reward, len(log_probs), gamma)

    # Optional: use baseline (average of recent returns)
    baseline = np.mean(baseline_buffer) if baseline_buffer else 0
    baseline_buffer.append(reward)

    # Policy loss
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * (G - baseline)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 1000 == 0:
        total = win_count + draw_count + loss_count
        win_ratio = win_count / total
        draw_ratio = draw_count / total
        loss_ratio = loss_count / total
        history.append((win_ratio, draw_ratio, loss_ratio))
        print(f"[Ep {episode+1}] Win: {win_ratio:.2f}, Draw: {draw_ratio:.2f}, Loss: {loss_ratio:.2f}")
        
        # Reset interval counts
        win_count = draw_count = loss_count = 0

# RLHF XD
def play_and_learn_from_me(model, optimizer, model_side = -1, gamma=0.97):
    env = TicTacToe()
    state = torch.tensor(env.reset(), dtype=torch.float32)
    log_probs = []
    done = False
    player = model_side
    
    if player == 1:  # model starts
        logits = model(state)
        mask = torch.tensor([1 if i in env.available_actions() else 0 for i in range(9)])
        masked_logits = logits + (mask + 1e-10).log()
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        
        state_np, reward, done = env.step_opp(action, player)
        state = torch.tensor(state_np, dtype=torch.float32)
        
        log_probs.append(dist.log_prob(action))
        
        env.render()
        player *= -1

    while not done: 
        if player == 1:
            logits = model(state)
            mask = torch.tensor([1 if i in env.available_actions() else 0 for i in range(9)])
            masked_logits = logits + (mask + 1e-10).log()
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
        else:
            env.render()
            action = int(input("Your move (1-9): ")) - 1

        next_state, reward, done = env.step_opp(action, player)
        state = torch.tensor(next_state, dtype=torch.float32)
        player *= -1

    if reward == 1:
        returns = compute_returns(1.0, len(log_probs), gamma)
    elif reward == -1:
        returns = compute_returns(-1.0, len(log_probs), gamma)
    else:
        returns = compute_returns(0.7, len(log_probs), gamma)

    loss = sum(-lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#cloning the weights after 1 vs random playing
model_a, model_b =PolicyNet(),PolicyNet()
model_a.load_state_dict(model.state_dict())
model_b.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model_a.parameters(), lr=1e-3)
gamma = 0.97
update_interval = 750  # episodes before syncing model_b
baseline_buffer = deque(maxlen=100)

def predict_action(model, next_state, env, player):
    with torch.no_grad():
        logits = model(next_state * player)
        mask = torch.tensor([1 if i in env.available_actions() else 0 for i in range(9)])
        masked_logits = logits + (mask + 1e-10).log()  # mask invalid actions
        action = torch.argmax(masked_logits).item()
    return action

def get_action_and_logprob(model, state, env):
    logits = model(state)
    mask = torch.tensor([1 if i in env.available_actions() else 0 for i in range(9)])
    masked_logits = logits + (mask + 1e-10).log()  # mask invalid actions
    dist = torch.distributions.Categorical(logits=masked_logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# Training

wins = draws = losses = 0
for episode in range(18001):
    env = TicTacToe()
    state = torch.tensor(env.reset(), dtype=torch.float32)

    player = -1 if episode % 10 <= 4 else 1
    log_probs = []
    rewards = []
    states = []
    
    if (episode+1) % 800 == 0:
        inx = np.power(-1, ((episode+1) // 1600)%2)
        play_and_learn_from_me(model_a, optimizer, inx.item(), gamma)

    while True:
        if player == 1:
            state_input = torch.tensor(env.get_state(), dtype=torch.float32)
            action, log_prob = get_action_and_logprob(model_a, state_input, env)
            log_probs.append(log_prob)
        else:
            # model_b is fixed (greedy, no sampling)
            action = predict_action(model_b, state, env, player)
        try:
            next_state, reward, done = env.step_opp(action, player)
        except:
            reward = -1
            done = True
            next_state = env.get_state()
            
        if player == 1:
            states.append(state)
            rewards.append(reward)  # reward is +1 win, -1 loss, 0.5 draw

        state = torch.tensor(next_state, dtype=torch.float32)
        player *= -1

        if done:
            if reward == 1:
                wins += 1
            elif reward == 0.7:
                draws += 1
            else:
                losses += 1
            break

    # Compute discounted returns for model_a
    returns = [reward * (gamma ** (len(log_probs) - t - 1)) for t in range(len(log_probs))]

    # Baseline
    baseline = sum(baseline_buffer) / len(baseline_buffer) if baseline_buffer else 0
    baseline_buffer.append(reward)

    # Policy gradient loss for model_a
    loss = sum(-lp * (G - baseline) for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (episode + 1) % update_interval == 0:
        model_b.load_state_dict(model_a.state_dict())

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode+1} | Reward: {reward:.2f} | Loss: {loss.item():.4f} | Wins: {wins}, Draws: {draws}, Losses: {losses}")
        wins = draws = losses = 0  # reset
        
torch.save(model_a.state_dict(), "model_a_weights.pth")
