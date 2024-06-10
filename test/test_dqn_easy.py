import gym

import numpy as np
import random
from random import seed
from gym.spaces import Box, Discrete, Dict, Tuple
from pathlib import Path
import re


from board import create_board, fortify_bfs, create_graph, display_graph, create_board_test
from atomic_actions import attack_territory, place_troops, take_cards, trade_cards, fortify, get_troops

#Rk Added
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

import statistics
import os


bot_types = ["Neutral"] * 1
class RiskEnvFlat(gym.Env):
    def __init__(self, bot_types=None, env_config=None):
        if env_config is None:
            env_config = {}  # Provide a default empty configuration dictionary if not provided
        self.num_players = env_config.get("num_players", 6)
        self.colors = env_config.get("colors", ['red', 'blue', 'green', 'yellow', 'purple', 'pink'][:self.num_players])
        self.invalid_move_penalty = -1

        #bot properties
        # self.bot_types =  ["TFT", "Random", "TFT", None, "Neutral"]  # corresponds to turn order (indexed at Agent + 1), None is just a Player
        if bot_types: self.bot_types = bot_types
        else: self.bot_types = ["Neutral"] * (self.num_players - 1)
        if all(bot == "Neutral" for bot in self.bot_types): # if neutral only, immediately apply negative reward to encourage faster game ends
            self.neutral_only = True
        else:
            self.neutral_only = False
        self.shuffle_bots = True # False likely teaches arbitrary overfitting to each bot based on turn order
        if self.shuffle_bots: random.shuffle(self.bot_types)
        self.skip_bots = True

        #environment properties
        self.continents, self.territories, self.players = create_board(self.num_players, self.bot_types)
        self.turns_passed = 0
        self.phase = 0
        self.troops = 0 # not sure we need, maybe could help observation calculations
        self.agent = self.players[0]
        self.from_terr = None
        self.prev_move = len(self.territories)
        state = self.get_state()
        self.action_space = Discrete(len(self.territories) + 1)  # choose territory or skip
        self.observation_space = Box(low=-np.iinfo(np.int32).max,high=np.iinfo(np.int32).max, shape=(state.shape[0],))

        #enviornment agent properites (for reward calculation)
        self.phasic_credit_assignment = True # should be default True once this is implemented
        self.agent_game_ended = False
        self.agent_troop_gain = 0 #always reset to 0 after reward is observed
        self.agent_gets_card = False #did the Agent win a card by taking at least 1 territory
        self.players_agent_survived = 0 #" "
        self.players_agent_eliminated = 0 #" "
        self.recurrence = False #is the Agent going back to an earlier phase?
        self.CARD_TROOP_VALUE = 3 # the expected troop gain value of trading a card
        self.EARLY_GAME = 6 # turn at which to stop giving rewards for basic survival skills
        self.LATE_GAME = 15 # turn at which to start punishments for taking too long (Agent gets bored)

    def reset(self, seed=None, options=None):
        #initlize board and get state
        if seed is not None:
            random.seed(seed)
        if self.shuffle_bots: random.shuffle(self.bot_types)
        self.continents, self.territories, self.players = create_board(self.num_players, self.bot_types)
        self.agent = self.players[0]
        self.turns_passed = 0
        self.phase = 0
        self.from_terr = None
        self.prev_move = len(self.territories)
        self.agent_game_ended = False
        self.agent_troop_gain = 0
        self.agent_gets_card = False
        self.players_agent_survived = 0 #
        self.players_agent_eliminated = 0
        self.recurrence = False
        self.agent.cumulative_reward = 0
        self.agent.positive_reward_only = 0
        self.agent.negative_reward_only = 0
        get_troops(self.agent, self.territories)
        self.troops = self.agent.placeable_troops
        return self.get_state(), {}

    def get_state(self, verbose=False):
        prev_move = len(self.territories) if self.phase not in (2, 4) else self.prev_move
        state = get_state(self.players, self.territories, self.phase, self.troops, self.turns_passed, prev_move)
        if verbose:
            print(f"normalized troop count: {state[:self.num_players]}")
            print(f"normalized turns passed: {state[-3]}")
            print(f"phases: {state[-8:-3]}")
            print(f"prev move: {state[-2]}")
        return state

    def get_reward(self, illegal=False):
        first_place_bonus = 100
        placement_bonus = 0.5
        elimination_bonus = 0.25 # prefer to win (placement_bonus) over be flashy, but also, reward for engaging gameplay
        survival_bonus = 0.1

        if(illegal):
            reward = self.invalid_move_penalty - survival_bonus # dont let it pick a trade between two penalites
            self.agent.cumulative_reward += reward
            self.agent.negative_reward_only += reward
            return reward
        reward = 0



        # attack_phase = (self.phase == 1 or self.phase == 2) actually, dont need this
        # first_held_bonus = 0.25 # may want to cut this for simplicty
        # could add additional reward for maintaining a higher relative strength (which works great until we break non-zero sum assumption)

        if not self.agent_game_ended:
            if self.neutral_only:
                reward -= survival_bonus
            elif self.turns_passed < self.EARLY_GAME:
                reward += survival_bonus
            elif self.turns_passed > self.LATE_GAME:
                reward -= survival_bonus # ouch! the floor is lava! is this ethical ):

            reward += placement_bonus * self.players_agent_survived # placement bonus
            reward += (elimination_bonus + placement_bonus) * self.players_agent_eliminated

            #held a continent for the first time bonus could go here  (meh)

            if (self.agent_troop_gain > 0 and (self.turns_passed < self.EARLY_GAME or self.players_agent_eliminated > 0)):
                reward += survival_bonus #bonus for winning a cheap card
        elif self.agent_game_ended:
            # the Agent just eliminated someone and the game ended
            if self.agent.territory_count > 0:
                reward += self.players_agent_eliminated  * (elimination_bonus + placement_bonus)
                reward += first_place_bonus
            # the Agent was eliminated and lost, but possibly after another player after their turn ended
            # the Agent must not be considered one of the players eliminated, and this must be called at the right timing (right when Agent is eliminated, not at start of next turn sequence)
            else:
                reward += placement_bonus * self.players_agent_survived # placement bonus
                reward += (elimination_bonus + placement_bonus) * self.players_agent_eliminated

        #dont ever get the same rewards twice
        self.players_agent_survived = 0
        self.players_agent_eliminated = 0
        self.agent_troop_gain = 0
        self.agent.cumulative_reward += reward
        if reward > 0:
            self.agent.positive_reward_only += reward
        else:
            self.agent.negative_reward_only += reward

        return reward

    #experience tuple: e = (s,a,r,s', done)
    # next_state, reward, done, _ = env.step(action)
    # returns: next_state, reward, done, info
    def step(self, action, verbose=False):

        placement_phase = 0
        attack_source_phase = 1
        attack_target_phase = 2
        fortify_from_phase = 3
        fortify_to_phase = 4

        skip_phase = action == len(self.territories)
        skip_action = len(self.territories)
        territory = -1
        if not skip_phase:
            territory = action

        # Placement phase, place all troops
        def handle_placement_phase(self, action):
            if self.phase != placement_phase:
                raise "out of order phase handling"

            #check if we got eliminated inbetween our last turn
            self.agent_game_ended = (self.agent.territory_count == 0)
            if self.agent_game_ended:
                reward = self.get_reward()
                return  self.get_state(), reward, self.agent_game_ended, False, {}

            # cannot skip placement phase
            if action == skip_action:
                reward = self.get_reward(illegal=True)
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

            #are we in placement phase because of a player elimination? dont get troops
            if not self.recurrence:
                get_troops(self.agent, self.territories)
                assert(self.agent.placeable_troops != 0)

            #wait until 5 cards to trade because trades are higher (sans a more nuanced strategy)
            if(self.agent.hand.count >= 5):
                trade_cards(self.agent)
            if not place_troops(self.agent, self.territories[action], self.agent.placeable_troops ):
                reward = self.get_reward(illegal=True)
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

            #self.troops = 0
            assert(self.agent.placeable_troops == 0)
            reward = self.get_reward()
            self.phase = attack_source_phase # progress the environment state
            state_prime = self.get_state() # show the Agent the changes they made

            #by convention, the Agent is rewarded at the start of its phase

            #but with phasic credit assignment, some of the reward is received during attacks
            #however that doesnt matter here, if phasic or not, we get the reward at placement
            return state_prime, reward, self.agent_game_ended, False, {}

        def handle_attack_phase(self, action):
            if self.phase != attack_source_phase and self.phase != attack_target_phase:
                raise "out of order phase handling"

            # agent decides to skip the attack
            if action == skip_action:
                self.phase = fortify_from_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()
                return self.get_state(), reward, self.agent_game_ended, False, {}
            # if we are choosing where to attack from
            if self.phase == attack_source_phase:
                from_terr = self.territories[action]
                # if invalid
                if (from_terr.troop_count < 2 or from_terr.owner != self.agent):
                    self.phase = fortify_from_phase
                    reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}
                else:
                    self.prev_move = action
                    self.from_terr = from_terr
                    self.phase = attack_target_phase
                    reward = 0
                    if self.phasic_credit_assignment: reward = self.get_reward()
                    return self.get_state(), reward, self.agent_game_ended, False, {}
            elif self.phase == attack_target_phase:
                to_terr = self.territories[action]
                # if invalid
                if (to_terr not in self.from_terr.neighbors or to_terr.owner == self.agent):
                    self.phase = fortify_from_phase
                    self.from_terr = None
                    self.prev_move = len(self.territories)
                    reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}
                else:
                    other_player = to_terr.owner
                    attacking_army_size = self.from_terr.troop_count - 1
                    troops_lost_attacker, _, attacker_won, is_legal = attack_territory(
                        self.from_terr, to_terr, attacking_army_size, self.players,
                        reduce_kurtosis=False, verbose=False)
                    self.agent_troop_gain -= troops_lost_attacker #0 if illegal
                    self.from_terr = None
                    self.phase = attack_source_phase #can keep attacking if desired
                    if not is_legal:
                        if(verbose): print("Agent tried illegal attack")
                        return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}
                    if not self.agent_gets_card and attacker_won:
                        self.agent_gets_card = True
                        self.agent_troop_gain += self.CARD_TROOP_VALUE
                    if attacker_won and other_player.territory_count == 0:
                        self.agent_troop_gain += self.CARD_TROOP_VALUE * other_player.hand.count
                        self.players_agent_eliminated += 1
                        take_cards(self.agent, other_player)
                        if self.agent.hand.count >= 5:
                            self.phase = placement_phase
                            self.recurrence = True
                            trade_cards(self.agent)
                            self.troops = self.agent.placeable_troops
                        self.agent_game_ended = (self.agent.territory_count == len(self.territories))
                    #if the Agent won the game
                    if self.agent_game_ended:
                        self.phase = placement_phase # should not do anything...
                        reward = self.get_reward()
                        return self.get_state(), reward, self.agent_game_ended, False, {}
                    else:
                        reward = 0
                        if self.phasic_credit_assignment: reward = self.get_reward()
                        return self.get_state(), reward, self.agent_game_ended, False, {}

        def handle_fortify_phase(self, action):
            if self.phase != fortify_to_phase and self.phase != fortify_from_phase:
                raise "out of order phase handling"

            if action == skip_action:
                self.phase = placement_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()
                return self.get_state(), reward, self.agent_game_ended, False, {}
            if self.phase == fortify_from_phase:
                from_terr = self.territories[action]

                #attempting an illegal fortify
                if (from_terr.owner != self.agent or from_terr.troop_count < 2):
                    self.phase = placement_phase
                    reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

                self.from_terr = from_terr
                self.prev_move = action
                self.phase = fortify_to_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()
                return self.get_state(), reward, self.agent_game_ended, False, {}

            elif self.phase == fortify_to_phase:
                to_terr = self.territories[action]
                #attempting an illegal fortify
                if (to_terr == self.from_terr or to_terr.owner != self.agent or to_terr not in fortify_bfs(self.from_terr)):
                    self.phase = placement_phase
                    self.from_terr = None
                    reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}
                else:
                    troops = self.from_terr.troop_count - 1
                    fortify(self.from_terr, to_terr, troops)
                    self.from_terr = None
                    self.phase = placement_phase

                    # if the agent is ending their turn, let other players go
                    self.turns_passed += 1
                    assert(self.phase == placement_phase)
                    players_survived = 0
                    players_starting = sum([p.territory_count != 0 for p in self.players[1:]])
                    #Play all other agent moves
                    if not self.skip_bots:
                        for p in self.players[1:]:
                            if p.is_bot and p.territory_count > 0:
                                assert(p.make_move(self.players, self.territories, False) == True) #only allow legal moves from the other players
                                game_ended = self.agent.territory_count == 0
                                if(game_ended):
                                    break # we end right here to know the Agent's relative placement (this assumes more than 1 player isnt eliminated by a bot in the same turn...)
                        players_ending = sum([p.territory_count != 0 for p in self.players[1:]])
                        self.players_agent_survived += players_starting - players_ending
                    reward = 0
                    if self.phasic_credit_assignment: reward = self.get_reward()
                    return self.get_state(), reward, self.agent_game_ended, False, {}

        starting_phase = self.phase
        if self.phase == placement_phase:
            return handle_placement_phase(self, action)
        elif self.phase == attack_source_phase or self.phase == attack_target_phase:
            return handle_attack_phase(self, action)
        elif self.phase == fortify_from_phase or self.phase == fortify_to_phase:
            return handle_fortify_phase(self, action)
        else:
            raise "impossible phase reached"


    def show_board(self, blocking=True):
        board_graph = create_graph(self.territories, display=False)
        display_graph(board_graph, self.territories, title="Current board state", save=True, blocking_display=blocking)

#TODO 1:
# Collect state observation, package into experience tuple

# collect state observation to build experience tuple. the observation incudes:
# normalized troop count of all players
# binary truth of owneship of each territory by player
# binary toggles of Agent turn phase
# heuristics associated with each player (CAN COME BACK TO THIS)
# normalized value of troops generated per continent bonus per territory (MAYBE NOT??)
# normalized value of troops per territory, where 1 = 0

# this operation is expensive
# an improvement would be for the Agent to call this at the start of its turn, then
# for other operations to make changes only as needed during the Agent's turn

#  1. trade cards: S = get_state before trade, S' = get_state after placing troops
#  2. place troops -> rolled into 1.
#  3. attack phase: S = S' from prior phase, S' = get_state after attack
##### IF WE ARE CHECKING S' AFTER EVERY ATTACK, WE RLLY NEED THIS MORE EFFICIENT ######
# 4. fortify phase: S = S' from attack phase, S' - get_state after fortiy

# a good way to handle this is likely to set a copy of S as an input to the action functions
# and update the copy S into S'

# expects a list of 6 players including those eliminated or never initialized
# returns the current state S
# current convenction is Agent is always starting as player #1

# we have a contract with gym api to only provide values inside range between 0 and 1
# self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n, ))
def get_state(players, board, agents_phase, troops, turns_passed, previous_move):

    # Need to add damage dealt/received. Its necessary to meaningfully learn for
    # any configuration but neutral or random bots only

    #AVERAGE_GAME_LENGTH = 10 #random but close enough to online games probably. proper way would be batch normalized but...
    #turns_passed_normalized = [turns_passed / AVERAGE_GAME_LENGTH]
    #turns_passed_normalized = [min(1, min(turns_passed_normalized))] # because we cant exceed 1 and because we don't batch normalize

    phases = [0] * 5
    phases[agents_phase] = 1

    # need to fix troop count
    #troop_counts = [player.total_troops for player in players] # Get troop counts
    #max_troop_count = max(troop_counts) # Get max troop count
    #normalized_troop_counts = [count / max_troop_count for count in troop_counts] # Normalize troop count between 0 and 1

    #ownership = [[int(territory.owner_color == player.name) for territory in board] for player in players] #This should convert the boolean to an int 1 or 0 if owned
    #troops_in_territories = [territory.troop_count for territory in board]

    # Combine all observations into a single state
    #state = normalized_troop_counts + sum(ownership, []) + troops_in_territories + phases + turns_passed_normalized  + [previous_move, troops]

    agent_terrs = players[0].territories

    agent_troops = [0] * len(agent_terrs)
    i = 0
    for t in agent_terrs:
        if t == 1:
            agent_troops[i] = board[i].troop_count
        else:
            agent_troops[i] = -board[i].troop_count
        i += 1
    #print(agent_troops)
    # Maximum number of troops on any territory
    max_troops = max(agent_troops)
    agent_troops = [troop_count / max_troops for troop_count in agent_troops]

    state = phases + agent_troops + [turns_passed, previous_move]

    # Convert the state to a numpy array and reshape it to match the DQN agent's state shape
    state = np.array(state, dtype="float32")
    # state = state.reshape((1, len(state)))

    return state

#TODO 2: DQN

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import torch
import numpy as np

# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            torch.tensor(np.array(states)).float().to(self.device),
            torch.tensor(np.array(actions)).long().to(self.device),
            torch.tensor(np.array(rewards)).unsqueeze(1).float().to(self.device),
            torch.tensor(np.array(next_states)).float().to(self.device),
            torch.tensor(np.array(dones)).unsqueeze(1).int().to(self.device)
        )


    def __len__(self):
        return len(self.buffer)

# Define the Double DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3, capacity=1000000,
                 discount_factor=0.99, tau=1e-3, update_every=4, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity)
        self.update_target_network()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + self.discount_factor * (Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        return loss

    def update_target_network(self):
        # Update target network parameters with polyak averaging
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

def main(env_name, num_episodes=1000, save_interval=100, load_model=False):
    os.system('cls') # assuming windows, clear debug output
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    checkpoint_path = Path('checkpoints')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    #start making 10% random moves, decay to .05%, with 10% soft update
    #note that when the majority of moves are illegal, this is a high epsilon
    #an idea is to have a legality weighted epsilon but aint nobody got time for that
    #
    #execution slows down dramatically with lower epsilons, I think its because the Agent
    #is dumb and just makes illegal moves over and over
    #agent = DQNAgent(state_size, action_size, batch_size=8, gamma=0.9, epsilon=0.1, epsilon_min=0, epsilon_decay=0.999997, learning_rate=0.005, tau=0.05)
    agent = DQNAgent(state_size, action_size, seed=1)

    if load_model:
        agent.load(checkpoint_path / 'dqn_model.pth')

    optimize_ratio = 1 #how much are we re-using old data
    render = False
    render_frequency = 200000 #out of
    best_reward = -np.inf
    eps = eps_start = 0.3
    eps_end = 0.01
    for episode in range(num_episodes):
        state, _ = env.reset(seed=1)
        terminated = False
        total_reward = 0
        actions = 0
        illegal_moves = 0
        max_actions = 50000
        loss = 0
        loss_count = 0
        action_counts = {i: 0 for i in range(len(env.territories) + 1)}
        phase_counts = {i: 0 for i in range(5)}
        action_hist = []
        state_hist = [state]


        eps -= (eps_start - eps_end) / num_episodes
        #env.show_board()
        while not terminated:
            action = agent.act(state, eps)
            action_hist.append(action)
            action_counts[action] += 1
            phase_counts[env.phase] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated)
            state_hist.append(next_state)
            if reward <= env.invalid_move_penalty:
                illegal_moves += 1
            state = next_state
            total_reward += reward
            actions += 1 #why the flipping heck is this so high??
            if render and random.randint(1, render_frequency) == render_frequency:
                env.show_board() #should prolly just save this or something, blocks are kinda annoying


            if actions == max_actions: #but why is this happening in the first place
                break
        #env.show_board()
        #env.show_board()
        # Print to check if there's anything fishy
        # if illegal_moves == 0 and actions < 10:
        #     print(action_hist)
        #     for s in state_hist:
        #         print(s.tolist())

        #if optimize_ratio > 1 and (episode + 1) % ((num_episodes/optimize_ratio)) == 0:
        #    optimize_ratio -= 1 # recycle data less as the Agent advances
        #if (episode + 1) % save_interval == 0 or episode == 1:
        #    agent.save(checkpoint_path / f'dqn_model_{episode}.pth')

        #if total_reward > best_reward and episode > 100:
        #    best_reward = total_reward
        #    agent.save(checkpoint_path / f'dqn_model_best_{episode}.pth')  # filename includes episode number
        #    agent.save(checkpoint_path / f'dqn_model_best.pth')

        print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {(total_reward/actions):.4f}, Loss: {(loss/max(loss_count, 1)):.6f}, Epsilon: {eps:.4f}, Illegal Move Ratio: {illegal_moves/actions:.3f} Actions: {actions}")
        action_values = list(action_counts.values())
        max_action_count = max(action_counts, key=action_counts.get)
        print(f"Positive: {env.agent.positive_reward_only:.2f}, Negative: {env.agent.negative_reward_only:.2f}, Cumulative {env.agent.cumulative_reward:.2f}")
        print(f"Min Action Count: {min(action_values)}, Median Action Count: {statistics.median(action_values)}, Action {max_action_count}, has the maximum count: {action_counts[max_action_count]}")
        print(f"Placement Phase: {phase_counts[0]}, Attack Source: {phase_counts[1]}, Attack Target {phase_counts[2]}, Fortify From: {phase_counts[3]}, Fortify To: {phase_counts[4]}\n")

    render_final_results = True
    if render_final_results:
        #agent.load(checkpoint_path / f'dqn_model_best.pth')

        state, _ = env.reset(seed=1)
        terminated = False
        turns_passed = 0
        attempts = 0
        max_attempts = 1000
        print(env.agent.total_troops)
        env.show_board()
        eps = 0
        while not terminated and attempts < max_attempts:
            if env.turns_passed != turns_passed:
                input()
                turns_passed = env.turns_passed
            action = agent.act(state, eps)
            print(env.phase, action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            print(next_state)
            state = next_state
            # env.show_board()
            attempts += 1
        env.show_board()
        print(env.agent.total_troops)
        print(agent.steps)
    env.close()

if __name__ == "__main__":
    gym.envs.register(id='RiskEnvFlat-v0', entry_point=RiskEnvFlat)
    main('RiskEnvFlat-v0')
