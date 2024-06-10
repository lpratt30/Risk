import gym

import numpy as np
import random
from random import seed
from gym.spaces import Box, Discrete, Dict, Tuple
from pathlib import Path
import re
import matplotlib.pyplot as plt


from board import create_board, fortify_bfs, create_graph, display_graph, create_board_test
from atomic_actions import attack_territory, place_troops, take_cards, trade_cards, fortify, get_troops, get_card

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
import time

import statistics
import os


class RiskEnvFlat(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}  # Provide a default empty configuration dictionary if not provided
        self.num_players = env_config.get("num_players", 2)
        self.colors = env_config.get("colors", ['red', 'blue', 'green', 'yellow', 'purple', 'pink'][:self.num_players])
        self.board_size = env_config.get("size", 0)
        self.invalid_move_penalty = -10

        #bot properties
        # self.bot_types =  ["TFT", "Random", "TFT", None, "Neutral"]  # corresponds to turn order (indexed at Agent + 1), None is just a Player
        self.bot_types = env_config.get("bot_types", ["Neutral"] * (self.num_players - 1))
        if all(bot == "Neutral" for bot in self.bot_types): # if neutral only, immediately apply negative reward to encourage faster game ends
            self.neutral_only = True
        else:
            self.neutral_only = False
        self.shuffle_bots = True # False likely teaches arbitrary overfitting to each bot based on turn order
        if self.shuffle_bots: random.shuffle(self.bot_types)
        self.skip_bots = False

        #environment properties
        self.continents, self.territories, self.players = create_board_test(
            self.num_players,
            self.bot_types,
            colors=self.colors,
            size=self.board_size
        )
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
        self.LATE_GAME = 10 # turn at which to start punishments for taking too long (Agent gets bored)

    def reset(self, seed=None, options=None):
        #initlize board and get state
        if seed is not None:
            random.seed(seed)
        if self.shuffle_bots: random.shuffle(self.bot_types)
        self.continents, self.territories, self.players = create_board_test(
            self.num_players,
            self.bot_types,
            colors=self.colors,
            size=self.board_size
        )
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
        first_place_bonus = 200
        placement_bonus = 0.5
        elimination_bonus = 0.25 # prefer to win (placement_bonus) over be flashy, but also, reward for engaging gameplay
        survival_bonus = 1

        if(illegal):
            reward = self.invalid_move_penalty
            if self.players[0].territory_count == 0:
                reward -= first_place_bonus
            if self.turns_passed > self.LATE_GAME:
                penalty = survival_bonus * (self.turns_passed - self.LATE_GAME)
                reward -= penalty
            self.agent.cumulative_reward += reward
            self.agent.negative_reward_only += reward
            return reward
        reward = 0

        if self.turns_passed > self.LATE_GAME:
            penalty = survival_bonus * (self.turns_passed - self.LATE_GAME)
            reward -= penalty

        reward += (self.players[0].territory_count - self.players[1].territory_count) * survival_bonus
        # attack_phase = (self.phase == 1 or self.phase == 2) actually, dont need this
        # first_held_bonus = 0.25 # may want to cut this for simplicty
        # could add additional reward for maintaining a higher relative strength (which works great until we break non-zero sum assumption)

        if not self.agent_game_ended:
            if self.phase == 0: #dont repeatedly give survival bonuses across all phases
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
                reward -= first_place_bonus
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
                self.agent.placeable_troops = 0
                reward = self.get_reward(illegal=True)
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

            #self.troops = 0
            assert(self.agent.placeable_troops == 0)
            self.phase = attack_source_phase # progress the environment state
            state_prime = self.get_state() # show the Agent the changes they made

            #by convention, the Agent is rewarded at the start of its phase
            reward = self.get_reward()
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
                    self.phase = attack_source_phase
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
                    self.phase = attack_source_phase
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
            terminated = False
            truncated = False
            state = None
            info = {}
            if self.phase != fortify_to_phase and self.phase != fortify_from_phase:
                raise "out of order phase handling"

            if self.agent_gets_card:
                get_card(self.agent.hand)
                self.agent_gets_card = False

            if action == skip_action:
                self.phase = placement_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()

            if self.phase == fortify_from_phase:
                from_terr = self.territories[action]

                #attempting an illegal fortify
                if (from_terr.owner != self.agent or from_terr.troop_count < 2):
                    self.phase = fortify_from_phase
                    reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

                self.from_terr = from_terr
                self.prev_move = action
                self.phase = fortify_to_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()

            elif self.phase == fortify_to_phase:
                to_terr = self.territories[action]
                #attempting an illegal fortify
                if (to_terr == self.from_terr or to_terr.owner != self.agent or to_terr not in fortify_bfs(self.from_terr)):
                    self.phase = fortify_from_phase
                    self.from_terr = None
                    reward = self.get_reward(illegal=True)
                else:
                    troops = self.from_terr.troop_count - 1
                    fortify(self.from_terr, to_terr, troops)
                    self.from_terr = None
                    self.phase = placement_phase
                    reward = 0
                    self.handle_other_players()


            return self.get_state(), reward, terminated, truncated, info

        starting_phase = self.phase
        if self.phase == placement_phase:
            return handle_placement_phase(self, action)
        elif self.phase == attack_source_phase or self.phase == attack_target_phase:
            return handle_attack_phase(self, action)
        elif self.phase == fortify_from_phase or self.phase == fortify_to_phase:
            return handle_fortify_phase(self, action)
        else:
            raise "impossible phase reached"

    # if the agent is ending their turn, let other players go
    def handle_other_players(self):
        self.turns_passed += 1
        players_starting = sum([p.territory_count != 0 for p in self.players[1:]])
        reward = 0
        #Play all other agent moves
        if not self.skip_bots:
            for p in self.players[1:]:
                if p.is_bot and p.territory_count > 0:
                    assert(p.make_move(self.players, self.territories, False) == True) #only allow legal moves from the other players
                    game_ended = self.agent.territory_count == 0
                    if(game_ended):
                        self.agent_game_ended = game_ended
                        break # we end right here to know the Agent's relative placement (this assumes more than 1 player isnt eliminated by a bot in the same turn...)
            players_ending = sum([p.territory_count != 0 for p in self.players[1:]])
            self.players_agent_survived += players_starting - players_ending
        if self.phasic_credit_assignment: reward = self.get_reward()
        return self.get_state(), reward, self.agent_game_ended, False, {}

    def show_board(self, blocking=True):
        board_graph = create_graph(self.territories, display=False)
        display_graph(board_graph, self.territories, title="Current board state", save=True, blocking_display=blocking)


def get_state(players, board, agents_phase, troops, turns_passed, previous_move):
    phases = [0] * 5
    phases[agents_phase] = 1
    agent_terrs = players[0].territories

    agent_troops = [0] * len(agent_terrs)
    i = 0
    for t in agent_terrs:
        if t == 1:
            agent_troops[i] = board[i].troop_count
        else:
            agent_troops[i] = -board[i].troop_count
        i += 1
    # Maximum number of troops on any territory
    max_troops = max([abs(x) for x in agent_troops])
    agent_troops = [troop_count / max_troops for troop_count in agent_troops]

    state = phases + agent_troops + [turns_passed]

    # Convert the state to a numpy array and reshape it to match the DQN agent's state shape
    state = np.array(state, dtype="float32")
    return state

#TODO 2: DQN

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import torch
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, tau=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Change deque to array for O(1) random sampling
        self.memory_limit = 50000
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau  # Rk added
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)  # RK added Target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.target_update_freq = 1000
        self.steps = 0

    def remember(self, state, action, reward, next_state, terminated, truncated):
        if len(self.memory) < self.memory_limit:
            self.memory.append((state, action, reward, next_state, terminated, truncated))
        else:
            replace_index = self.steps % self.memory_limit # O(1) replacement
            self.memory[replace_index] = (state, action, reward, next_state, terminated, truncated)

    def act(self, state, eval=False):
        if not eval and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def optimize_network(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.choices(self.memory, k=self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, term_batch, trunc_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        term_batch = torch.FloatTensor(np.array(term_batch)).to(self.device)
        trunc_batch = torch.FloatTensor(np.array(trunc_batch)).to(self.device)

        q_values = self.model(state_batch)
        next_q_values = self.target_model(next_state_batch)
        target_q_values = reward_batch + (1 - term_batch) * self.gamma * torch.max(next_q_values, dim=1)[0]

        q_value = torch.gather(q_values, 1, action_batch.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_value, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # RK added soft update of target network
        if self.steps % self.target_update_freq == 0:
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        self.steps += 1

        return loss.item()


    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(torch.load(filepath, map_location=self.device))

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

def evaluate(agent, env):
    agent.epsilon = 0
    state, _ = env.reset()
    actions = 0
    max_actions = 500
    total_reward = 0
    illegal_moves = 0
    terminated = False
    while not terminated and actions < max_actions:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        actions += 1
        total_reward += reward
        if reward == env.invalid_move_penalty:
            illegal_moves += 1
    opponents_alive = 0
    for p in env.players[1:]:
        if p.territory_count > 0:
            opponents_alive += 1
    result = None
    if opponents_alive == 0: result  = "win"
    elif env.agent.territory_count == 0: result = "lose"
    else: result = "draw"
    print(f"Evaluation Reward: {(total_reward):.3f}, Illegal Move Ratio: {illegal_moves/actions:.3f} Result: {result}")

def create_output_graphs(num_episodes, num_players, bot_types, size, output_lists):
    sizes = ["Small", "Medium", "Large"]
    bots = ', '.join(set(bot_types))
    output_path = Path('outputs')
    output_path.mkdir(parents=True, exist_ok=True)
    names = ["Average Reward", "Cumulative Reward", "Loss", "Illegal Move Ratio", "Number of Actions", "Number of Turns", "Episode Time"]
    def create_graph(output, name):
        plt.plot(range(1, num_episodes+1), output)
        plt.xlabel("Episode")
        plt.ylabel(name)
        plt.title(f"{name} - {sizes[size]} Board; {num_players} Players; {bots} Bots")
        plt.savefig(output_path / f"{sizes[size].lower()}_{num_players}_{'_'.join(set(bot_types)).lower()}_{name.lower().replace(' ', '_')}.png")
        plt.clf()
    for out, name in zip(output_lists, names):
        create_graph(out, name)

def main(env_name, num_episodes=3500, save_interval=100, load_model=False):
    os.system('cls') # assuming windows, clear debug output
    num_players = 2
    bot_types = ["Random"] * (num_players - 1)
    size = 2
    env = gym.make(env_name, env_config={"num_players": num_players, "size": size, "bot_types": bot_types})
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
    agent = DQNAgent(state_size, action_size, batch_size=64, gamma=0.99, epsilon=1, epsilon_min=0, epsilon_decay=0.999997, learning_rate=0.00005, tau=0.001)

    if load_model:
        agent.load(checkpoint_path / 'dqn_model_best.pth')
    print("I am using device", agent.device)

    optimize_ratio = agent.batch_size #how much are we re-using old data
    render = False
    render_frequency = 200000 #out of
    eval_freq = 100
    best_reward = -np.inf
    avg_rewards, cum_rewards, losses, illegal_move_ratios, num_actions, num_turns, ep_time = tuple([] for i in range(7))
    state, _ = env.reset()
    num_oscillations = 10  # number of full oscillations over num_episodes
    eps = eps_start = 1
    eps_end = 0.01
    print(env.bot_types)
    env.show_board()
    for episode in range(num_episodes):
        amplitude = eps_start - (episode / num_episodes) * (eps_start - eps_end)
        phase = episode / num_episodes * num_oscillations * 2 * np.pi
        eps = eps_end + amplitude * np.sin(phase)
        eps = abs(eps)
        agent.epsilon = eps
        if episode % eval_freq == 0:
            state, _ = env.reset()
            evaluate(agent, env)
        state, _ = env.reset()
        terminated = False
        total_reward = 0
        actions = 0
        illegal_moves = 0
        max_actions = 1200
        loss = 0
        loss_count = 0
        action_counts = {i: 0 for i in range(len(env.territories) + 1)}
        phase_counts = {i: 0 for i in range(5)}
        action_hist = []
        state_hist = [state]
        start = time.time()

        while not terminated:
            action = agent.act(state)

            action_hist.append(action)
            action_counts[action] += 1
            phase_counts[env.phase] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            state_hist.append(next_state)
            if reward == env.invalid_move_penalty:
                illegal_moves += 1
            reward /= 100
            agent.remember(state, action, reward, next_state, terminated, truncated)
            state = next_state
            total_reward += reward
            actions += 1 #why the flipping heck is this so high??
            if render and random.randint(1, render_frequency) == render_frequency:
                env.show_board() #should prolly just save this or something, blocks are kinda annoying
            if actions % agent.batch_size == 0:
                for i in range(optimize_ratio):
                    loss += agent.optimize_network()
                    loss_count += 1

            if actions == max_actions: #but why is this happening in the first place
                break
        # Print to check if there's anything fishy
        # if illegal_moves == 0 and actions < 10:
        #     print(action_hist)
        #     for s in state_hist:
        #         print(s.tolist())

        #if optimize_ratio > 1 and (episode + 1) % ((num_episodes/optimize_ratio)) == 0:
        #    optimize_ratio -= 1 # recycle data less as the Agent advances
        if (episode + 1) % save_interval == 0 or episode == 1:
            agent.save(checkpoint_path / f'dqn_model_{episode}.pth')

        if total_reward > best_reward and episode > save_interval:
            best_reward = total_reward
            agent.save(checkpoint_path / f'dqn_model_best_{episode}.pth')  # filename includes episode number
            agent.save(checkpoint_path / f'dqn_model_best.pth')

        ep_time.append(time.time() - start)
        print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {(total_reward/actions):.4f}, Loss: {(loss/max(loss_count, 1)):.6f}, Epsilon: {agent.epsilon:.4f}, Illegal Move Ratio: {illegal_moves/actions:.3f} Actions: {actions}")
        print(f"Turns passed {env.turns_passed}, Agent territories remaining {env.agent.territory_count}, Opponent territories remaining {len(env.territories) - env.agent.territory_count}")
        action_values = list(action_counts.values())
        max_action_count = max(action_counts, key=action_counts.get)
        print(f"Positive: {env.agent.positive_reward_only:.2f}, Negative: {env.agent.negative_reward_only:.2f}, Cumulative {env.agent.cumulative_reward:.2f}")
        print(f"Min Action Count: {min(action_values)}, Median Action Count: {statistics.median(action_values)}, Action {max_action_count}, has the maximum count: {action_counts[max_action_count]}")
        print(f"Placement Phase: {phase_counts[0]}, Attack Source: {phase_counts[1]}, Attack Target {phase_counts[2]}, Fortify From: {phase_counts[3]}, Fortify To: {phase_counts[4]}\n")
        avg_rewards.append(total_reward / actions)
        cum_rewards.append(env.agent.cumulative_reward)
        losses.append(loss/max(loss_count, 1))
        illegal_move_ratios.append(illegal_moves/actions)
        num_actions.append(actions)
        num_turns.append(env.turns_passed + 1)

    out = (avg_rewards, cum_rewards, losses, illegal_move_ratios, num_actions, num_turns, ep_time)
    create_output_graphs(num_episodes, num_players, bot_types, size, out)
    render_final_results = True
    if render_final_results:
        agent.load(checkpoint_path / f'dqn_model_best.pth')
        state, _ = env.reset()
        terminated = False
        turns_passed = 0
        env.show_board()
        while not terminated:
            if env.turns_passed != turns_passed:
                turns_passed = env.turns_passed
                env.show_board()
            action = agent.act(state, eval=True)
            # print(env.phase, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
        print(f"Cumulative rewards: {env.agent.cumulative_reward}")
    env.close()

if __name__ == "__main__":
    gym.envs.register(id='RiskEnvFlat-v0', entry_point=RiskEnvFlat)
    main('RiskEnvFlat-v0')
