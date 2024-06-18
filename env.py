import gym
import numpy as np
import random
from random import seed
from gym.spaces import Box, Discrete
from collections import deque

from board import create_board, fortify_bfs, create_graph, display_graph, create_board_test
from atomic_actions import attack_territory, place_troops, take_cards, trade_cards, fortify, get_troops, get_card

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def can_attack(player, territories, agents_phase):
    if agents_phase != 1 and agents_phase != 2:
        return False
    territories_owned = [i for i, owned in enumerate(player.territories) if owned]
    territories_owned_obj = [territories[i] for i in territories_owned]
    
    for t in territories_owned_obj:
        if t.troop_count > 1:
            for n in t.neighbors:
                if n.owner != t.owner:
                    return True
    return False

def can_fortify(player, territories, agents_phase):
    if agents_phase != 3 and agents_phase != 4:
        return False
    territories_owned = [i for i, owned in enumerate(player.territories) if owned]
    territories_owned_obj = [territories[i] for i in territories_owned]

    for t in territories_owned_obj:
        if t.troop_count > 1 and len(fortify_bfs(t)) > 0:
            return True
    return False


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
        fortify_source_phase = 3
        fortify_target_phase = 4

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

        def handle_attack_source_phase(self, action):
            if self.phase != attack_source_phase:
                raise "out of order phase handling found in call to handle_attack_source_phase(self, action)"

            # agent decides to skip the attack
            if action == skip_action:
                self.phase = fortify_source_phase
                reward = 0
                if self.phasic_credit_assignment: reward = self.get_reward()
                return self.get_state(), reward, self.agent_game_ended, False, {}

            from_terr = self.territories[action]
            # if invalid selected source territory
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

        def handle_attack_target_phase(self, action):
            if self.phase != attack_target_phase:
                raise "out of order phase handling found in call to handle_attack_target_phase(self, action)"

            # agent decides to skip the attack
            if action == skip_action:
                # not letting agent skip attack_target phase as it must've not skipped attack_source to get here
                # this will cause redundant ways to get to the same state
                if verbose: print("Agent tried illegal skip during attack target phase")
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

            to_terr = self.territories[action]
            # if invalid
            if (to_terr not in self.from_terr.neighbors or to_terr.owner == self.agent):
                self.phase = attack_source_phase
                self.from_terr = None
                self.prev_move = len(self.territories)
                # reward = self.get_reward(illegal=True)
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}
            else:
                other_player = to_terr.owner
                attacking_army_size = self.from_terr.troop_count - 1
                troops_lost_attacker, _, attacker_won, is_legal = attack_territory(
                    self.from_terr, to_terr, attacking_army_size, verbose=False)
                self.agent_troop_gain -= troops_lost_attacker  # 0 if illegal
                self.from_terr = None
                self.phase = attack_source_phase  # can keep attacking if desired

                if not is_legal:
                    if verbose: print("Agent tried illegal attack")
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

                # if the Agent won the game
                if self.agent_game_ended:
                    self.phase = placement_phase  # should not do anything...
                    reward = self.get_reward()
                    return self.get_state(), reward, self.agent_game_ended, False, {}
                else:
                    reward = 0
                    if self.phasic_credit_assignment: reward = self.get_reward()
                    return self.get_state(), reward, self.agent_game_ended, False, {}

        def handle_fortify_source_phase(self, action):
            self.recurrence = False
            if self.phase != fortify_source_phase:
                raise "out of order phase handling found in call to handle_fortify_source_phase(self, action)"

            if self.agent_gets_card:
                get_card(self.agent.hand)
                self.agent_gets_card = False

            if action == skip_action:
                reward = 0
                self.phase = placement_phase

            # attempting an illegal fortify
            if self.phase == fortify_source_phase:
                from_terr = self.territories[action]
                if (from_terr.owner != self.agent or from_terr.troop_count < 2):
                    self.phase = fortify_source_phase
                    # reward = self.get_reward(illegal=True)
                    return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

                self.from_terr = from_terr
                self.prev_move = action
                self.phase = fortify_target_phase
                reward = 0

            if self.phase == placement_phase:
                state, new_reward, terminated, truncated, info = self.handle_other_players()
            else:
                state, new_reward, terminated, truncated, info = self.get_state(), 0, self.agent_game_ended, False, {}
            return state, reward + new_reward, terminated, truncated, info

        def handle_fortify_target_phase(self, action):
            self.recurrence = False
            if self.phase != fortify_target_phase:
                raise "out of order phase handling found in call to handle_fortify_target_phase(self, action)"

            if action == skip_action:
                # not letting agent skip fortify_target phase as it must've not skipped fortifysource to get here
                # this will cause redundant ways to get to the same state
                self.phase = fortify_source_phase
                if verbose: print("Agent tried illegal skip during fortify phase")
                return self.get_state(), self.invalid_move_penalty, self.agent_game_ended, False, {}

            to_terr = self.territories[action]
            # attempting an illegal fortify
            if (to_terr == self.from_terr or to_terr.owner != self.agent or to_terr not in fortify_bfs(self.from_terr)):
                self.phase = fortify_source_phase
                self.from_terr = None
                reward = self.get_reward(illegal=True)
            else:
                troops = self.from_terr.troop_count - 1
                fortify(self.from_terr, to_terr, troops)
                self.from_terr = None
                self.phase = placement_phase
                reward = 0

            new_reward = 0
            if self.phase == placement_phase:
                state, new_reward, terminated, truncated, info = self.handle_other_players()
            else:
                state, new_reward, terminated, truncated, info = self.get_state(), 0, self.agent_game_ended, False, {}
            return state, reward + new_reward, terminated, truncated, info

        if self.phase == placement_phase:
            return handle_placement_phase(self, action)
        elif self.phase == attack_source_phase:
            return handle_attack_source_phase(self, action)
        elif self.phase == attack_target_phase:
            return handle_attack_target_phase(self, action)
        elif self.phase == fortify_source_phase:
            return handle_fortify_source_phase(self, action)
        elif self.phase == fortify_target_phase:
            return handle_fortify_target_phase(self, action)
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

    can_move = [can_attack(players[0], board, agents_phase), can_fortify(players[0], board, agents_phase)]

    state = phases + agent_troops + can_move + [previous_move]

    # Convert the state to a numpy array and reshape it to match the DQN agent's state shape
    state = np.array(state, dtype="float32")
    return state

#TODO 2: DQN

class DQN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=5, hidden_dim_max=512, hidden_dim_min=128):
        super(DQN, self).__init__()
        dim = hidden_dim_max
        layers = [nn.Linear(input_size, dim), nn.ReLU()]
        for i in range(num_layers-2):
            next = dim // 2 if dim / 2 >= hidden_dim_min else dim
            layers.append(nn.Linear(dim, next))
            layers.append(nn.ReLU())
            dim = next
        layers.append(nn.Linear(dim, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class DQNAgent:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 batch_size=64, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995, 
                 learning_rate=0.001, 
                 tau=0.9,
                 num_layers=5, 
                 hidden_dim_max=512, 
                 hidden_dim_min=128,
                 target_update_freq=1000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Change deque to array for O(1) random sampling
        self.memory_limit = 100000
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau  # Rk added
        self.model = DQN(state_size, action_size, num_layers, hidden_dim_max, hidden_dim_min).to(self.device)
        self.target_model = DQN(state_size, action_size, num_layers, hidden_dim_max, hidden_dim_min).to(self.device)  # RK added Target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.target_update_freq = target_update_freq
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

        batch = random.sample(self.memory, self.batch_size)
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

        if self.epsilon_decay is not None:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(torch.load(filepath, map_location=self.device))

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
