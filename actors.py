from operator import itemgetter
import random
from atomic_actions import get_troops, place_troops, attack_territory, fortify, trade_cards, get_card, take_cards

######################################### Main Notes #########################################
# This .py defines Risk players as either an RL Agent (Player) or as a type of hard-coded opponent
# 
#  These methods should not make physically illegal moves as defined by atomic_actions.py
#  if they attempt an illegal move, they should raise exception 
#
# The hard-coded opponents do each control their own progression through turn-phases, 
# which is a potential area for bugs as there are no checks on their turn-progression 
##############################################################################################

class Player:
    def __init__(self, color, turn_order, num_territories):
        self.name = color
        self.turn_order = turn_order
        self.total_troops = 0
        self.placeable_troops = 0
        # binary truth array for efficient updates, also usable as NN input
        self.territories = [0] * num_territories
        self.territory_count = 0
        self.hand = Hand()
        self.key = turn_order
        self.is_bot = False

        self.damage_received = [0] * 6
        self.damage_dealt = [0] * 6

        #reward based attributes
        #self.has_held_bonus = False
        #self.got_reward_held_bonus = False
        self.cumulative_reward = 0
        self.positive_reward_only = 0
        self.negative_reward_only = 0

class Hand:
    def __init__(self):
        self.artillery = 0
        self.cavalry = 0
        self.soldier = 0
        self.wild = 0
        self.count = 0




#Tit-for-tat strategy
# class TFT_Bot(Player):


class Random_Bot(Player):
    def __init__(self, color, turn_order, num_territories, num_players):
        super().__init__(color, turn_order, num_territories)

        self.is_bot = True

    def make_move(self, players, territories, verbose=False):

        if self.territory_count < 1:
            return True #trying to make attack from dead bot, just return


        def handle_placement_phase(self, place_on = None):
            if self.hand.count >= 5:
                trade_cards(self)
                if(verbose): print(f"{self.name} is trading cards")
            if place_on:
                territory = place_on
                if verbose: print(f"{self.name} is placing {self.placeable_troops} in {place_on.name}")
                assert(place_troops(self, place_on, self.placeable_troops) == True)
                return place_on
            owned_territories = [i for i, t in enumerate(self.territories) if t == 1]
            territory = random.choice(owned_territories)
            if verbose: print(f"{self.name} is placing {self.placeable_troops} in {territories[territory].name}")
            assert(place_troops(self, territories[territory], self.placeable_troops) == True)
            return territories[territory]

        #returns the new territory if an attack won, None if one wasnt made or if it was lost
        def handle_attack_phase(self, attack_from_territory):
            valid_neighbor_list = []
            for neighbor in attack_from_territory.neighbors:
                if neighbor.owner != attack_from_territory.owner:
                    valid_neighbor_list.append(neighbor)
            if not valid_neighbor_list:
                if verbose: print(f"{from_terr.name} has no neighbors")
                return None
            valid_neighbor_list = sorted(valid_neighbor_list, key=lambda x: x.troop_count)
            to_terr = valid_neighbor_list[0] #attack the weakest neighbor
            #but not if they outnumber us or if we can't roll 3 dice
            if to_terr.troop_count > attack_from_territory.troop_count - 2 or from_terr.troop_count < 4:
                if verbose: print(f"{self.name} decided not to attack {to_terr.name} because of low troops")
                return None
            to_owner = to_terr.owner
            attacking_army_size = from_terr.troop_count - 1
            troops_lost_attacker, _, attacker_won, is_legal = attack_territory(
                attack_from_territory, to_terr, attacking_army_size, verbose=False)
            result = "lost"
            if attacker_won: result = "won"
            if verbose: print(f"{self.name} attacked {to_terr.name} and {result}")
            if not is_legal:
                print("attacking from", attack_from_territory.name, "owned by", attack_from_territory.owner, "with troops", attack_from_territory.troop_count)
                print("attacking to", to_terr.name, "owned by", to_terr.owner, "with troops", to_terr.troop_count)
            assert(is_legal)
            if attacker_won:
                if to_owner.territory_count == 0:
                    take_cards(self, to_owner)
                    if verbose: print(f"{self.name} eliminated {to_terr.owner.name} and took their cards, now they have {self.hand.count}")
                return to_terr
            return None

        def handle_fortify_phase(self, from_terr):
            max_count = 0
            max_neighbor = None
            for neighbor in from_terr.neighbors:
                if neighbor.owner == from_terr.owner:
                    troops = neighbor.troop_count
                    if troops > max_count:
                        max_count = troops
                        max_neighbor = neighbor
            if max_count != 0:
                fortify(from_terr, max_neighbor, from_terr.troop_count - 1)

        attacks_maximum = 3
        attacks_minimum = 0
        attacks_to_make = random.randint(attacks_minimum, attacks_maximum)
        attacks_made = 0

        get_troops(self, territories)

        from_terr = handle_placement_phase(self)
        wins_card = False
        while attacks_made < attacks_to_make:
            from_terr =  handle_attack_phase(self, from_terr)
            if from_terr == None:
                break
            wins_card = True
            attacks_made += 1
            if self.hand.count >= 5:
                from_terr = handle_placement_phase(self, place_on=from_terr)
                attacks_made = 0
        if from_terr:
            handle_fortify_phase(self, from_terr)
        if(wins_card):
            get_card(self.hand)

        return True

class Neutral_Bot(Player):
    def __init__(self, color, turn_order, num_territories):
        super().__init__(color, turn_order, num_territories)

        self.is_bot = True

    def make_move(self, players, territories, verbose=False):
        owned_territories = [i for i, t in enumerate(self.territories) if t == 1]
        #will never (practically, but it is possible for some online settings...) have cards to trade
        #will never eliminate a player
        get_troops(self, territories)
        territory = random.choice(owned_territories)
        if verbose: print(f"{self.name} is placing {self.placeable_troops} in {territories[territory].name}")
        return place_troops(self, territories[territory], self.placeable_troops)
