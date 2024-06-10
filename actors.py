from operator import itemgetter
import random
from atomic_actions import get_troops, place_troops, attack_territory, fortify, trade_cards, get_card, take_cards

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

#The legality of *when* to make a move is left up to the simulation calling these bots
#Otherwise, these methods will not make physically illegal moves as defined by atomic_actions.py
#if they have a bug, they return False, and the main environment should cease execution


#Tit-for-tat strategy
class TFT_Bot(Player):
    def __init__(self, color, turn_order, num_territories, num_players):
        super().__init__(color, turn_order, num_territories)

        self.is_bot = True
        self.num_players = num_players

        self.has_cataracts = True #bot only considers territories touching its borders if True
        self.damage_received = [0] * num_players
        self.damage_dealt = [0] * num_players
        self.RETALIATION_THRESHOLD = 10 # delta of troops lost to start retaliation
        self.min_attacks = 1
        self.max_attacks = 4

    def make_move(self, players, territories, verbose=False):
        if not self.has_cataracts:
            raise Exception("Bots without cataracts havent been made yet. ")

        retaliating = False
        target = None
        player_is_alive = [1] * self.num_players

        for player in players:
            if player.territory_count == 0:
                player_is_alive[player.turn_order] = 0

        # Construct a list of tuples. Each tuple consists of the player's index and
        # the difference between the damage the bot received from the player and
        # the damage the bot dealt to the player.
        damage_diffs = []
        for player in players:
            if player_is_alive[player.turn_order] == 1:
                damage_diff = self.damage_received[player.turn_order] - self.damage_dealt[player.turn_order]
                damage_diffs.append((player.turn_order, damage_diff))

        # Sort the list of tuples based on the second item (i.e., the damage difference)
        # in descending order.
        sorted_damage_diffs = sorted(damage_diffs, key=itemgetter(1), reverse=True)

        if not sorted_damage_diffs:
            raise Exception("Bot attempting to make a move, but no players are alive.")

        if sorted_damage_diffs[0][1] >= self.RETALIATION_THRESHOLD:
            retaliating = True
            target = players[sorted_damage_diffs[0][0]].name  # the target player is the player that caused the most damage


        #get_troops(self, self.territories)
        #because there is overlap with get_troops and other required calculations, we blow up that function here
        has_continent = False
        owned_territories = [i for i, t in enumerate(self.territories) if t == 1]
        territories_owned_obj = [territories[i] for i in owned_territories]
        new_troops = max(3, len(owned_territories) // 3)
        continents_map = {}
        for t in territories_owned_obj:
            continents_map[t.continent] = continents_map.get(t.continent, 0) + 1
        for cont in continents_map:
            if continents_map[cont] == len(cont.territories):
                new_troops += cont.bonus_troop_count
                has_continent = True
        player.placeable_troops += new_troops

        end_my_turn = False
        land_locked = False
        attacks_to_make = random.randint(self.min_attacks, self.max_attacks)
        attacks_made = 0
        while not end_my_turn:
            end_my_turn = True

            #trade cards if on 5
            if self.hand.count >= 5: trade_cards(self)

            #place troops onto the highest troop count
            owned_territories = [i for i, t in enumerate(self.territories) if t == 1]
            territory_with_most_troops = max(owned_territories, key=lambda x: territories[x].troop_count)
            assert(place_troops(self, territories[territory_with_most_troops], self.placeable_troops) == True)

            #handle case of retaliating or not
            #if eliminates a player resulting in 5+ cards, restart this loop
            active_territory = territories[territory_with_most_troops]
            target_territory = None

            if not retaliating:
                #progress towards taking the bonus with the highest troop count in it
                desired_continent = active_territory.continent
                if not has_continent and desired_continent.name != "Asia":
                    if (verbose): print(self.name, "Trying to take", desired_continent.name, "with up to", attacks_to_make, "attacks")
                    while attacks_to_make > attacks_made:
                        for neighbor in active_territory.neighbors:
                            if neighbor.owner != active_territory.owner and neighbor.continent == desired_continent:
                                target_territory = neighbor
                                attacking_troops =  active_territory.troop_count -1
                                troops_lost_attacker, _, attacker_won, is_legal = attack_territory(
                                    active_territory, target_territory, attacking_troops, players,
                                    reduce_kurtosis=False, verbose=False
                                )
                                if (verbose): print(self.name, "Decided to attack", target_territory.name, "using", attacking_troops, "troops")
                                assert(is_legal)
                                attacks_made += 1
                                active_territory = target_territory
                                break
                        land_locked = True
                        attacks_to_make = 0 #territory could not find a relevant target
                        break
                else:
                    if (verbose): print(self.name, "No real goals, just making some mostly random attacks on any available neighbor near biggest stack")
                    while attacks_to_make > attacks_made:
                        for neighbor in active_territory.neighbors:
                            if neighbor.owner != active_territory.owner:
                                target_territory = neighbor
                                attacking_troops =  active_territory.troop_count -1
                                troops_lost_attacker, _, attacker_won, is_legal = attack_territory(
                                    active_territory, target_territory, attacking_troops, players,
                                    reduce_kurtosis=False, verbose=False
                                )
                                if (verbose): print(self.name, "Decided to attack", target_territory.name, "using", attacking_troops, "troops")
                                assert(is_legal)
                                if(attacker_won):
                                    if neighbor.owner.territory_count == 0:
                                        take_cards(self, neighbor.owner)
                                        attacks_to_make = 0
                                attacks_made += 1
                                active_territory = target_territory
                                break
                        land_locked = True
                        attacks_to_make = 0 #territory could not find a relevant target
                        break

                if self.hand.count >= 5:
                    end_my_turn = False
                    continue
                else:
                    #fortify
                    to_fortify = (active_territory.troop_count > 1 and land_locked)
                    if to_fortify:
                        if (verbose): print(self.name, "Decided to fortify due to not having a good target, coming from", active_territory.name)
                    if to_fortify:
                        can_fortify = False
                        fortify_to =  active_territory.neighbors[0]
                        max_activity = 0
                        for neighbor in active_territory.neighbors:
                            activity = 0
                            if neighbor.owner == active_territory.owner:
                                can_fortify = True
                                for far_neighbor in neighbor.neighbors:
                                    if far_neighbor.owner != active_territory.owner:
                                        activity += 1
                                if activity > max_activity:
                                    max_activity = activity
                                    fortify_to = neighbor
                        if can_fortify:
                            if (verbose): print(self.name, "Fortifying to", fortify_to.name)
                            fortify(active_territory, fortify_to, active_territory.troop_count - 1)
                        else:
                            if (verbose): print(self.name, "wanted  to fortify, but could not")
                    else:
                        if (verbose): print(self.name, "Decided not to fortify")
            else:
                if (verbose): print(self.name, "TRYING TO RETALIATE, NOT IMPLEMENTED")
                pass
        return True


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
                attack_from_territory, to_terr, attacking_army_size, players,
                reduce_kurtosis=False, verbose=False)
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
