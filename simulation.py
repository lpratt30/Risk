import random
import numpy as np
import time

from board import create_board, create_graph, display_graph, Territory, find_shortest_path
from atomic_actions import attack_territory, get_card, trade_cards, fortify, take_cards, check_cards, place_troops
from board import Player, fortify_bfs

import os

######################################### Main Notes #########################################
# looking at top of output, player colors or territory owners looks decoupled from expectation
# on some seeds (5000), there is error "UnboundLocalError: local variable 'attacker_won' referenced before assignment"
# but not others. seed is set to working seed 1 now.
##############################################################################################



# Takes number of players as an argument
# Pulled in board to create, graph and display
def simulate_game(num_players, bot_types):
    os.system('cls') # assuming windows, clear debug output
    print("START SIMULATION \n")
    random.seed(1)
    continents, territories, players = create_board(num_players, bot_types)
    board_graph = create_graph(territories, display=False)
    display_graph(board_graph, territories, title="Initialized board", save=True, blocking_display=True)

    winning_player = None
    turn_count = 0
    max_turn_count = 1000

    while True and turn_count < max_turn_count:
        for player in players:
            if player.territory_count == 0:
                continue

            territories_owned = [i for i, owned in enumerate(player.territories) if owned]
            territories_owned_obj = [territories[i] for i in territories_owned]

            print(player.name +" player owns:", end=' ')
            for t in territories_owned_obj:
                print(t.owner.name + "/" + t.name, end=' ')
            print(" ")

            # Made a check to see if players own all territories and therefore win the game
            if player.territory_count == len(territories):
                winning_player = player.name
                break

            # Placement phase
            new_troops = max(3, len(territories_owned) // 3)
            continents_map = {}
            for t in territories_owned_obj:
                continents_map[t.continent] = continents_map.get(t.continent, 0) + 1
            for cont in continents_map:
                if continents_map[cont] == len(cont.territories):
                    new_troops += cont.bonus_troop_count

            # Trade in card whenever possible
            # Doing this instead of max troops or forced to helps game proceed
            attacking = True
            while attacking:
                attacking = False
                trade_cards(player)
                player.placeable_troops += new_troops

                # Randomly place troops into territories neighboring other players
                territory_choices = [t for t in territories_owned_obj if any([t.owner != n.owner for n in t.neighbors])]
                if len(territory_choices) == 0:
                    continue
                while new_troops > 0:
                    territory = random.choice(territory_choices)
                    add_troops = random.randint(new_troops, new_troops)
                    print(f"Player {player.name} places {new_troops} troops into {territory.name} (owner: {territory.owner.name})")
                    is_legal = place_troops(player, territory, add_troops)
                    assert(is_legal)
                    new_troops -= add_troops

                # Attack phase
                # Search for any possible attack
                attack_possible = False
                territories_owned_copy = territories_owned.copy()
                random.shuffle(territories_owned_copy)
                for from_territory_index in territories_owned_copy:
                    from_territory = territories[from_territory_index]

                    # Check if there are enough troops to attack
                    if from_territory.troop_count < 2:
                        continue

                    # Get neighboring territories
                    neighbor_territories = from_territory.neighbors
                    if not neighbor_territories:
                        continue

                    # Select a random neighboring territory to attack
                    random.shuffle(neighbor_territories)
                    to_territory = None
                    for territory in neighbor_territories:
                        if territory.owner != from_territory.owner:
                            to_territory = territory
                            break

                    # If there's a potential attack, stop searching
                    if to_territory is not None:
                        attack_possible = True
                        break
                if not attack_possible:
                    continue
                # Determine the number of troops to attack with
                troops_to_attack_with = from_territory.troop_count - 1

                print(f"Player {player.name} attacks {to_territory.name} (owner: {to_territory.owner.name}) from {from_territory.name} (owner: {from_territory.owner.name}) with {troops_to_attack_with} troops.")

                # Attack the territory, from the atomic_actions.py file
                other_player = to_territory.owner
                _, _, attacker_won, is_legal = attack_territory(
                    from_territory, to_territory, troops_to_attack_with,
                    players, reduce_kurtosis=False, verbose=False
                )

                assert(is_legal)

                #DEBBUGING ONLY LINE
                if attacker_won: assert(from_territory.owner == to_territory.owner)

                if attacker_won and other_player.territory_count == 0:
                    take_cards(player, other_player)
                    if player.hand.count >= 5:
                        attacking = True

            # Fortify phase, fortify from and to random territories for simplicity
            fortify_from = [t for t in territories_owned_obj if t.troop_count > 1 and any(t.owner == n.owner for n in t.neighbors)]
            if len(fortify_from) >= 1:
                fortify_from_territory = random.choice(fortify_from)
                fortify_to = fortify_bfs(fortify_from_territory)
                fortify_to_territory = random.choice(fortify_to)
                fortify_troops = fortify_from_territory.troop_count - 1  # Move max for simplicity
                fortify(fortify_from_territory, fortify_to_territory, fortify_troops)
                print(f"Player {player.name} fortifies {fortify_troops} troops from {fortify_from_territory.name} to {fortify_to_territory.name}")

            if attacker_won:
                get_card(player.hand)

        if winning_player:
            print("Player", winning_player, "wins!")
            display_graph(board_graph, territories, title="Final board state", save=True, blocking_display=True)
            return

        print("Territory ownership:")
        for player in players:
            print("Player", player.name, "owns", player.territory_count, "territories.")

        turn_count += 1

    display_graph(board_graph, territories, title="Final board state", save=True, blocking_display=True)

if __name__ == "__main__":
    print("Board initialized")
    num_players = 6  # Adjust the number of players here
    bot_types =   [None, None, None, None, None]
    simulate_game(num_players, bot_types)
