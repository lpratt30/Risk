import networkx as nx
import matplotlib.pyplot as plt
import random
from actors import Player, Hand, Random_Bot, Neutral_Bot

pos = {
    'Alaska': (0, 10),
    'Northwest Territory': (1, 9),
    'Greenland': (4, 10),
    'Alberta': (1, 8),
    'Ontario': (2, 8),
    'Quebec': (3, 8),
    'Western US': (1, 7),
    'Eastern US': (2, 7),
    'Central America': (2, 6),
    'Iceland': (5, 10),
    'Scandinavia': (6, 10),
    'Ukraine': (7, 10),
    'Great Britain': (5, 9),
    'Northern Europe': (6, 9),
    'Western Europe': (5, 8),
    'Southern Europe': (6, 8),
    'Ural': (8, 9),
    'Siberia': (9, 9),
    'Yakutsk': (10, 9),
    'Kamchatka': (11, 9),
    'Irkutsk': (10, 8),
    'Mongolia': (9, 8),
    'Japan': (11, 8),
    'Afghanistan': (8, 8),
    'China': (9, 7),
    'Middle East': (7, 7),
    'India': (8, 7),
    'Siam': (9, 6),
    'Venezuela': (2, 5),
    'Peru': (2, 4),
    'Brazil': (3, 4),
    'Argentina': (2, 3),
    'North Africa': (5, 5),
    'Egypt': (6, 5),
    'East Africa': (6, 4),
    'Congo': (5, 4),
    'South Africa': (5, 3),
    'Madagascar': (7, 3),
    'Indonesia': (10, 5),
    'New Guinea': (11, 5),
    'West Australia': (10, 4),
    'East Australia': (11, 4)
}
class Territory:
    def __init__(self, name, continent, owner = None, owner_color=None):
        self.name = name
        self.continent = continent
        self.owner = owner
        self.owner_color = owner_color
        self.troop_count = 0
        self.key = -1
        self.neighbors = []
        self.neighbor_names = []

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.neighbor_names.append(neighbor.name)

class Continent:
    def __init__(self, name, bonus_troop_count):
        self.name = name
        self.bonus_troop_count = bonus_troop_count
        self.territories = []

    def add_territory(self, territory):
        if territory not in self.territories:
            self.territories.append(territory)



def initialize(territories, players, player_colors):
    #Determine initial troop count based on the number of players
    num_players = len(players)
    initial_troops = -1
    if num_players == 6:
        initial_troops = 20
    elif num_players == 5:
        initial_troops = 25
    elif num_players == 4:
        initial_troops = 30
    elif num_players == 3:
        initial_troops = 35
    elif num_players == 2:
        initial_troops = 40
    else:
        raise ValueError("Invalid number of players")

    for p in players:
        p.total_troops = initial_troops

    # the following is only pseudo random
    #
    # Create a shuffled copy of players
    shuffled_players = players.copy()
    random.shuffle(shuffled_players)

    # Assign territories to the shuffled players
    for i, territory in enumerate(territories):
        territory.key = i
        territory.owner = shuffled_players[i % num_players]
        territory.owner_color = shuffled_players[i % num_players].name
        shuffled_players[i % num_players].territory_count += 1
        shuffled_players[i % num_players].territories[i] = 1


    #Assign initial troops to territories
    remaining_troops = [initial_troops for _ in range(num_players)]

    #Add at least one troop to each territory
    for territory in territories:
        player_index = player_colors.index(territory.owner_color)
        territory.troop_count += 1
        remaining_troops[player_index] -= 1

    while not all(val == 0 for val in remaining_troops):
        for i, territory in enumerate(territories):
            player = i % num_players
            if remaining_troops[player] > 0:
                if (territory.troop_count == 1 and random.random() < 0.25) or \
                   (territory.troop_count > 1 and random.random() < 0.50):
                    territory.troop_count += 1
                    remaining_troops[player] -= 1
            if all(val == 0 for val in remaining_troops):
                break
    return

# colors must be passed with the same ordering as player turns
def create_board(num_players, bot_types, colors=None):
    continents = [
        Continent('North America', 5),
        Continent('Europe', 5),
        Continent('Asia', 7),
        Continent('South America', 2),
        Continent('Africa', 3),
        Continent('Australia', 2),
    ]

    continent_index = {
        'North America': 0,
        'Europe': 1,
        'Asia': 2,
        'South America': 3,
        'Africa': 4,
        'Australia': 5
    }


    territories = [
        Territory('Alaska', continents[0]),
        Territory('Northwest Territory', continents[0]),
        Territory('Greenland', continents[0]),
        Territory('Alberta', continents[0]),
        Territory('Ontario', continents[0]),
        Territory('Quebec', continents[0]),
        Territory('Western US', continents[0]),
        Territory('Eastern US', continents[0]),
        Territory('Central America', continents[0]),

        Territory('Iceland', continents[1]),
        Territory('Scandinavia', continents[1]),
        Territory('Ukraine', continents[1]),
        Territory('Great Britain', continents[1]),
        Territory('Northern Europe', continents[1]),
        Territory('Western Europe', continents[1]),
        Territory('Southern Europe', continents[1]),

        Territory('Ural', continents[2]),
        Territory('Siberia', continents[2]),
        Territory('Yakutsk', continents[2]),
        Territory('Kamchatka', continents[2]),
        Territory('Irkutsk', continents[2]),
        Territory('Mongolia', continents[2]),
        Territory('Japan', continents[2]),
        Territory('Afghanistan', continents[2]),
        Territory('China', continents[2]),
        Territory('Middle East', continents[2]),
        Territory('India', continents[2]),
        Territory('Siam', continents[2]),

        Territory('Venezuela', continents[3]),
        Territory('Peru', continents[3]),
        Territory('Brazil', continents[3]),
        Territory('Argentina', continents[3]),

        Territory('North Africa', continents[4]),
        Territory('Egypt', continents[4]),
        Territory('East Africa', continents[4]),
        Territory('Congo', continents[4]),
        Territory('South Africa', continents[4]),
        Territory('Madagascar', continents[4]),

        Territory('Indonesia', continents[5]),
        Territory('New Guinea', continents[5]),
        Territory('West Australia', continents[5]),
        Territory('East Australia', continents[5]),
    ]


    territory_dict = {territory.name: territory for territory in territories}
    connections = [
        ('Alaska', 'Northwest Territory'),
        ('Alaska', 'Kamchatka'),
        ('Northwest Territory', 'Greenland'),
        ('Northwest Territory', 'Alberta'),
        ('Northwest Territory', 'Ontario'),
        ('Greenland', 'Quebec'),
        ('Greenland', 'Ontario'),
        ('Greenland', 'Iceland'),
        ('Alberta', 'Ontario'),
        ('Alberta', 'Western US'),
        ('Alberta', 'Alaska'),
        ('Ontario', 'Quebec'),
        ('Ontario', 'Western US'),
        ('Ontario', 'Eastern US'),
        ('Quebec', 'Eastern US'),
        ('Western US', 'Eastern US'),
        ('Western US', 'Central America'),
        ('Eastern US', 'Central America'),
        ('Central America', 'Venezuela'),

        ('Iceland', 'Scandinavia'),
        ('Iceland', 'Great Britain'),
        ('Scandinavia', 'Ukraine'),
        ('Scandinavia', 'Northern Europe'),
        ('Scandinavia', 'Great Britain'),
        ('Ukraine', 'Northern Europe'),
        ('Ukraine', 'Southern Europe'),
        ('Ukraine', 'Ural'),
        ('Ukraine', 'Middle East'),
        ('Ukraine', 'Afghanistan'),
        ('Great Britain', 'Northern Europe'),
        ('Great Britain', 'Western Europe'),
        ('Northern Europe', 'Western Europe'),
        ('Northern Europe', 'Southern Europe'),
        ('Western Europe', 'Southern Europe'),
        ('Southern Europe', 'Middle East'),

        ('Ural', 'Siberia'),
        ('Ural', 'Afghanistan'),
        ('Ural', 'China'),
        ('Siberia', 'Yakutsk'),
        ('Siberia', 'Irkutsk'),
        ('Yakutsk', 'Kamchatka'),
        ('Yakutsk', 'Irkutsk'),
        ('Kamchatka', 'Mongolia'),
        ('Kamchatka', 'Japan'),
        ('Irkutsk', 'Mongolia'),
        ('Mongolia', 'China'),
        ('Mongolia', 'Japan'),
        ('Afghanistan', 'China'),
        ('Afghanistan', 'Middle East'),
        ('Afghanistan', 'India'),
        ('China', 'India'),
        ('China', 'Siam'),
        ('China', 'Siberia'),
        ('India', 'Siam'),
        ('Middle East', 'India'),
        ('Middle East', 'Egypt'),
        ('Middle East', 'East Africa'),

        ('Venezuela', 'Brazil'),
        ('Venezuela', 'Peru'),
        ('Brazil', 'Peru'),
        ('Brazil', 'Argentina'),
        ('Brazil', 'North Africa'),
        ('Peru', 'Argentina'),

        ('North Africa', 'Egypt'),
        ('North Africa', 'East Africa'),
        ('North Africa', 'Congo'),
        ('North Africa', 'Western Europe'),
        ('North Africa', 'Southern Europe'),
        ('Egypt', 'East Africa'),
        ('Egypt', 'Southern Europe'),
        ('East Africa', 'Congo'),
        ('East Africa', 'South Africa'),
        ('East Africa', 'Madagascar'),
        ('Congo', 'South Africa'),
        ('South Africa', 'Madagascar'),

        ('Indonesia', 'New Guinea'),
        ('Indonesia', 'West Australia'),
        ('Indonesia', 'Siam'),
        ('New Guinea', 'West Australia'),
        ('New Guinea', 'East Australia'),
        ('West Australia', 'East Australia'),
    ]

    for territory1, territory2 in connections:
        territory_dict[territory1].add_neighbor(territory_dict[territory2])
        territory_dict[territory2].add_neighbor(territory_dict[territory1])  # Make the connection bidirectional


    for territory in territories:
        continents[continent_index[territory.continent.name]].add_territory(territory)

    player_colors = colors
    if not colors:
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink']
        player_colors = colors[:num_players]

    #players = []
    #for p in range(num_players):
    #    turn_order = p
    #    player = Player(player_colors[p], turn_order, len(territories))
    #    player.key = p
    #    players.append(player)
    #initialize(territories, players, player_colors)
    #return continents, territories, players

    players = []
    turn_order = 0
    player = Player(player_colors[0], turn_order, len(territories)) #this is the Agent, the red player
    players.append(player)
    assert(len(bot_types) == 5)
    for p in range(len(bot_types)):
        turn_order = p + 1 # because we exclusively reserve the first slot for the Agent
        if bot_types[p] is None:
            player = Player(player_colors[turn_order], turn_order, len(territories))
        elif bot_types[p] == "Neutral":
            player = Neutral_Bot(player_colors[turn_order], turn_order, len(territories))
        elif bot_types[p] == "Random":
            player = Random_Bot(player_colors[turn_order], turn_order, len(territories), num_players)
        elif bot_types[p] == "TFT":
            player = TFT_Bot(player_colors[turn_order], turn_order, len(territories), num_players)
        else:
            raise ValueError(f"Invalid bot type: {bot_types[p]}")

        player.key = turn_order
        players.append(player)

    assert(len(players) == 6)
    initialize(territories, players, player_colors)
    return continents, territories, players


def create_board_test(num_players, bot_types, colors=None, size=0):
    continents = [
        Continent('North America', 5),
        Continent('South America', 2),
    ]

    continent_index = {
        'North America': 0,
        'South America': 1,
    }

    territories = [
        Territory('Alaska', continents[0]),
        Territory('Northwest Territory', continents[0]),
        Territory('Alberta', continents[0]),
        Territory('Ontario', continents[0]),
    ]
    if size >= 1:
        territories += [
            Territory('Greenland', continents[0]),
            Territory('Quebec', continents[0]),
            Territory('Western US', continents[0]),
            Territory('Eastern US', continents[0]),
            Territory('Central America', continents[0]),
        ]
    if size == 2:
        territories += [
            Territory('Venezuela', continents[1]),
            Territory('Peru', continents[1]),
            Territory('Brazil', continents[1]),
            Territory('Argentina', continents[1]),
        ]

    territory_dict = {territory.name: territory for territory in territories}
    connections = [
        ('Alaska', 'Northwest Territory'),
        ('Northwest Territory', 'Alberta'),
        ('Northwest Territory', 'Ontario'),
        ('Alberta', 'Ontario'),
        ('Alberta', 'Alaska'),
    ]

    if size >= 1:
        connections += [
            ('Greenland', 'Quebec'),
            ('Greenland', 'Ontario'),
            ('Alberta', 'Western US'),
            ('Ontario', 'Quebec'),
            ('Ontario', 'Western US'),
            ('Ontario', 'Eastern US'),
            ('Quebec', 'Eastern US'),
            ('Western US', 'Eastern US'),
            ('Western US', 'Central America'),
            ('Eastern US', 'Central America'),
        ]
    if size == 2:
        connections += [
            ('Central America', 'Venezuela'),
            ('Venezuela', 'Brazil'),
            ('Venezuela', 'Peru'),
            ('Brazil', 'Peru'),
            ('Brazil', 'Argentina'),
            ('Peru', 'Argentina'),
        ]

    for territory1, territory2 in connections:
        territory_dict[territory1].add_neighbor(territory_dict[territory2])
        territory_dict[territory2].add_neighbor(territory_dict[territory1])  # Make the connection bidirectional


    for territory in territories:
        continents[continent_index[territory.continent.name]].add_territory(territory)

    player_colors = colors
    if not colors:
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink']
        player_colors = colors[:num_players]

    players = []
    turn_order = 0
    player = Player(player_colors[0], turn_order, len(territories)) #this is the Agent, the red player
    players.append(player)
    assert(len(bot_types) == (num_players - 1))
    for p in range(len(bot_types)):
        turn_order = p + 1 # because we exclusively reserve the first slot for the Agent
        if bot_types[p] is None:
            player = Player(player_colors[turn_order], turn_order, len(territories))
        elif bot_types[p] == "Neutral":
            player = Neutral_Bot(player_colors[turn_order], turn_order, len(territories))
        elif bot_types[p] == "Random":
            player = Random_Bot(player_colors[turn_order], turn_order, len(territories), num_players)
        elif bot_types[p] == "TFT":
            player = TFT_Bot(player_colors[turn_order], turn_order, len(territories), num_players)
        else:
            raise ValueError(f"Invalid bot type: {bot_types[p]}")

        player.key = turn_order
        players.append(player)

    assert(len(players) == num_players)
    initialize(territories, players, player_colors)
    return continents, territories, players

# returns the board graph
def create_graph(territories, display=False):
    G = nx.Graph()

    for territory in territories:
        G.add_node(territory)

    for territory in territories:
        for neighbor in territory.neighbors:
            G.add_edge(territory, neighbor)

    if(display):
        plt.figure(figsize=(15, 10))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=2000, font_size=10)
        # Add troop count labels
        for territory in territories:
            x, y = pos[territory.name]
            plt.text(x, y - 0.1, "Troops: " + str(territory.troop_count), horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')
            plt.show()

    return G

def find_shortest_path(graph, from_teritory, to_teritory, player_color, attack_turn=True, display=False):
    # if attack turn, we cannot attack our own color
    # if fortify turn, we can only fortify through our territory
    def attribute_check(node):
        if node == from_teritory: return True
        if attack_turn:
            return node.owner_color != player_color
        else:
            return node.owner_color == player_color

    # Creating a subgraph of only possible legal paths
    nodes_subgraph = [node for node in graph.nodes() if attribute_check(node)]

    subgraph = graph.subgraph(nodes_subgraph)

    path = None
    try:
        path = nx.shortest_path(subgraph, from_teritory, to_teritory, weight=None, method='dijkstra')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    if display:
        # Draw the graph with the generated path
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(graph)

        # Create labels for the nodes
        labels = {territory: territory.name for territory in graph.nodes()}

        # Define a color map, where different owner_colors are assigned different colors
        # Here we assume the 'owner_color' attribute is a string and use it directly for coloring
        color_map = {territory: territory.owner_color for territory in graph.nodes()}

        nx.draw_networkx_nodes(graph, pos, node_color=list(color_map.values()))
        nx.draw_networkx_labels(graph, pos, labels=labels)  # add the labels here
        nx.draw_networkx_edges(graph, pos, alpha=0.2)

        if path is not None:
            edges_in_path = [(path[i-1], path[i]) for i in range(1, len(path))]
            nx.draw_networkx_edges(graph, pos, edgelist=edges_in_path, edge_color='red', width=2)

        plt.show()

    return path

# Find all possible territories to fortify to given the source territory
def fortify_bfs(territory):
    visited = {territory}
    queue = [territory]
    while queue:
        curr = queue.pop(0)
        for n in curr.neighbors:
            if n.owner == curr.owner and n not in visited:
                visited.add(n)
                queue.append(n)
    return list(visited - {territory})

def display_graph(graph, territories, title="Figure", save=False, blocking_display=True):
    # TODO
    # better align graph to world map if the input graph isnt a world map

    node_colors = []
    for territory in graph.nodes():
        node_colors.append(territory.owner_color)

    plt.figure(title, figsize=(12, 6))

    # this could be done faster by defining the dict as a dict of objects in the first place
    # but there are bigger fish
    pos_with_objects = {}
    for territory in graph.nodes():
        pos_with_objects[territory] = pos[territory.name]

    labels = {territory: territory.name for territory in graph.nodes()}  # map the nodes to their names

    nx.draw(graph, pos_with_objects, node_color=node_colors, labels=labels, with_labels=True, node_size=2000, font_size=10)

    # Add troop count labels
    for territory in territories:
        x, y = pos[territory.name]
        plt.text(x, y - 0.1, "Troops: " + str(territory.troop_count), horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

    if(save):
        plt.savefig(title)

    plt.show(block=blocking_display)




if __name__ == "__main__":
    continents, territories = create_board()
    display_board(territories)
