#dumb function for debug only purposes
def get_key_for_name(territory_name, territory_list):
    for territory in territory_list:
        if territory.name == territory_name:
            return territory, territory.key
    return None

def get_name_for_key(key, territory_list):
        return territory_list[key].name

def print_territory_keys(territory_list):
        for territory in territory_list:
            print(territory.name, str(territory.key))
