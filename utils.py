def graph(y_axis,  x_label, y_label, title, file_name, is_scatter=False, x_axis = None, show=False, save=True):
    if is_scatter:
        assert(x_axis != None)
        plt.scatter(x_axis, y_axis)
    else:
        plt.plot(y_axis)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()
    if save:
        plt.savefig(file_name)
    if show:
        plt.show(block=False)
        plt.pause(180)
        plt.close()

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
