import numpy as np


def split_action(action):
    '''
        Helper, which returns
        (grid_to_resource, resource_to_grid)
        for an action
    '''
    return action[0], action[1]


def print_action(action):
    grid_to_resource, resource_to_grid = split_action(action)

    print(f'GridToResource: {grid_to_resource} - ResourceToGrid: {resource_to_grid}')


def print_state(self):
    print(f"Battery: {self.resource} - Price: {self.price}")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
