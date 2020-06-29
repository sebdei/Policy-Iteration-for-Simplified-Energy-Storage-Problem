import numpy as np
import utils

np.random.seed(42)

DISCOUNT_RATIO = 0.9
EPSILON = 1e-2

RESOURCE_CAPACITY = 4
RESOURCE_MAX_TRANSFER_RATE_IN = 2.5
RESOURCE_MAX_TRANSFER_RATE_OUT = 2.5
RESOURCE_TRANSFER_LOSS = 0.8


class Environment():

    def get_price_transition_matrix(self):
        return {
            1.0: [0.40, 0.30, 0.20, 0.10, 0.00],
            2.0: [0.20, 0.40, 0.25, 0.15, 0.10],
            3.0: [0.10, 0.20, 0.40, 0.20, 0.10],
            4.0: [0.10, 0.15, 0.25, 0.40, 0.20],
            5.0: [0.00, 0.10, 0.20, 0.30, 0.40]
        }

    def get_transition_probabilities(self, current_price):
        transition_matrix = self.get_price_transition_matrix()
        probs = transition_matrix[current_price]

        return {price: prob for price, prob in zip(transition_matrix.keys(), probs)}

    def calculate_contribution_for_valid_state_action(self, state, action):
        grid_to_resource, resource_to_grid = utils.split_action(action)

        return (RESOURCE_TRANSFER_LOSS * state.price * resource_to_grid) - (state.price * grid_to_resource)

    def determine_contribution_transition(self, state, action, successor_state):
        return self.calculate_contribution_for_valid_state_action(state, action) if successor_state.is_valid() else -10


class State():

    def __init__(self, price, resource):
        self.price = price
        self.resource = round(resource, 4)

    def create_successor_state(self, price, price_prob, resource):
        prob = price_prob
        return prob, State(price=price, resource=resource)

    def determine_next_resource(self, action):
        grid_to_resource, resource_to_grid = action

        return self.resource + RESOURCE_TRANSFER_LOSS * grid_to_resource - RESOURCE_TRANSFER_LOSS * resource_to_grid

    def is_valid(self):
        return self.resource >= 0 and self.resource <= RESOURCE_CAPACITY

    def transition(self, action, env):
        resource = self.determine_next_resource(action)
        transition_probs = env.get_transition_probabilities(self.price)

        probs, successor_states = zip(*[self.create_successor_state(next_price, next_price_prob, resource)
                                        for next_price, next_price_prob in transition_probs.items()])
        contributions = [env.determine_contribution_transition(self, action, successor_state) for successor_state in successor_states]

        return probs, successor_states, contributions


class Agent():

    def __init__(self):
        self.env = None

    def are_constraints_satisfied(self, state, action):
        grid_to_resource, resource_to_grid = action

        energy_purchase_valid = RESOURCE_TRANSFER_LOSS * grid_to_resource <= RESOURCE_CAPACITY - state.resource
        energy_sell_valid = resource_to_grid <= state.resource

        return energy_purchase_valid and energy_sell_valid

    def calculate_action_value(self, state, action, values, possible_states):
        if not self.are_constraints_satisfied(state, action):
            result = -100
        else:
            result = 0

            for prob, successor_state, contribution in zip(*state.transition(action, self.env)):
                value_of_successor_state = self.find_value_for_state(successor_state, values, possible_states)
                result += prob * (contribution + DISCOUNT_RATIO * value_of_successor_state)

        return result

    def find_value_for_state(self, state, values, possible_states):
        effective_resource = round(utils.find_nearest(self.get_possible_resources(), state.resource), 4)

        if not state.is_valid():
            result = 0
        else:
            result = None
            for value, iter_state in zip(values, possible_states):
                if (iter_state.price == state.price and iter_state.resource == effective_resource):
                    result = value
                    break

            if result is None:
                utils.print_state(state)
                raise Exception("State was not found!")

        return result

    def get_discretized_actions(self):
        grid_to_resource = np.linspace(0, RESOURCE_MAX_TRANSFER_RATE_IN, num=21)
        resource_to_grid = np.linspace(0, RESOURCE_MAX_TRANSFER_RATE_OUT, num=21)

        return np.array(
            np.meshgrid(grid_to_resource, resource_to_grid)
        ).T.reshape(-1, 2)  # 2: number of decisions

    def get_possible_resources(self):
        return np.linspace(0, RESOURCE_CAPACITY, num=81)

    def get_discretized_states(self):
        price = np.linspace(1, 5, 5)
        resource = self.get_possible_resources()

        state_grid = np.array(
            np.meshgrid(price, resource)
        ).T.reshape(-1, 2)  # 2: number of state variables

        return np.array([State(price=state[0], resource=state[1]) for state in state_grid])

    def pick_random_action_for_state(self, state, possible_actions):
        while True:
            random_choice = np.random.choice(len(possible_actions))
            action_candidate = possible_actions[random_choice, :]

            _, successor_states, _ = state.transition(action_candidate, self.env)

            contraints_satisfied = [self.are_constraints_satisfied(successor_state, action_candidate)
                                    for successor_state in successor_states]

            if False in contraints_satisfied:
                continue
            else:
                result = action_candidate
                break

        return result

    def policy_evaluation(self, policy, possible_states):
        values = np.zeros(len(possible_states))

        while True:
            prev_values = np.copy(values)

            for i, (action, state) in enumerate(zip(policy, possible_states)):
                values[i] = self.calculate_action_value(state, action, values, possible_states)

            if (np.sum(np.fabs(prev_values - values)) < EPSILON):
                break

        return values

    def policy_improvement(self, values, possible_actions, possible_states):
        new_policy = [None] * (len(possible_states))

        for s, state in enumerate(possible_states):
            action_values = [self.calculate_action_value(state, action, values, possible_states) for action in possible_actions]

            best_action = possible_actions[np.argmax(action_values)]
            new_policy[s] = best_action

        return new_policy

    def is_identical(self, policy, new_policy):
        return next((False for action, old_action in zip(policy, new_policy) if not np.array_equal(action, old_action)), True)

    def policy_iteration(self, possible_actions, possible_states):
        policy_stable = False
        policy = [self.pick_random_action_for_state(state, possible_actions) for state in possible_states]

        while not policy_stable:
            values = self.policy_evaluation(policy, possible_states)
            new_policy = self.policy_improvement(values, possible_actions, possible_states)

            policy_stable = self.is_identical(policy, new_policy)
            policy = new_policy

            break

        optimal_policy = policy
        optimal_values = self.policy_evaluation(policy, possible_states)

        return optimal_policy, optimal_values

    def run_policy_iteration(self, env):
        self.env = env

        possible_actions = self.get_discretized_actions()
        possible_states = self.get_discretized_states()

        print(f"Possible actions: {len(possible_actions)}")
        print(f"Possible states: {len(possible_states)}")

        return self.policy_iteration(possible_actions=possible_actions, possible_states=possible_states)


env = Environment()
policy, optimal_values = Agent().run_policy_iteration(env)
