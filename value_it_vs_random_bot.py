# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example to use value iteration to solve a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import six
import numpy as np
import random
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import value_iteration
from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import random_agent
import logging
import sys


# Calculate values with value iteration:
def solve_goofspiel(num_cards=3):
  """Solves goofspiel.

  Returns:
    Dictionary of values
  """
  game = pyspiel.load_game('goofspiel(imp_info=False,num_cards={})'.format(num_cards))

  print("Solving the game; depth_limit = {}".format(-1))
  values = value_iteration.value_iteration(game, -1, 0.01)

  for state, value in six.iteritems(values):
    print("")
    print(str(state))
    print("Value = {}".format(value))

  return values

########################################################################################################################

class ValueItAgent(rl_agent.AbstractAgent):
  """Tabular value agent.

  """

  def __init__(self,
               player_id,
               num_actions,
               values):
    """Initialize the value agent."""
    self._player_id = player_id
    self._num_actions = num_actions
    self._values = values
    self._prev_info_state = None
    self._last_loss_value = None

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    If the agent has not been to `info_state`, a valid random action is chosen.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float, prob of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
    greedy_actions = [
        a for a in legal_actions if self._q_values[info_state][a] == greedy_q
    ]
    probs[legal_actions] = epsilon / len(legal_actions)
    probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
    action = np.random.choice(range(self._num_actions), p=probs)
    return action, probs

  def _matrix_game(self, state):
    # This function sets up a matrix game, solves it and returns the policies

    p0_utils = []  # row player
    p1_utils = []  # col player
    row = 0
    key = str(state)
    states = {key: state}
    transitions = {}
    value_iteration._initialize_maps(states, self._values, transitions)
    for p0action in state.legal_actions(0):
      # new row
      p0_utils.append([])
      p1_utils.append([])
      for p1action in state.legal_actions(1):
        # loop from left-to-right of columns
        next_states = transitions[(key, p0action, p1action)]
        joint_q_value = sum(p * self._values[next_state] for next_state, p in next_states)
        p0_utils[row].append(joint_q_value)
        p1_utils[row].append(-joint_q_value)
      row += 1
    stage_game = pyspiel.create_matrix_game(p0_utils, p1_utils)
    solution = lp_solver.solve_zero_sum_matrix_game(stage_game)
    probs = solution[0]
    actions = state.legal_actions(0) # double check that order is consistent with probs
    return actions, probs

  def step(self, time_step, state):
    """Returns the action to be taken.

      Args:
        time_step: an instance of rl_environment.TimeStep.
        state: should be able to recover this from time_step but dont know how.. therefore we just add this argument

      Returns:
        A `rl_agent.StepOutput` containing the action probs and actions.
    """
    # state = time_step.observations["info_state"][self._player_id]
    legal_actions = time_step.observations["legal_actions"][self._player_id]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    # Act step: don't act at terminal states.
    if not time_step.last():
      actions, probs = self._matrix_game(state)
      probs = abs(np.array(probs).flatten()) # convert to np.array and make sure they are positive (small negative outputs)
      probs = probs / sum(probs) # make sure they are properly normalized
      action = np.random.choice(actions, p=probs)

    return rl_agent.StepOutput(action=action, probs=probs)

########################################################################################################################

# command line actions:
# Changed the format to print more user friendly on lines 8 and 12 the addition and subtraction by 1.
def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    cards_actions = [x+1 for x in legal_actions]
    action = -1
    while action not in legal_actions:
        print("Choose an card to play from your hand {}:".format(cards_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str) - 1
        except ValueError:
            continue
    return action


# Copied from pyspiel (not needed so far)
def evaluate_bots(state, bots, rng):
  """Plays bots against each other, returns terminal utility for each bot."""
  for bot in bots:
    bot.restart_at(state)
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      action = rng.choice(outcomes, p=probs)
      for bot in bots:
        bot.inform_action(state, pyspiel.PlayerId.CHANCE, action)
      state.apply_action(action)
    elif state.is_simultaneous_node():
      joint_actions = [
          bot.step(state)
          if state.legal_actions(player_id) else pyspiel.INVALID_ACTION
          for player_id, bot in enumerate(bots)
      ]
      state.apply_actions(joint_actions)
    else:
      current_player = state.current_player()
      action = bots[current_player].step(state)
      for i, bot in enumerate(bots):
        if i != current_player:
          bot.inform_action(state, current_player, action)
      state.apply_action(action)
  return state.returns()





def main(argv):
  del argv

  # calculate state values:
  num_cards = 5
  values = solve_goofspiel(num_cards)

  # setup environment:
  game = pyspiel.load_game('goofspiel(imp_info=False,num_cards={})'.format(num_cards))
  env = rl_environment.Environment(game)
  num_actions = env.action_spec()["num_actions"]

  # define agent:
  value_it_agent = ValueItAgent(0, num_actions, values)
  rand_agent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

  # play against human:
  print("=============================")
  num_episodes = 1000
  wins = 0
  draws = 0
  logging.info(
      "Playing goofspiel with {} cards over {} episodes. value_it_agent (p0) vs. random_agent (p1)".format(num_cards,
                                                                                                           num_episodes))
  for i in range(num_episodes):

    logging.info("episode {}".format(i))
    time_step = env.reset()

    while not time_step.last():

      # print current state:
      curr_state = env.get_state
      # print("Next turn. Current state is: ")
      # print(str(curr_state))
      logging.info(str(curr_state))

      # value it player:
      agent_out = value_it_agent.step(time_step, curr_state)
      logging.info("\n%s", agent_out.probs)
      p0_action = agent_out.action
      logging.info('Agent 0 played: {}'.format(p0_action + 1))

      # human player:
      agent_out = rand_agent.step(time_step)
      p1_action = agent_out.action
      logging.info('Agent 1 played: {}'.format(p1_action + 1))

      # print(time_step.observations['info_state'][0])
      # print(time_step.observations['info_state'][1])
      # print(len(time_step.observations['info_state'][0]))
      # print(env.observation_spec())

      # state = time_step.observations['info_state'][0]
      # state = np.asarray(state)
      #
      # P_ob = np.where(state[points_ob_b:points_op_b] == 1)[0][0]
      # P_op = np.where(state[points_op_b:seq_b] == 1)[0][0]
      # logging.info('Points: P%d = %d P%d = %d', player_id, P_ob, 0 if player_id == 1 else 1, P_op)
      #
      # which = num_cards - np.sum(state[np.size(state) - num_cards:])
      # curr = np.where(state[int(seq_b + which * num_cards): int(seq_b + (which + 1) * num_cards)] == 1)[0][0] + 1
      # logging.info('Point Card (Middle Card): %d', curr)
      #

      time_step = env.step([p0_action, p1_action])

    # logging.info("\n%s", pretty_board(time_step))

    logging.info("End of game!")
    if time_step.rewards[1] > 0:
      logging.info("You win")
    elif time_step.rewards[1] < 0:
      logging.info("You lose")
    else:
      logging.info("Draw")

    if time_step.rewards[0] > 0:
        wins += 1
    if time_step.rewards[0] == 0:
        draws += 1

    p0_win = wins / num_episodes

    logging.info("Summary: ==============")
    logging.info("Wins: {}, Draws: {}, Estimated pwin: {}".format(wins, draws, p0_win))



if __name__ == "__main__":
  app.run(main)
