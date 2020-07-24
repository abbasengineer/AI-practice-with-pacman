# some Approximate Q-learning ideas have been inspired from this Stanford paper
# http://cs229.stanford.edu/proj2017/final-reports/5241109.pdf

from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util import probability
import random
# from pacai.core.featureExtractors import *

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.QValues = counter.Counter()

        # You can initialize Q-values here.

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        if (state, action) in self.QValues:
            return self.QValues[(state, action)]

        return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        QValues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if len(QValues) == 0:
            return 0.0
        return max(QValues)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        bestActions = [action for action in self.getLegalActions(state)
                       if self.getQValue(state, action) == self.getValue(state)]

        # FIXME: This check might not be necessary
        if len(bestActions) == 0:
            return None
        return random.choice(bestActions)

    def update(self, state, action, nextState, reward):
        # adds the error times learning rate
        self.QValues[(state, action)] += self.getAlpha() * ((reward + self.getDiscountRate()
                     * self.getValue(nextState)) - self.QValues[(state, action)])

    def getAction(self, state):
        # use the exploration probability to choose randomly
        # this helps it explore the 0 Q-Values
        if probability.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        return self.getPolicy(state)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    I used the formula from slide 77 in MDPs.pdf
    This helped with the update.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        self.weights = counter.Counter()

        # You might want to initialize weights here.

    # calcualtes the expected utility of being at a certain state and about to take an action
    def getQValue(self, state, action):
        features = self.featExtractor().getFeatures(state, action)
        expectedUtil = 0
        for key in features.keys():
            expectedUtil += self.weights[key] * features[key]
        return expectedUtil

    def update(self, state, action, nextState, reward):
        # adds the error times learning rate
        error = (reward + self.getDiscountRate() * self.getValue(nextState))\
            - (self.getQValue(state, action))
        features = self.featExtractor().getFeatures(state, action)
        for key in features.keys():
            self.weights[key] += self.getAlpha() * error * features[key]

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass
