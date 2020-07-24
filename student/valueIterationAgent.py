from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.
        # FIXME: can be more efficient, getQValue is called twice.
        for i in range(iters):
            newVals = counter.Counter()
            for state in mdp.getStates():
                bestAction = self.getAction(state)
                if bestAction is not None:
                    newVals[state] = self.getQValue(state, bestAction)
            self.values = newVals.copy()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getPolicy(self, state):
        # FIXME: not sure if this is even needed
        if self.mdp.isTerminal(state):
            return None

        memo = counter.Counter()
        for action in self.mdp.getPossibleActions(state):
            memo[action] = self.getQValue(state, action)
        return memo.argMax()

    def getQValue(self, state, action):
        expectedUtil = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            expectedUtil += transition[1] * (self.mdp.getReward(state, action, transition[0])
                                           + self.discountRate * self.getValue(transition[0]))
        return expectedUtil

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
