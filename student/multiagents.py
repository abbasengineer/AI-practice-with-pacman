# some ideas have been inspired from debuggers on Stack Overflow
# these links helped me understand the general implementation for minimax and expectimax
# https://stackoverflow.com/questions/36022941/why-is-my-minimax-not-expanding-and-making-moves-correctly
# https://stackoverflow.com/questions/33848759/expectimax-algorithm-for-2048-not-performing-expectation-as-intended

import random
import math

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        newPosition = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        # found out through trial and error, and hardcoding the constants
        # this gives us the average ghost distance
        ghostDistance = sum([manhattan(newPosition, ghostState.getPosition())
            for ghostState in newGhostStates]) / (currentGameState.getNumAgents() - 1)
        oldFood = successorGameState.getFood()
        foodDistances = [manhattan(newPosition, food) for food in oldFood.asList()] + [10000]
        return successorGameState.getScore() + 10 / (min(foodDistances) + 0.0001) - \
            10 / (ghostDistance + 0.0001)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        # FIXME Actions seems non-optimal, b/c the scores are too low
        return self.value(gameState, self.getTreeDepth(), 0)[0]

    # MiniMax algorithm consists of 3 functions: value(), max-value(), min-value()
    # returns a "pair of action and evaluation"
    def value(self, gameState, depth, agentIndex = 0):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (Directions.STOP, self.getEvaluationFunction()(gameState))
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex)
        else:
            return self.min_value(gameState, depth, agentIndex)

    def min_value(self, gameState, depth, agentIndex):
        scores = []
        for action in gameState.getLegalActions(agentIndex):
            # wrap around case
            if agentIndex == gameState.getNumAgents() - 1:
                # can probably use generatePacmanSuccessor as well
                scores.append(self.value(gameState.generateSuccessor(agentIndex, action),
                        depth - 1, 0)[1])
            else:
                scores.append(self.value(gameState.generateSuccessor(agentIndex, action),
                        depth, agentIndex + 1)[1])

        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return gameState.getLegalActions(agentIndex)[chosenIndex], bestScore

    # Pacman case
    def max_value(self, gameState, depth, agentIndex):
        scores = []
        for action in gameState.getLegalActions(agentIndex):
            scores.append(self.value(gameState.generateSuccessor(agentIndex, action),
                    depth, agentIndex + 1)[1])

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return gameState.getLegalActions(agentIndex)[chosenIndex], bestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        return self.value(gameState, self.getTreeDepth(), 0, -math.inf, math.inf)[0]

    # MiniMax algorithm consists of 3 functions: value(), max-value(), min-value()
    # returns a "pair of action and evaluation"
    def value(self, gameState, depth, agentIndex, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (Directions.STOP, self.getEvaluationFunction()(gameState))
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.min_value(gameState, depth, agentIndex, alpha, beta)

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        scores = []
        for action in gameState.getLegalActions(agentIndex):
            # wrap around case
            if agentIndex == gameState.getNumAgents() - 1:
                # can probably use generatePacmanSuccessor as well
                scores.append(self.value(gameState.generateSuccessor(agentIndex, action),
                        depth - 1, 0, alpha, beta)[1])
            else:
                scores.append(self.value(gameState.generateSuccessor(agentIndex, action),
                                depth, agentIndex + 1, alpha, beta)[1])
            if min(beta, min(scores)) < alpha:
                break

        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return gameState.getLegalActions(agentIndex)[chosenIndex], bestScore

    # Pacman case
    def max_value(self, gameState, depth, agentIndex, alpha, beta):
        scores = []
        for action in gameState.getLegalActions(agentIndex):
            scores.append(self.value(gameState.generateSuccessor(agentIndex, action), depth,
                            agentIndex + 1, alpha, beta)[1])
            if max(alpha, max(scores)) > beta:
                break

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return gameState.getLegalActions(agentIndex)[chosenIndex], bestScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        bestAction = Directions.STOP
        score = -math.inf
        # FIXME Actions seems non-optimal
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            prevscore = score
            score = max(score, self.expectedValue(nextState, 1, self.getTreeDepth()))
            if score > prevscore:
                bestAction = action
        return bestAction

    def expectedValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.getEvaluationFunction()(gameState)
        numGhosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions(agentIndex)
        totalValue = 0
        for action in legalActions:
            nextState = gameState.generateSuccessor(agentIndex, action)
            # tricky part: # FIXME
            if (agentIndex == numGhosts):
                totalValue += self.maxValue(nextState, depth - 1)
            else:
                totalValue += self.expectedValue(nextState, agentIndex + 1, depth)
        return totalValue / len(legalActions)

    def maxValue(self, gameState, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)
        legalActions = gameState.getLegalActions(0)
        score = -math.inf
        # FIXME can write online
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            score = max(score, self.expectedValue(nextState, 1, depth))
        return score

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    This is nearly the same as the reflex one, but with modified constants,
    and I added food and capsule distances
    """

    oldFood = currentGameState.getFood()
    foodDistances = [manhattan(currentGameState.getPacmanPosition(), food)
                for food in oldFood.asList()] + [10000]
    minDist = min(foodDistances)
    newGhostStates = currentGameState.getGhostStates()
    ghostDistance = sum([manhattan(currentGameState.getPacmanPosition(), ghostState.getPosition())
        for ghostState in newGhostStates]) / (currentGameState.getNumAgents() - 1)

    capsules = currentGameState.getCapsules()
    capsuleDistances = [manhattan(currentGameState.getPacmanPosition(), capsule)
                for capsule in capsules] + [10000]
    minCapDist = min(capsuleDistances)

    return 100 * currentGameState.getScore() + 10 / (minDist + 0.001) +\
        10 / (ghostDistance + 0.001) + 100 / (minCapDist + 0.001)

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
