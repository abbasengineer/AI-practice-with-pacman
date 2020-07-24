# from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.agents.capture.capture import CaptureAgent
# from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.directions import Directions
# from pacai.core.actions import Actions
# from pacai.util import reflection
from pacai.util import counter
from pacai.util import util
# from pacai.util import probability
import logging
import time
import random
import math


def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = ModifiedExpectimaxAgent(firstIndex)
    secondAgent = DefensiveReflexAgent(secondIndex)

    return [firstAgent, secondAgent]

class ModifiedExpectimaxAgent(CaptureAgent):
    # So far the algorithm only works for TREE_DEPTH <= 2. Depth > 2 yields infinite recursion.
    TREE_DEPTH = 2

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Wrapper function for findAction, it's not really necessary
    def chooseAction(self, gameState):
        return self.findAction(gameState)

    def findAction(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return self.evaluate(gameState)

        bestAction = Directions.STOP
        score = -math.inf
        for action in gameState.getLegalActions(self.index):
            # filter out the STOP action to make branching factor smaller
            if action != 'Stop':
                nextState = self.getSuccessor(gameState, action)
                prevscore = score
                score = max(score, self.expectedValue(nextState, self.index + 1, self.TREE_DEPTH))
                if score > prevscore:
                    bestAction = action
        return bestAction

    def expectedValue(self, gameState, agentIndex, depth):
        # We don't have to worry about the other teammate, so skip over him
        if agentIndex == self.index + 2:
            agentIndex += 1

        # wrap around case
        if agentIndex == 4:
            agentIndex = 0
        
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluate(gameState)

        totalValue = 0

        # Agent case
        if agentIndex == self.index:
            totalValue += self.maxValue(gameState, depth - 1)
            return totalValue

        # Opponent case
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                # filter out the STOP action to make branching factor smaller
                if action != 'Stop':
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    totalValue += self.expectedValue(nextState, agentIndex + 1, depth)

            return totalValue / len(legalActions)

    def maxValue(self, gameState, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluate(gameState)
        legalActions = gameState.getLegalActions(self.index)
        score = -math.inf
        for action in legalActions:
            nextState = self.getSuccessor(gameState, action)
            score = max(score, self.expectedValue(nextState, self.index + 1, depth))

        return score

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        # Only half a grid position was covered.
        if (pos != util.nearestPoint(pos)):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, gameState):
        features = counter.Counter()
        features['successorScore'] = self.getScore(gameState)

        # FIXME: extract more features here.
        myPos = gameState.getAgentState(self.index).getPosition()

        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()
        if (len(foodList) > 0):
            minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['DistanceToFoodTarget'] = minDist

        # Compute distance to the nearest Capsule #
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            capDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
            features['capsuleDistance'] = capDistance
        else:
            features['capsuleDistance'] = 0

        return features

    def getWeights(self, gameState):
        # FIXME: add more feature weights here
        return {'successorScore': 100,
                'DistanceToFoodTarget': -10,
                'capDistance': -20
                }

    def evaluate(self, gameState):
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)

        return features * weights

# This agent is pretty dumb at the moment, don't use it unless it can be improved/rewritten
class OffensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myPos = successor.getAgentState(self.index).getPosition()

        # Compute distance to the nearest Capsule #
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            capDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
            features['capsuleDistance'] = capDistance
        else:
            features['capsuleDistance'] = 0

        # Compute distance to nearest ghost (usually a defender goalie) #
        # enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        # chasers = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        # uncomment this for invader information
        # invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        if (len(foodList) > 0):
            minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['DistanceToFoodTarget'] = minDist

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'DistanceToFoodTarget': -10,
            'capsuleDistance': -20,
            'goalieDistance': 0,
        }

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        # Only half a grid position was covered.
        if (pos != util.nearestPoint(pos)):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

class DefensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # Compute distance to nearest ghost (before it is invading)
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['ghostDist'] = min(dists)

        # Computes distance to invaders we can see.
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        """ # Abbas put this in here for some reason, it doesn't seem to do anything yet
             closestDistance = dists[0]
             position = [a.getPosition() for a in invaders]
             closestPosition = position[0]

             for i in range(len(dists)):
                 if dists[i] < closestDistance:
                     closestDistance = dists[i]
                     closestPosition = position[i]
        """
        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000,
                'onDefense': 100,
                'invaderDistance': -20,
                'ghostDist': -10,
                'stop': -100,
                'reverse': -2}
