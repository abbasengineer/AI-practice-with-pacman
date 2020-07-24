"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueueWithFunction

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # initialize stuff for DFS
    explored = []
    goalNotFound = True

    fringe = Stack()
    fringe.push((problem.startingState(), []))
    while(not fringe.isEmpty() and goalNotFound):
        (node, path) = fringe.pop()
        explored.append(node)
        for (position, direction, cost) in problem.successorStates(node):
            if position not in explored and goalNotFound:
                if problem.isGoal(position):
                    goalNotFound = False
                    return path + [direction]
                else:
                    fringe.push((position, path + [direction]))
                explored.append(position)

    print("No path found: ")
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    explored = []
    goalNotFound = True

    fringe = Queue()
    fringe.push((problem.startingState(), []))
    while(not fringe.isEmpty() and goalNotFound):
        (node, path) = fringe.pop()
        explored.append(node)
        for (position, direction, cost) in problem.successorStates(node):
            if position not in explored and goalNotFound:
                if problem.isGoal(position):
                    goalNotFound = False
                    return path + [direction]
                else:
                    fringe.push((position, path + [direction]))
                explored.append(position)

    print("No path found: ")
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    explored = []
    goalNotFound = True

    fringe = PriorityQueueWithFunction(lambda x: x[2])
    fringe.push((problem.startingState(), [], 0))
    while(not fringe.isEmpty() and goalNotFound):
        (node, path, cost) = fringe.pop()
        explored.append(node)
        for (position, direction, newCost) in problem.successorStates(node):
            if position not in explored and goalNotFound:
                if problem.isGoal(position):
                    goalNotFound = False
                    return path + [direction]
                else:
                    fringe.push((position, path + [direction], cost + newCost))
                explored.append(position)

    print("No path found: ")
    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    explored = []
    goalNotFound = True

    # fringe = PriorityQueueWithFunction(lambda x: x[2] + manhattan(x[0], problem))
    fringe = PriorityQueueWithFunction(lambda x: x[2] + heuristic(x[0], problem))
    fringe.push((problem.startingState(), [], 0))
    while(not fringe.isEmpty() and goalNotFound):
        (node, path, cost) = fringe.pop()
        explored.append(node)
        for (position, direction, newCost) in problem.successorStates(node):
            if position not in explored and goalNotFound:
                if problem.isGoal(position):
                    goalNotFound = False
                    return path + [direction]
                else:
                    fringe.push((position, path + [direction], cost + newCost))
                explored.append(position)

    print("No path found: ")
    return []
