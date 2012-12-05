# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    
    """
    "*** YOUR CODE HERE ***"
    solution = []
    node = [problem.getStartState(),'',1,[]]
    if problem.isGoalState(problem.getStartState()):
        return solution

    explored = set()
    explored.add(node[0])
    frontier = util.Stack()
    frontier.push(node)
    
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())

    found = False
    count = 0
    while not found:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        #explored.add(node[0])
        children = problem.getSuccessors(node[0])
        #children.reverse()
        
        for child in children:
            if child[0] not in explored:#(explored or frontier[:][0]):
                current_path = list(node[3])
                current_path.append(child[1])
                child = list(child)
                child.append(current_path)
                explored.add(child[0])
                frontier.push(child)
                #solution.append(child[1])
                if problem.isGoalState(child[0]):
                    found = True
                    solution = child[3]
                    break
    return solution
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    solution = []
    node = [problem.getStartState(),'',1,[]]
    if problem.isGoalState(problem.getStartState()):
        return solution

    explored = set()
    explored.add(node[0])
    frontier = util.Queue()
    frontier.push(node)
        
    found = False
    count = 0
    while not found:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        
        children = problem.getSuccessors(node[0])
        for child in children:
            if child[0] not in explored:
                current_path = list(node[3])
                current_path.append(child[1])
                
                child = list(child)
                child.append(current_path)
                
                frontier.push(child)
                explored.add(child[0])
                if problem.isGoalState(child[0]):
                    found = True
                    solution = child[3]
                    break
    return solution
    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    node = [problem.getStartState(),'',1,[]]
    frontier = util.PriorityQueue()
    frontier.push(node,0)
    explored = set()
    explored.add(node[0])
    found = False
    while not found:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        if problem.isGoalState(node[0]):
            found = True
            solution = node[3]
        explored.add(node[0])
        children = problem.getSuccessors(node[0])
        for child in children:
            if child[0] not in explored:#(explored or frontier[:][0]):
                current_path = list(node[3])
                current_path.append(child[1])
                child = list(child)
                child.append(current_path)
                #explored.add(child[0])
                frontier.push(child, problem.getCostOfActions(current_path))
            #elif len([True for item in frontier.heap if child[0] in item[1]])>0:
            #    print 'child: '+str(child)
            #    print 'frontier items:'+ frontier
    return solution

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** Your CODE HERE ***"
    #def priorityfunction(x):
    #    return x[1]
    #frontier = util.PriorityQueueWithFunction(priorityfunction)
    #return greedySearch(problem, frontier, heuristic)
    
    node = [problem.getStartState(),'',1,[]]
    frontier = util.PriorityQueue()
    frontier.push(node,0)
    explored = set()
    found = False
    #print node[0]
    while not found:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        
        if problem.isGoalState(node[0]):
            found = True
            solution = node[3]
            break
        children = problem.getSuccessors(node[0])
        #print node
        #print node[0][0]
        explored.add(node[0])
        for child in children:
            if child[0] not in explored:#(explored or frontier[:][0]):
                current_path = list(node[3])
                current_path.append(child[1])
                child = list(child)
                child.append(current_path)
                #explored.add(child[0])
                g = problem.getCostOfActions(current_path)
                h = heuristic(child[0],problem)
                #if h>g:
                #    print'********************'
                #    print 'child:'+str(child)
                #    print 'cost:'+str(g)
                #    print 'H: '+str(h)
                #else:
                #    print 'child:'+str(child)
                #    print 'cost:'+str(g)
                #    print 'h:'+str(h)
                frontier.push(child, problem.getCostOfActions(current_path)+heuristic(child[0],problem))
                
    return solution
    util.raiseNotDefined()

def greedySearch(problem, frontier, heuristic):
  import copy
  frontier.push((problem.getStartState(), 0))
  visitedstates = {str(problem.getStartState()) : (0, [])}
  endState = None
  endStateFound= False
  while not endStateFound and not frontier.isEmpty():
    currstate = frontier.pop()[0]
    currstatehistory = visitedstates[str(currstate)]
    currstatecost = currstatehistory[0]
    currstatepath = currstatehistory[1]
    newoptions = problem.getSuccessors(currstate)
    for option in newoptions:
      if str(option[0]) not in visitedstates:
        newpath = copy.copy(currstatepath)
        newpath.append(option[1])
        newstatehistory = (currstatecost+option[2]+heuristic(option[0], problem), newpath)
        visitedstates[str(option[0])] = newstatehistory
        frontier.push((option[0], currstatecost+option[2]))
        if problem.isGoalState(option[0]):
          endStateFound=True
          endState=option[0]
          break
  if endStateFound:
    return visitedstates[str(endState)][1] 
  else:
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
