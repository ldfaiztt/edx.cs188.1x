# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #------------------------------
    # ghost scared calculations
    #------------------------------
    v = False
    v_gs = False
    bAreGhostsScared = False
    if newScaredTimes[0]>5:
      bAreGhostsScared = True
      if v_gs:
        print newScaredTimes
        print currentGameState.getCapsules()
      #raw_input()
    #------------------------------
    # food calculations
    #------------------------------
    food_dist = []
    food_avg_dist = 0
    food_min_dist = 0
    food_list = successorGameState.getFood().asList()
    if v: print food_list
    for f in food_list:
      food_dist.append(manhattanDistance(newPos, f))
    if v : print 'distances: ' + str(food_dist)
    if len(food_dist)>0:
      food_avg_dist = sum(food_dist)/(len(food_dist)*1.0)
      food_min_dist = min(food_dist)
    if v :print 'average food distance: '+ str(food_avg_dist)
    #food_inv_min = (1/food_min_dist)**4 if food_min_dist >0 else 0
    #food_inv_min = (1/food_min_dist) if food_min_dist >0 else 0
    #food_inv_min = (food_min_dist) if food_min_dist >0 else 0
    food_inv_min = (1.0/food_min_dist)**2 if food_min_dist >0 else 0
    if bAreGhostsScared:
      food_inv_min = food_inv_min * 10
    #------------------------------
    # ghost calculations
    #------------------------------
    ghost_dist = []
    ghost_avg_dist = 0
    ghost_min_dist =0    
    for g in newGhostStates:
      ghost_dist.append(manhattanDistance(newPos,g.getPosition()))
    
    if len(ghost_dist)>0:
      ghost_avg_dist = sum(ghost_dist)/(len(ghost_dist)*1.0)
      #ghost_min_dist = -2/min(ghost_dist) if min(ghost_dist)>0 else 0
      ghost_min_dist = -2.0/min(ghost_dist) if min(ghost_dist)>0 else 0
      #ghost_min_dist = -1/min(ghost_dist) if min(ghost_dist)>0 else 0
      #ghost_min_dist = -min(ghost_dist) if min(ghost_dist)>0 else 0

    # if ghosts are too far away it is not so dangerous
    walls = successorGameState.getWalls()
    hypothenuse = ( (walls.width)**2 + (walls.height)**2 )**0.5
    if min(ghost_dist) > hypothenuse/5.0 and not bAreGhostsScared:
      #print min(ghost_dist)
      #print hypothenuse/2.
      #raw_input()
      ghost_min_dist = ghost_min_dist * -1
    elif min(ghost_dist) < hypothenuse/4.0 and not bAreGhostsScared:
      ghost_min_dist = ghost_min_dist * 4
      
    # Ghost are scared, pacman should be atracted by them
    if bAreGhostsScared:
      ghost_min_dist = ghost_min_dist * -10
    if v: print 'average gost distance: '+str(ghost_avg_dist)
    
    s = successorGameState.getScore()

    return s + ghost_min_dist + food_inv_min

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    #return (action, value)
    v = False
    if v:
      pacman_id = 0
      pacman_actions = gameState.getLegalActions(pacman_id) 
      print 'pacman legal actions: '+str(pacman_actions)
      print 'Directions.STOP: ' + str( Directions.STOP )
      pacman_scores = []
      for action in pacman_actions:
        pacman_succ = gameState.generateSuccessor(pacman_id,action)
        print '\taction: '+ str(action)
        #print '\tsuccessor: ' + str(pacman_succ)
        print '\tscore: ' + str(pacman_succ.getScore())
        pacman_scores.append(pacman_succ.getScore())
      print pacman_scores
      print 'max index: ' + str( pacman_scores.index(max(pacman_scores)) )
      agents = gameState.getNumAgents()
      print 'Total number of agents: ' + str(agents)
      
      print '------------------------------'
      for agent_id in range(1,agents):
        print 'agent id: ' + str(agent_id)
        agent_actions = gameState.getLegalActions(agent_id) 
        agent_scores = []
        print 'agent actions: ' + str(agent_actions)
        for action in agent_actions:
          agent_succ = gameState.generateSuccessor(agent_id,action)
          print '\taction: ' + str(action)
          #print '\tsuccessor: ' + str(agent_succ)
          print '\tscore: ' + str(agent_succ.getScore())
          agent_scores.append(agent_succ.getScore())
        print agent_scores
        print 'min agent score: ' + str( agent_scores.index(min(agent_scores)) )
        print '----------'
    else:
      #number_agents = gameState.getNumAgents()
      #(value,action) = self.maxValue(gameState,self.depth)
      (action, value) = self.minmax_decision(gameState,self.depth)
      if v:
        print 'value: '+str(value)
        print 'action: '+str( action)
        print '****************************************'
      #return pacman_actions[action_index]
      
      return action

    util.raiseNotDefined()
  def minmax_decision(self,state,depth):
    actions = []
    pacman_id = 0

    for action in state.getLegalActions(pacman_id):
      if action != Directions.STOP:
        actions.append(action)
    results = []

    for action in actions:
      minval = self.minValue(state.generateSuccessor(pacman_id,action),depth)
      results.append((action,minval))

    action, value = max(results, key= lambda x: x[1])
    #print (action, value)
    return (action, value)
                    
  def maxValue(self, state, depth, v = False):
    #v = False
    if v : print 'maxValue(): '+str(state) + ' depth: '+str(depth)
    if v: raw_input()
    if state.isLose() or state.isWin() or depth < 0:
      #v = True
      if v: print 'evalfn() = '+ str( self.evaluationFunction(state) )
      return self.evaluationFunction(state)
    
    depth = depth - 1
    pacman_id = 0
    value = float('-inf')
    agent_actions = state.getLegalActions(pacman_id)
    for action in agent_actions:
      if action != Directions.STOP:
        succ = state.generateSuccessor(pacman_id,action)
        value = max(value, self.minValue(succ, depth) )
    return value

  def minValue(self,state,  depth):
    v = False
    if v: print 'minValue(): '+str(state) + ' agent_id: '+ str(agent_id) + ' depth: '+str(depth)
    if v: raw_input()
    if state.isLose() or state.isWin() or depth < 0:
      if v: print 'evalfn() '+ str( self.evaluationFunction(state) ) 
      return self.evaluationFunction(state)
    
    depth = depth - 1
    value = float('inf')
    for agent_id in range(1,state.getNumAgents()):
      for action in state.getLegalActions(agent_id):
        succ = state.generateSuccessor(agent_id,action)
        value = min(value, self.maxValue(succ,depth))
    return value
                  
  

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    pacman_id = 0
    minus_inf = float('-inf')
    plus_inf = float('inf')
    action = self.maxValue(gameState,minus_inf, plus_inf,0)
    return action

  def maxValue(self, state, alpha, beta,depth):
    
    if state.isLose() or state.isWin() or depth == self.depth:
      return self.evaluationFunction(state)

    pacman_id = 0
    pacman_actions = state.getLegalActions(pacman_id)
    actions = [a for a in pacman_actions if a != Directions.STOP]

    values = {}
        
    for action in actions:
      succ = state.generateSuccessor(pacman_id,action)
      #start at agent_id = 1 then the min will take care of it
      minvalue = self.minValue(succ, 1, alpha, beta,depth)
      if minvalue > beta:
        return minvalue

      if values.has_key(minvalue):
        values[minvalue].append(action)
      else:
        values[minvalue] = [action]

      alpha = max(alpha, minvalue)
  
    value = max(values)
    if depth > 0:
      return value
    else:
      return random.choice(values[value])
    
  def minValue(self, state, agent_id, alpha, beta, depth):

    if state.isLose() or state.isWin():
        return self.evaluationFunction(state)
    
    agent_actions = state.getLegalActions(agent_id)
    agent_succs = []
    for action in agent_actions:
      agent_succs.append(state.generateSuccessor(agent_id,action))
    
    if agent_id == (state.getNumAgents() - 1): #last
      
        values = []
        for succ in agent_succs:
            value = self.maxValue(succ, alpha, beta, depth + 1)
            
            if value < alpha:
              return value
          
            beta = min(beta, value)
            values.append(value)
            
        return min(values)
    else:
        values = []
        for succ in agent_succs:
            value = self.minValue(succ, agent_id + 1, alpha, beta, depth)
            if value < alpha:
                return value
            
            values.append(value)

        return min(values)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    import random
    next_id = (0+1)%gameState.getNumAgents()
    depth = 0
    actions = gameState.getLegalActions(0)
    chosen_actions = []
    chosen_actions.append(actions.pop())
    succ = gameState.generateSuccessor(0,chosen_actions[0])
    current_value = self.expectiminimax(succ,next_id,depth+1)
        
    for action in actions:
      succ = gameState.generateSuccessor(0,action)
      value = self.expectiminimax(succ,next_id,depth + 1)
      
      if value > current_value:
        chosen_actions = [action]
        current_value = value
      elif value == current_value:
        chosen_actions.append(action)
    if Directions.STOP == chosen_actions[0]:#move it to the end
      chosen_actions.pop(0)
      chosen_actions.append(Directions.STOP)
    return chosen_actions[0]
    action = chosen_actions[random.randint(0,len(chosen_actions)-1)]
    return action

  def expectiminimax(self, state, agent_id, depth):
    if state.isWin() or state.isLose() or depth >=self.depth:#Terminal Test
      return self.evaluationFunction(state)
    elif agent_id ==0: #MAX
      depth = depth + 1
      actions = state.getLegalActions(agent_id)
      values = []
      next_id = (agent_id + 1)%state.getNumAgents()
      for action in actions:
        if action != Directions.STOP:
          succ = state.generateSuccessor(agent_id,action)
          value = self.expectiminimax(succ,next_id,depth)
          values.append((succ,value))
      return max(values, key = lambda x: x[1])[1]
    else: # MIN (we assume player min always plays by chance)
      actions = state.getLegalActions(agent_id)
      values = []
      next_id = (agent_id + 1)%state.getNumAgents()
      for action in actions:
        succ = state.generateSuccessor(agent_id,action)
        value = self.expectiminimax(succ,next_id,depth)
        values.append(value)
      ratio = 1./len(actions)
      res = 0.0
      for v in values:
        res = res + v*ratio
      return res
    
def betterEvaluationFunction(state):
  """
  Design a better evaluation function here.
  
  The evaluation function takes in the current and proposed successor
  GameStates (pacman.py) and returns a number, where higher numbers are better.
  
  The code below extracts some useful information from the state, like the
  remaining food (newFood) and Pacman position after moving (newPos).
  newScaredTimes holds the number of moves that each ghost will remain
  scared because of Pacman having eaten a power pellet.
  
  Print out these variables to see what you're getting, then combine them
  to create a masterful evaluation function.
  """
  # Useful information you can extract from a GameState (pacman.py)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
  newPos = state.getPacmanPosition()
  newFood = state.getFood()
  newGhostStates = state.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #------------------------------
    # ghost scared calculations
    #------------------------------
  v = False
  v_gs = False
  bAreGhostsScared = False
  if newScaredTimes[0]>5:
    bAreGhostsScared = True
    if v_gs:
      print newScaredTimes
      print currentGameState.getCapsules()
      #raw_input()
    #------------------------------
    # food calculations
    #------------------------------
  food_dist = []
  food_avg_dist = 0
  food_min_dist = 0
  food_list = state.getFood().asList()
  if v: print food_list
  for f in food_list:
    food_dist.append(manhattanDistance(newPos, f))
  if v : print 'distances: ' + str(food_dist)
  if len(food_dist)>0:
    food_avg_dist = sum(food_dist)/(len(food_dist)*1.0)
    food_min_dist = min(food_dist)
  if v :print 'average food distance: '+ str(food_avg_dist)
  food_inv_min = (2.0/food_min_dist)**2 if food_min_dist >0 else 0
  if bAreGhostsScared:
    food_inv_min = food_inv_min * 10
    #------------------------------
    # ghost calculations
    #------------------------------
  ghost_dist = []
  ghost_avg_dist = 0
  ghost_min_dist =0    
  for g in newGhostStates:
    ghost_dist.append(manhattanDistance(newPos,g.getPosition()))
    
  if len(ghost_dist)>0:
    ghost_avg_dist = sum(ghost_dist)/(len(ghost_dist)*1.0)
    ghost_min_dist = -2.0/min(ghost_dist) if min(ghost_dist)>0 else 0
    
    # if ghosts are too far away it is not so dangerous
  walls = state.getWalls()
  hypothenuse = ( (walls.width)**2 + (walls.height)**2 )**0.5
  if min(ghost_dist) > hypothenuse/5.0 and not bAreGhostsScared:
    ghost_min_dist = ghost_min_dist * 0* -1
  # Ghost are scared, pacman should be atracted by them
  if bAreGhostsScared:
    ghost_min_dist = ghost_min_dist * -10
  if v: print 'average gost distance: '+str(ghost_avg_dist)
    
  s = state.getScore()
  
  return s + ghost_min_dist + food_inv_min

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

