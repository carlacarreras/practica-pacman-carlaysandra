# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        pass

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = []
        for action in legal:
            successor = state.generateSuccessor(0, action)
            
            # --- AQUÍ METEMOS TUS HEURÍSTICAS DE LA TAREA 1 ---
            score = successor.getScore()
            pacman_pos = successor.getPacmanPosition()
            food = successor.getFood().asList()
            ghost_states = successor.getGhostStates()
            
            if food:
                min_food_distance = min(util.manhattanDistance(pacman_pos, f) for f in food)
                score += 1.0 / (min_food_distance + 1)
            
            for g in ghost_states:
                g_pos = g.getPosition()
                g_dist = util.manhattanDistance(pacman_pos, g_pos)
                if g.scaredTimer > 0:
                    score += 50 / (g_dist + 1)
                else:
                    if g_dist <= 2: score -= 200

            # Heurística A: Cápsulas de poder
            capsulas = successor.getCapsules()
            if capsulas:
                min_cap_dist = min(util.manhattanDistance(pacman_pos, c) for c in capsulas)
                score += 10.0 / (min_cap_dist + 1)
                
            # Heurística B: Evitar callejones (más acciones legales)
            score += len(successor.getLegalActions(0)) * 3.0
            
            successors.append((action, score))
            
        successors.sort(key=lambda x: x[1], reverse=True)
        return successors[0][0]


def scoreEvaluation(state):
    return state.getScore()
class AlphaBetaNeuralAgent(Agent):
    def __init__(self, w_heuristic=1.0, w_neural=1.0, depth="2", **kwargs):
        super().__init__(**kwargs)
        self.depth = int(depth)
        self.w_heuristic = float(w_heuristic)
        self.w_neural = float(w_neural)
        self.neural_brain = None
        try:
            self.neural_brain = NeuralAgent()
        except:
            pass

    def evaluation_combined(self, state):
        trad_score = state.getScore()
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        if food:
            min_food_distance = min(util.manhattanDistance(pacman_pos, f) for f in food)
            trad_score += 1.0 / (min_food_distance + 1)
        
        for g in ghost_states:
            g_pos = g.getPosition()
            g_dist = util.manhattanDistance(pacman_pos, g_pos)
            if g.scaredTimer > 0:
                trad_score += 50 / (g_dist + 1)
            else:
                if g_dist <= 2: 
                    trad_score -= 200

        capsulas = state.getCapsules()
        if capsulas:
            min_cap_dist = min(util.manhattanDistance(pacman_pos, c) for c in capsulas)
            trad_score += 10.0 / (min_cap_dist + 1)
            
        trad_score += len(state.getLegalActions(0)) * 3.0

        neural_score = 0
        if self.neural_brain and self.neural_brain.model is not None:
            state_matrix = self.neural_brain.state_to_matrix(state)
            state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.neural_brain.device)
            with torch.no_grad():
                output = self.neural_brain.model(state_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            
            legal_actions = state.getLegalActions()
            for i, action in enumerate(self.neural_brain.idx_to_action.values()):
                if action in legal_actions:
                    neural_score += probabilities[i] * 100

        any_scared = any(g.scaredTimer > 0 for g in ghost_states)
        current_w_heuristic = self.w_heuristic * 2.0 if any_scared else self.w_heuristic
        current_w_neural = self.w_neural * 0.5 if any_scared else self.w_neural

        return current_w_heuristic * trad_score + current_w_neural * neural_score

    def getAction(self, gameState):
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluation_combined(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)
            
            if not actions:
                return self.evaluation_combined(state)

            if agentIndex == 0:
                v = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if v > beta: return v
                    alpha = max(alpha, v)
                return v
            else:
                v = float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v)
                return v

        legalActions = gameState.getLegalActions(0)
        if not legalActions or gameState.isWin() or gameState.isLose():
            return Directions.STOP
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)

        bestAction = legalActions[0]
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, self.depth, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

