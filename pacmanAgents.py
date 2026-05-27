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

