# multiAgents.py
# --------------
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

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
random.seed(15)
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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
        def minimax(agentIndex, depth, state):
            # Caso base: victoria, derrota o alcanzar profundidad máxima
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0: # Turno de Pacman (MAX)
                return max(minimax(1, depth, state.generateSuccessor(0, a)) for a in state.getLegalActions(0))
            else: # Turno de los fantasmas (MIN)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1 # Incrementa profundidad al volver a Pacman
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, a)) for a in state.getLegalActions(agentIndex))
        
        # Toma la mejor acción basada en la raíz del árbol
        return max(gameState.getLegalActions(0), key=lambda a: minimax(1, 0, gameState.generateSuccessor(0, a)))

class AlphaBetaNeuralAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, w_heuristic=2.0, w_neural=1.0, depth="2", **kwargs):
        super().__init__(**kwargs)
        self.depth = int(depth)
        self.w_heuristic = float(w_heuristic)
        self.w_neural = float(w_neural)
        self.neural_brain = None
        
        # CORRECCIÓN DE IMPORTACIÓN: Cargamos la clase localmente desde este mismo archivo
        try:
            self.neural_brain = NeuralAgent()
            print("¡ÉXITO! Cerebro neuronal conectado correctamente a AlphaBetaNeuralAgent.")
        except Exception as e:
            print(f"Aviso al cargar cerebro neuronal en AlphaBetaNeuralAgent: {e}")

    def evaluation_combined(self, state):
        # 1) Puntuación tradicional + Heurísticas personalizadas de la Tarea 1
        trad_score = 0.0
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Heurística base de comida
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, f) for f in food)
            trad_score += 1.0 / (min_food_distance + 1)
        
        # Heurística base de proximidad a fantasmas
        for g in ghost_states:
            g_pos = g.getPosition()
            g_dist = manhattanDistance(pacman_pos, g_pos)
            if g.scaredTimer > 0:
                trad_score += 50.0 / (g_dist + 1) # Persecución activa
            else:
                if g_dist <= 2: 
                    trad_score -= 200.0 # Penalización crítica por peligro

        # NUEVA HEURÍSTICA 1: Distancia a las píldoras de poder (Cápsulas)
        capsulas = state.getCapsules()
        if capsulas:
            min_cap_dist = min(manhattanDistance(pacman_pos, c) for c in capsulas)
            trad_score += 10.0 / (min_cap_dist + 1)
            
        # NUEVA HEURÍSTICA 2: Seguridad contra callejones (Acciones legales disponibles)
        trad_score += len(state.getLegalActions(0)) * 3.0

        # 2) Evaluación de la Red Neuronal
        neural_score = 0.0
        if self.neural_brain and self.neural_brain.model is not None:
            state_matrix = self.neural_brain.state_to_matrix(state)
            state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.neural_brain.device)
            with torch.no_grad():
                output = self.neural_brain.model(state_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            
            legal_actions = state.getLegalActions(0)
            for i, action in enumerate(self.neural_brain.idx_to_action.values()):
                if action in legal_actions:
                    neural_score += probabilities[i] * 100.0

        # 3) Estrategia de pesos dinámicos (Opcional Tarea 3)
        any_scared = any(g.scaredTimer > 0 for g in ghost_states)
        current_w_heuristic = self.w_heuristic * 2.0 if any_scared else self.w_heuristic
        current_w_neural = self.w_neural * 0.5 if any_scared else self.w_neural

        # CORRECCIÓN DE FÓRMULA: Sumamos la puntuación real del juego para dar realismo físico
        return state.getScore() + (current_w_heuristic * trad_score) + (current_w_neural * neural_score)

    def getAction(self, gameState: GameState):
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluation_combined(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)
            
            if not actions:
                return self.evaluation_combined(state)

            if agentIndex == 0: # Nodo MAX (Pacman)
                v = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if v > beta: return v
                    alpha = max(alpha, v)
                return v
            else: # Nodo MIN (Fantasmas)
                v = float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v)
                return v

        # Selección de movimiento en la raíz
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

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            # si todavía no hemos entrenado el modelo usamos la 
            # heurística para poder jugar y sacar los datos
            score = state.getScore()
            pacman_pos = state.getPacmanPosition()
            food = state.getFood().asList()
            ghost_states = state.getGhostStates()
            
            if food:
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
                score += 1.0 / (min_food_distance + 1)
            
            for ghost_state in ghost_states:
                ghost_pos = ghost_state.getPosition()
                ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
                if ghost_state.scaredTimer > 0:
                    score += 50 / (ghost_distance + 1)
                else:
                    if ghost_distance <= 2:
                        score -= 200
            capsulas = state.getCapsules()
            if capsulas:
                min_capsule_distance = min(manhattanDistance(pacman_pos, cap_pos) for cap_pos in capsulas)
                score += 10.0 / (min_capsule_distance + 1)
            acciones_legales = state.getLegalActions(0)
            score += len(acciones_legales) * 3.0
            
            return score
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca
        capsulas = state.getCapsules()
        if capsulas:
            min_capsule_distance = min(manhattanDistance(pacman_pos, cap_pos) for cap_pos in capsulas)
            score += 10.0 / (min_capsule_distance + 1)
            
        acciones_legales = state.getLegalActions(0)
        score += len(acciones_legales) * 3.0
        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)
# Registramos la clase para que el importador de pacman.py la encuentre
