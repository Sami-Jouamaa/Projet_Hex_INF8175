import copy
import random
import numpy as np
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.game.stateful_action import StatefulAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time


class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """
    BRIDGE_TO_GAPS = {
        (1, 1): [(1, 0), (0, 1)],
        (2, -1): [(1, 0), (1, -1)],
        (1, -2): [(0, -1), (1, -1)],
        (-1, -1): [(-1, 0), (0, -1)],
        (-2, 1): [(-1, 0), (-1, 1)],
        (-1, 2): [(0, 1), (-1, 1)],
    }

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self.nb_moves = 0
        self.dimensions = [0, 0]
        self.players = []
        # {(0,0): [(1,0), (0,1)]}
        self.bridge_protection = dict()
        self.opponent_bridges = dict()
        self.opponent_bridges_attack = dict()
        # distance between every point
        self.distances = []
        # list of moves as position in order from first to last
        self.move_history = []
        self.opponent_move_history = []
        self.center = [0, 0]
        self.last_move = [0, 0]
        self.last_opponent_move = [0, 0]

        self.nb_moves_until_loss = np.inf
        self.nb_moves_until_win = np.inf


    def center_control_score(self, state):

        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]
        center = size // 2

        score = 0

        for (i, j), piece in board.items():

            dist = abs(i - center) + abs(j - center)

            value = max(0, (size // 2 - dist))  # plus proche centre = mieux

            if piece.get_type() == self.get_piece_type():
                score += value
            else:
                score -= value

        return score

    def shortest_path(self, state, player_piece):
        import heapq

        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]

        INF = float("inf")
        dist = {}
        parent = {}

        heap = []

        if player_piece == "R":
            for j in range(size):
                pos = (0, j)

                if pos in board:
                    if board[pos].get_type() == player_piece:
                        cost = 0
                    else:
                        continue
                else:
                    cost = 1

                dist[pos] = cost
                parent[pos] = None
                heapq.heappush(heap, (cost, pos))

        else:
            for i in range(size):
                pos = (i, 0)

                if pos in board:
                    if board[pos].get_type() == player_piece:
                        cost = 0
                    else:
                        continue
                else:
                    cost = 1

                dist[pos] = cost
                parent[pos] = None
                heapq.heappush(heap, (cost, pos))

        visited = set()

        while heap:
            cost, pos = heapq.heappop(heap)

            if pos in visited:
                continue
            visited.add(pos)

            i, j = pos

            if (player_piece == "R" and i == size - 1) or \
            (player_piece == "B" and j == size - 1):

                # reconstruct path
                path = []
                cur = pos
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()

                return cost, path

            for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():

                if not state.in_board((ni, nj)):
                    continue

                if (ni, nj) in visited:
                    continue

                if (ni, nj) in board:
                    if board[(ni, nj)].get_type() == player_piece:
                        new_cost = cost
                    else:
                        continue
                else:
                    new_cost = cost + 1

                if (ni, nj) not in dist or new_cost < dist[(ni, nj)]:
                    dist[(ni, nj)] = new_cost
                    parent[(ni, nj)] = pos
                    heapq.heappush(heap, (new_cost, (ni, nj)))

        return INF, []

    def find_bridge_cells(self, state, my_piece):
        """
        Find all empty cells that would create a bridge between two of our pieces.
        Returns a dict with bridge cell as key and list of the two pieces it connects.
        """
        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]

        # Bridge directions in hex grid (2-step moves)
        bridge_directions = [
            (2, 0), (0, 2), (2, -1), (1, 1),
            (-1, 2), (-2, 1), (-2, 0), (0, -2),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

        bridge_candidates = {}



        # For each of our pieces
        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue

            # Check all bridge directions
            for di, dj in bridge_directions:
                ni, nj = i + di, j + dj

                # Check if target position is within board
                if not (0 <= ni < size and 0 <= nj < size):
                    continue

                # Check if target position has our piece
                if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                    # Found two pieces that could form a bridge

                    # Find the middle cell(s) that would complete the bridge
                    middle_cells = self.get_bridge_middle_cells((i, j), (ni, nj))

                    for mid_cell in middle_cells:
                        # Check if middle cell is empty
                        # Is the bridge useful?

                        if mid_cell not in board:
                            # current_dist = self.shortest_path(state, my_piece)
                            #
                            # tmp_state = self.simulate_move(state, mid_cell, my_piece)
                            #
                            # new_dist = self.shortest_path(tmp_state, my_piece)
                            #
                            #
                            # if new_dist < current_dist:
                            if mid_cell not in bridge_candidates:
                                bridge_candidates[mid_cell] = []
                            bridge_candidates[mid_cell].append(((i, j), (ni, nj)))

        return bridge_candidates

    def get_bridge_middle_cells(self, pos1, pos2):
        """
        Calculate the middle cell(s) that would connect two pieces into a bridge.
        For hex grid, a bridge typically has 1-2 empty cells between pieces.
        """
        i1, j1 = pos1
        i2, j2 = pos2

        di = i2 - i1
        dj = j2 - j1

        # Different bridge patterns require different middle cells
        if (di, dj) == (2, 0):
            return [(i1 + 1, j1)]
        elif (di, dj) == (0, 2):
            return [(i1, j1 + 1)]
        elif (di, dj) == (2, -1):
            return [(i1 + 1, j1), (i1 + 1, j1 - 1)]
        elif (di, dj) == (1, 1):
            return [(i1, j1 + 1), (i1 + 1, j1)]
        elif (di, dj) == (-1, 2):
            return [(i1, j1 + 1), (i1 - 1, j1 + 1)]
        elif (di, dj) == (-2, 1):
            return [(i1 - 1, j1), (i1 - 1, j1 + 1)]
        elif (di, dj) == (-2, 0):
            return [(i1 - 1, j1)]
        elif (di, dj) == (0, -2):
            return [(i1, j1 - 1)]
        elif (di, dj) == (1, -2):
            return [(i1, j1 - 1), (i1 + 1, j1 - 1)]
        elif (di, dj) == (2, -1):
            return [(i1 + 1, j1), (i1 + 1, j1 - 1)]
        else:
            # Default: return the midpoint
            return [((i1 + i2) // 2, (j1 + j2) // 2)]

    def bridge_formation_score(self, state, my_piece):
        """
        Score how many bridge opportunities we have and prioritize them.
        """
        bridge_candidates = self.find_bridge_cells(state, my_piece)

        if not bridge_candidates:
            return 0

        score = 0
        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]

        for bridge_cell, connected_pieces in bridge_candidates.items():
            # Base score for creating a bridge
            cell_score = 15

            # Bonus if bridge extends toward winning side
            i, j = bridge_cell

            if my_piece == "R":  # Red connects top-bottom
                # Closer to bottom = better
                distance_to_bottom = size - 1 - i
                cell_score += (size - distance_to_bottom) * 2
            else:  # Blue connects left-right
                distance_to_right = size - 1 - j
                cell_score += (size - distance_to_right) * 2

            # Bonus if bridge connects to existing cluster
            for piece1, piece2 in connected_pieces:
                # Check if either piece is already connected to a side
                if self.is_connected_to_side(state, piece1, my_piece):
                    cell_score += 20
                if self.is_connected_to_side(state, piece2, my_piece):
                    cell_score += 20

            # Bonus if bridge is in center (strategic position)
            center = size // 2
            if abs(i - center) + abs(j - center) < size // 3:
                cell_score += 10

            score += cell_score

        return score


    def is_connected_to_side(self, state, piece_pos, my_piece):
        """
        Check if a piece is connected to its starting side.
        """
        size = state.get_rep().get_dimensions()[0]
        board = state.get_rep().get_env()

        i, j = piece_pos

        if my_piece == "R":
            # Check if connected to top row
            if i == 0:
                return True

            # BFS to check connection to top
            visited = set()
            stack = [piece_pos]

            while stack:
                ci, cj = stack.pop()
                if (ci, cj) in visited:
                    continue
                visited.add((ci, cj))

                if ci == 0:  # Reached top
                    return True

                for _, (ptype, (ni, nj)) in state.get_neighbours(ci, cj).items():
                    if 0 <= ni < size and 0 <= nj < size:
                        if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                            if (ni, nj) not in visited:
                                stack.append((ni, nj))

        else:  # "B"
            # Check if connected to left column
            if j == 0:
                return True

            visited = set()
            stack = [piece_pos]

            while stack:
                ci, cj = stack.pop()
                if (ci, cj) in visited:
                    continue
                visited.add((ci, cj))

                if cj == 0:  # Reached left
                    return True

                for _, (ptype, (ni, nj)) in state.get_neighbours(ci, cj).items():
                    if 0 <= ni < size and 0 <= nj < size:
                        if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                            if (ni, nj) not in visited:
                                stack.append((ni, nj))

        return False

    def simulate_move(self, state, position, player):

        if not state.in_board(position):
            return None

        if position in state.get_rep().get_env():
            return None  # ❗ IMPORTANT
        return state.apply_action(
            StatelessAction({"piece": player, "position": position})
        )

    def evaluate(self, state):
        early_game = state.get_step() < 10 #equivaut a 5 coups
        my_piece = self.get_piece_type()
        opponent = "B" if my_piece == "R" else "R"

        my_dist, my_path = self.shortest_path(state, my_piece)
        opp_dist, opp_path = self.shortest_path(state, opponent)

        contact = self.has_contact(state, my_piece, opponent)

        our_bridges = self.how_many_bridges(state, my_piece)
        bridge_gain = self.should_complete_bridge(state, my_piece, our_bridges)

        bridge_gap = self.bridge_formation_score(state, my_piece)

        block_score = self.blocking_score(state, my_piece, opponent)
        center_score = self.center_control_score(state)

        # =========================================
        # 🔴 PHASE 3 — DEFENSE (PRIORITÉ GLOBALE)
        # =========================================
        if opp_dist < my_dist:
            print("Phase 3")

            return (
                    (opp_dist - my_dist) * 6
                    + block_score * 4
            )

        # =========================================
        # 🔴 PHASE 2 — CONTACT
        # =========================================
        if contact:
            print("Phase 2")

            # =========================================
            # Pre-Phase — Center control
            # =========================================
            if early_game:
                print("Early Game Contact")
                return (
                        (opp_dist - my_dist) * 2
                        + self.center_control_score(state) * 5
                        + len(self.how_many_bridges(state, my_piece)) * 4
                )
            # 🛡️ PRIORITÉ ABSOLUE : compléter bridge menacé
            if bridge_gain > 0:
                return 10000 + bridge_gain * 50

            # 🔗 sinon : expansion intelligente
            return (
                    (opp_dist - my_dist) * 4
                    #+ len(our_bridges) * 3 #Aggrandi la surface
                    #+ bridge_gap * 4  # Soit on cree plus de gap-bridges
                    + block_score * 2
                    #+ center_score * 2
            )

        # =========================================
        # 🟢 PHASE 1 — BUILD (PAS DE CONTACT)
        # =========================================
        else:
            print("Phase 1")
            return (
                    (opp_dist - my_dist) * 3
                    #+ len(our_bridges) * 6  # 🔥 priorité bridges
                    + bridge_gap * 6
                    + center_score * 2
            )


    def dead_cells(self, my_dist, opp_dist):
        diff = my_dist - opp_dist
        if diff > 3:
            return -4
        elif diff < -3:
            return 6
        return 0

    def edge_progress(self, state, my_piece):
        board = state.get_rep().get_env()
        coords = []

        for (i, j), piece in board.items():
            if piece.get_type() == my_piece:
                coords.append((i, j))

        if not coords:
            return 0

        if my_piece == "R":
            rows = [i for i, _ in coords]
            return max(rows) - min(rows)

        else:
            columns = [j for _, j in coords]
            return max(columns) - min(columns)

    def calculate_connection_score(self, state, my_piece):
        board = state.get_rep().get_env()
        score = 0
        for (i, j), pieces in board.items():
            if pieces.get_type() == my_piece:
                for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
                    if ptype == my_piece:
                        score += 1
        return score / 2

    def has_contact(self, state, my_piece, opponent):

        board = state.get_rep().get_env()

        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue

            for _, (ptype, _) in state.get_neighbours(i, j).items():
                if ptype == opponent:
                    print("Opp Piece made Contact!", ptype)
                    return True

        return False

    def are_connected(self, state, start, end, my_piece):
        visited = set()
        stack = [start]

        board = state.get_rep().get_env()

        while stack:
            node = stack.pop()

            if node == end:
                return True

            for _, (ptype, (ni, nj)) in state.get_neighbours(*node).items():
                if (ni, nj) in visited:
                    continue

                if not state.in_board((ni, nj)):
                    continue

                if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                    visited.add((ni, nj))
                    stack.append((ni, nj))

        return False

    def how_many_bridges(self, state, my_piece):
        board = state.get_rep().get_env()
        bridges = set()
        directions = [
            (1, 1), (2, -1), (1, -2),
            (-1, -1), (-2, 1), (-1, 2)
        ]
        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue
            for di, dj in directions:
                bridge_end = (i + di, j + dj)
                gaps_filled_by_enemy = 0
                if (bridge_end in board) and board[bridge_end].get_type() == my_piece:
                    dx, dy = di, dj
                    valid_bridge = True

                    if (dx, dy) in self.BRIDGE_TO_GAPS:
                        for gx, gy in self.BRIDGE_TO_GAPS[(dx, dy)]:
                            gap = (i + gx, j + gy)
                            if (gap in board) and (board[gap].get_type() == my_piece):
                                valid_bridge = False
                                break
                            gaps_filled_by_enemy += 1
                    if gaps_filled_by_enemy == 2:
                        valid_bridge = False #seems to be always false
                    if valid_bridge: #will probably never go enter
                        # 🔥 NOUVEAU : skip si déjà connecté
                        if self.are_connected(state, (i, j), bridge_end, my_piece):
                            print("They are connected", bridge_end)
                            continue

                        bridge = tuple(sorted([(i, j), bridge_end]))
                        bridges.add(bridge)
        return bridges

    def should_complete_bridge(self, state, my_piece, our_bridges):
        board = state.get_rep().get_env()

        for start, end in our_bridges:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            for gx, gy in self.BRIDGE_TO_GAPS[(dx, dy)]:
                gap = (start[0] + gx, start[1] + gy)

                if gap in board and board[gap].get_type() != my_piece:
                    return 1
        return 0

    def blocking_score(self, state, my_piece, opponent):

        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]

        score = 0

        for i in range(size):
            for j in range(size):

                pos = (i, j)

                # skip occupied cells
                if pos in board:
                    continue

                neighbours = state.get_neighbours(i, j)

                opp_neighbors = []

                for _, (ptype, (ni, nj)) in neighbours.items():
                    if ptype == opponent:
                        opp_neighbors.append((ni, nj))

                if not opp_neighbors:
                    continue

                if opponent == "R": 
                    for (ni, nj) in opp_neighbors:
                        progress = ni / size
                        score += 10 * progress

                else:
                    for (ni, nj) in opp_neighbors:
                        progress = nj / size
                        score += 10 * progress

                if len(opp_neighbors) >= 2:
                    score += 20

                if len(opp_neighbors) == 2:
                    (a_i, a_j), (b_i, b_j) = opp_neighbors

                    if abs(a_i - b_i) > 1 or abs(a_j - b_j) > 1:
                        score += 40

                if opponent == "R" and i == size - 1:
                    score += 25

                if opponent == "B" and j == size - 1:
                    score += 25

        return score

    def vulnerability_score(self, state, my_piece):

        board = state.get_rep().get_env()
        score = 0

        for (i, j), piece in board.items():

            if piece.get_type() != my_piece:
                continue

            neighbours = state.get_neighbours(i, j)

            empty = 0
            enemy = 0

            for _, (ptype, _) in neighbours.items():
                if ptype == "EMPTY":
                    empty += 1
                elif ptype != my_piece:
                    enemy += 1

            # pièce entourée = danger
            if enemy >= 3:
                score -= 2

            # peu de liberté = mauvais
            if empty <= 1:
                score -= 2

        return score

    def get_top_actions(self, state):

        my_piece = self.get_piece_type()
        opponent = "B" if my_piece == "R" else "R"

        my_dist, _ = self.shortest_path(state, my_piece)
        opp_dist, opp_path = self.shortest_path(state, opponent)
        
        if opp_dist <= 5:
            emergency = True
        else:
            emergency = False
        
        board = state.get_rep().get_env()
        critical_blocks = {pos for pos in opp_path if pos not in board}
        for pos in opp_path:
            if pos not in board:
                critical_blocks.add(pos)

        if opp_dist < 3:
            k = 15
        elif my_dist < 3:
            k = 12
        else:
            k = 8

        actions = list(state.generate_possible_stateful_actions())

        scored = []
        blocking_moves = []

        for action in actions:
            next_state = action.get_next_game_state()
            position = self.get_action_position(state, action)

            new_opp_dist, _ = self.shortest_path(next_state, opponent)

            if position in critical_blocks or new_opp_dist > opp_dist:
                blocking_moves.append(action)

            score = self.evaluate(next_state)
            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])
        blocking_moves.sort(
            key=lambda a: self.shortest_path(a.get_next_game_state(), opponent)[0],
            reverse=True
        )
        if emergency and blocking_moves:
            return blocking_moves
    
    # FIX: Merge blocking consideration into scoring
        scored = []
        for action in actions:
            next_state = action.get_next_game_state()
            position = self.get_action_position(state, action)
            
            # Base evaluation
            score = self.evaluate(next_state)
            
            # Boost blocking moves even in non-emergency
            new_opp_dist, _ = self.shortest_path(next_state, opponent)
            if position in critical_blocks or new_opp_dist > opp_dist:
                score += 25  # Significant boost for blocking potential
            
            scored.append((score, action))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [a for _, a in scored[:k]]

    def get_action_position(self, state, action):
        current_board = state.get_rep().get_env()
        next_board = action.get_next_game_state().get_rep().get_env()

        for pos in next_board:
            if pos not in current_board:
                return pos

        return None

    def minimax(self, state, depth, alpha, beta, maximizing):

        if depth == 0 or state.is_done():
            return self.evaluate(state)

        actions = self.get_top_actions(state)

        if maximizing:
            value = float("-inf")

            for action in actions:
                next_state = action.get_next_game_state()

                value = max(value, self.minimax(next_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:
            value = float("inf")

            for action in actions:
                next_state = action.get_next_game_state()

                value = min(value, self.minimax(next_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)

                if beta <= alpha:
                    break

            return value

    def is_bridge_under_threat(self, state, start, end, opponent):
        """
        Un bridge est menacé si un des deux endpoints est en contact avec l’adversaire.
        """
        for pos in [start, end]:
            for _, (ptype, _) in state.get_neighbours(*pos).items():
                if ptype == opponent:
                    return True
        return False

    def get_bridge_gap(self, state, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        for gx, gy in self.BRIDGE_TO_GAPS[(dx, dy)]:
            gap = (start[0] + gx, start[1] + gy)

            if state.in_board(gap) and gap not in state.get_rep().get_env():
                return gap

        return None

    def get_critical_bridge_move(self, state, my_piece):
        opponent = "B" if my_piece == "R" else "R"

        bridges = self.how_many_bridges(state, my_piece)

        for start, end in bridges:

            if self.is_bridge_under_threat(state, start, end, opponent):

                gap = self.get_bridge_gap(state, start, end)

                if gap is not None:
                    return gap

        return None
    
    def determineFirstMove(self, state, my_piece):
        board = state.get_rep().get_env()
        if len(board.items()) == 0:
            return StatelessAction({"piece": self.piece_type, "position": self.center})
        
        opponent_pos = list(board.keys())[0]
        lastValidIndex = self.dimensions[0] - 1
        gridSize = self.dimensions[0]
        if my_piece == "R":
            if (opponent_pos[0] < gridSize/2) and (opponent_pos[0] > 0):
                new_position = (opponent_pos[0] - 1, opponent_pos[1])
            else:
                new_position = (opponent_pos[0] + 1, opponent_pos[1])
        else:
            if (opponent_pos[1] < gridSize/2) and (opponent_pos[1] > 0) and (opponent_pos[1] < lastValidIndex):
                new_position = (opponent_pos[0], opponent_pos[1] - 1)
            else:
                new_position = (opponent_pos[0], opponent_pos[1] + 1)
        return StatelessAction({"piece": self.piece_type, "position": new_position})

        

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15 * 60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        # TODO
        self.nb_moves += 1
        self.dimensions = current_state.get_rep().get_dimensions()
        self.center = (self.dimensions[0] // 2, self.dimensions[1] // 2)

        # fallback (au cas où timeout direct)
        fallback_actions = list(current_state.generate_possible_stateful_actions())
        if fallback_actions:
            best_action = fallback_actions[0]

        # premier coup = centre
        if self.nb_moves == 1:
            return self.determineFirstMove(current_state, self.get_piece_type())
        
        #When a bridge is in danger, it connects it
        critical_move = self.get_critical_bridge_move(current_state, self.piece_type)

        if critical_move is not None:
            return StatelessAction({
                "piece": self.piece_type,
                "position": critical_move
            })
        start = time.time()

        # budget temps intelligent (adaptatif)
        moves_left = max(1, (self.dimensions[0] * self.dimensions[1]) - current_state.get_step())
        time_limit = min(remaining_time / moves_left * 1.5, 2.5)

        best_action = None #If best_actions ends up to be None, System Crashes!
        #actions = list(current_state.generate_possible_stateful_actions())
        #best_action = random.choice(actions) if actions else None
        depth = 1

        while True:

            if time.time() - start > time_limit:
                break

            best_score = float("-inf")
            actions = self.get_top_actions(current_state)
            for action in actions:

                if time.time() - start > time_limit:
                    break

                next_state = action.get_next_game_state()

                score = self.minimax(
                    next_state,
                    depth,
                    float("-inf"),
                    float("inf"),
                    False
                )
                print("Scores:", score, best_score)
                print("Action:", action)
                if score > best_score:
                    best_score = score
                    best_action = action

                print("Best action:", best_action)

            depth += 1  # iterative deepening
            print("Depth:", depth)
        if best_action is None: #les blocs vont etre dans les edges
            actions = list(current_state.generate_possible_stateful_actions())
            best_action = random.choice(actions) if actions else None
            print("Random action:", best_action)
        return best_action