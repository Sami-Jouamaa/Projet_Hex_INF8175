import copy
import random
import numpy as np
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.game.stateful_action import StatefulAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import bridges as bridges
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
        # distance between every point
        self.distances = []
        # list of moves as position in order from first to last
        self.previous_board = None
        self.move_history = []
        self.opponent_move_history = []
        self.center = [0, 0]
        # self.bridge_protection = dict()
        # self.opponent_bridges = dict()
        # self.opponent_bridges_attack = dict()
        self.transposition_table = {}
        self.openning_msg = ''

    def opponent_history(self, move):
        if move is not None:
            pos = move.get_action()["position"]
            self.opponent_move_history.append(pos)

    def priority_heuristic_paramaters(self, state: GameStateHex, win_path):
        '''if nb_chain > 8 and state.get_step() > 8:
            return [1, 0, 0]
        elif nb_chain > 7 and state.get_step() <= 8:
            return [1, 0, 0.1]
        elif nb_chain <= 7 and state.get_step() <= 8:
            return [1, 0.5, 0.1]
        else:
            return [1, 0.5, 0]
         Chains are potential bridges
            '''
        if win_path >= 11:
            return [1, 0, 0, 0]
        elif state.get_step() >= 30:
            return [1, 0.3, 0.7, 0.2]
        else:
            return [1, 0.5, 0.4, 0.7]




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

    # def shortest_path(self, state, player_piece):
    #     import heapq
    #
    #     board = state.get_rep().get_env()
    #     size = state.get_rep().get_dimensions()[0]
    #
    #     INF = float("inf")
    #     dist = {}
    #
    #     heap = []
    #
    #     # initialisation (bords de départ)
    #     if player_piece == "R":  # top -> bottom
    #         for j in range(size):
    #             pos = (0, j)
    #
    #             if pos in board:
    #                 if board[pos].get_type() == player_piece:
    #                     cost = 0
    #                 else:
    #                     cost = INF
    #             else:
    #                 cost = 1
    #
    #             dist[pos] = cost
    #             heapq.heappush(heap, (cost, pos))
    #
    #     else:  # "B" left -> right
    #         for i in range(size):
    #             pos = (i, 0)
    #
    #             if pos in board:
    #                 if board[pos].get_type() == player_piece:
    #                     cost = 0
    #                 else:
    #                     cost = INF
    #             else:
    #                 cost = 1
    #
    #             dist[pos] = cost
    #             heapq.heappush(heap, (cost, pos))
    #
    #     visited = set()
    #
    #     while heap:
    #         cost, (i, j) = heapq.heappop(heap)
    #
    #         if (i, j) in visited:
    #             continue
    #         visited.add((i, j))
    #
    #         # condition de victoire
    #         if player_piece == "R" and i == size - 1:
    #             return cost
    #         if player_piece == "B" and j == size - 1:
    #             return cost
    #
    #         for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
    #
    #             if not state.in_board((ni, nj)):
    #                 continue
    #
    #             if (ni, nj) in visited:
    #                 continue
    #
    #             if (ni, nj) in board:
    #                 if board[(ni, nj)].get_type() == player_piece:
    #                     new_cost = cost
    #                 else:
    #                     continue  # bloqué
    #             else:
    #                 new_cost = cost + 1
    #
    #             if (ni, nj) not in dist or new_cost < dist[(ni, nj)]:
    #                 dist[(ni, nj)] = new_cost
    #                 heapq.heappush(heap, (new_cost, (ni, nj)))
    #
    #     return INF
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

    def are_connected(self, state, start, end, my_piece):
        if start == end:
            return True

        board = state.get_rep().get_env()
        if start not in board or end not in board:
            return False
        if board[start].get_type() != my_piece or board[end].get_type() != my_piece:
            return False

        visited = set()
        stack = [start]

        while stack:
            current = stack.pop()
            if current == end:
                return True
            if current in visited:
                continue
            visited.add(current)

            i, j = current
            for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
                if ptype == my_piece and (ni, nj) not in visited:
                    stack.append((ni, nj))

        return False

    def is_connected_to_goal_side(self, state, piece_pos, my_piece):
        size = state.get_rep().get_dimensions()[0]
        board = state.get_rep().get_env()

        if piece_pos not in board or board[piece_pos].get_type() != my_piece:
            return False

        i, j = piece_pos
        if my_piece == "R" and i == size - 1:
            return True
        if my_piece == "B" and j == size - 1:
            return True

        visited = set()
        stack = [piece_pos]

        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            visited.add((ci, cj))

            if my_piece == "R" and ci == size - 1:
                return True
            if my_piece == "B" and cj == size - 1:
                return True

            for _, (ptype, (ni, nj)) in state.get_neighbours(ci, cj).items():
                if 0 <= ni < size and 0 <= nj < size:
                    if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                        if (ni, nj) not in visited:
                            stack.append((ni, nj))

        return False

    def bridge_axis_progress(self, pos, my_piece):
        return pos[0] if my_piece == "R" else pos[1]

    def bridge_lateral_progress(self, pos, my_piece):
        return pos[1] if my_piece == "R" else pos[0]

    def distance_to_path(self, cell, path):
        if not path:
            return 99
        return min(abs(cell[0] - pi) + abs(cell[1] - pj) for pi, pj in path)

    def bridge_cell_priority(self, state, my_piece, bridge_cell, connected_pieces, win_path=None):
        size = state.get_rep().get_dimensions()[0]
        center = (size - 1) / 2

        if win_path is None:
            _, win_path = self.shortest_path(state, my_piece)

        path_distance = self.distance_to_path(bridge_cell, win_path)
        lateral = self.bridge_lateral_progress(bridge_cell, my_piece)

        score = 12

        if path_distance == 0:
            score += 40
        elif path_distance == 1:
            score += 24
        elif path_distance == 2:
            score += 10
        else:
            score -= 10

        score -= int(abs(lateral - center) * 2)

        for piece1, piece2 in connected_pieces:
            forward_span = abs(
                self.bridge_axis_progress(piece1, my_piece) -
                self.bridge_axis_progress(piece2, my_piece)
            )
            lateral_span = abs(
                self.bridge_lateral_progress(piece1, my_piece) -
                self.bridge_lateral_progress(piece2, my_piece)
            )

            score += forward_span * 14
            score -= lateral_span * 4

            start_link = (
                self.is_connected_to_side(state, piece1, my_piece) or
                self.is_connected_to_side(state, piece2, my_piece)
            )
            goal_link = (
                self.is_connected_to_goal_side(state, piece1, my_piece) or
                self.is_connected_to_goal_side(state, piece2, my_piece)
            )

            if start_link and goal_link:
                score += 45
            elif start_link or goal_link:
                score += 22

            if path_distance <= 1:
                score += 8

        if my_piece == "R":
            edge_distance = min(bridge_cell[0], size - 1 - bridge_cell[0])
        else:
            edge_distance = min(bridge_cell[1], size - 1 - bridge_cell[1])

        score += min(edge_distance, 4) * 2
        return score

    def get_relevant_bridge_moves(self, state, my_piece):
        bridge_candidates = self.find_bridge_cells(state, my_piece)
        if not bridge_candidates:
            return {}

        _, win_path = self.shortest_path(state, my_piece)
        priorities = {}

        for bridge_cell, connected_pieces in bridge_candidates.items():
            priorities[bridge_cell] = self.bridge_cell_priority(
                state,
                my_piece,
                bridge_cell,
                connected_pieces,
                win_path
            )

        return priorities

    def future_bridge_growth(self, state, next_state, my_piece):
        current_moves = self.get_relevant_bridge_moves(state, my_piece)
        next_moves = self.get_relevant_bridge_moves(next_state, my_piece)

        current_count = sum(1 for score in current_moves.values() if score >= 30)
        next_count = sum(1 for score in next_moves.values() if score >= 30)
        current_total = sum(current_moves.values())
        next_total = sum(next_moves.values())

        return (next_count - current_count), (next_total - current_total)
    def find_bridge_cells(self, state, my_piece):
        board = state.get_rep().get_env()
        size = state.get_rep().get_dimensions()[0]

        bridge_directions = [
            (2, -1), (1, 1),
            (-1, 2), (-2, 1),
            (1, -2), (-1, -1)
        ]

        bridge_candidates = {}

        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue

            for di, dj in bridge_directions:
                ni, nj = i + di, j + dj

                if not (0 <= ni < size and 0 <= nj < size):
                    continue

                if (ni, nj) not in board:
                    continue

                if board[(ni, nj)].get_type() != my_piece:
                    continue

                if self.are_connected(state, (i, j), (ni, nj), my_piece):
                    continue
                #Évite les bridges déjà connectés
                # if self.are_connected(state, (i, j), (ni, nj), my_piece):
                #     continue

                middle_cells = self.get_bridge_middle_cells((i, j), (ni, nj))

                # 🔥 DOIT être exactement 2
                if len(middle_cells) != 2:
                    continue

                # 🔥 NOUVELLE CONDITION CRITIQUE
                if not all(mid_cell not in board for mid_cell in middle_cells):
                    continue

                # 🔥 Ajouter les DEUX cellules comme options
                for mid_cell in middle_cells:
                    if mid_cell not in bridge_candidates:
                        bridge_candidates[mid_cell] = []

                    bridge_candidates[mid_cell].append(((i, j), (ni, nj)))

        return bridge_candidates

    def get_bridge_middle_cells(self, pos1, pos2):
        """
        Return ONLY bridges with exactly 2 empty cells (robust Hex bridges).
        """
        i1, j1 = pos1
        i2, j2 = pos2

        di = i2 - i1
        dj = j2 - j1

        if (di, dj) == (2, -1):
            return [(i1 + 1, j1), (i1 + 1, j1 - 1)]

        elif (di, dj) == (1, 1): #good
            return [(i1, j1 + 1), (i1 + 1, j1)]

        elif (di, dj) == (-1, 2):
            return [(i1, j1 + 1), (i1 - 1, j1 + 1)]

        elif (di, dj) == (-2, 1):
            return [(i1 - 1, j1), (i1 - 1, j1 + 1)]

        elif (di, dj) == (1, -2):
            return [(i1, j1 - 1), (i1 + 1, j1 - 1)]

        elif (di, dj) == (-1, -1):
            return [(i1, j1 - 1), (i1 - 1, j1)]

        return []

    def bridge_formation_score(self, state, my_piece):
        """
        Score only the bridge opportunities that really help the winning corridor.
        """
        bridge_candidates = self.get_relevant_bridge_moves(state, my_piece)

        if not bridge_candidates:
            return 0

        best_scores = sorted(bridge_candidates.values(), reverse=True)
        return sum(best_scores[:3])

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


    def count_potential_bridges(self, state, my_piece):
        relevant_bridges = self.get_relevant_bridge_moves(state, my_piece)
        return sum(1 for score in relevant_bridges.values() if score >= 30)

    def bridge_timing_score(self, state, my_piece, opponent):
        relevant_bridges = self.get_relevant_bridge_moves(state, my_piece)
        if not relevant_bridges:
            return 0

        bridge_values = sorted(relevant_bridges.values(), reverse=True)
        latent_pressure = sum(bridge_values[:3])
        strong_bridge_count = sum(1 for value in bridge_values if value >= 30)
        live_bridges = len(self.how_many_bridges(state, my_piece))
        contact = self.has_contact(state, my_piece, opponent)

        if contact:
            return latent_pressure * 0.45 + live_bridges * 8

        return latent_pressure * 0.7 + strong_bridge_count * 14 + live_bridges * 10

    def evaluate(self, state):
        early_game = state.get_step() < 10 #equivaut a 5 coups
        my_piece = self.get_piece_type()
        opponent = "B" if my_piece == "R" else "R"

        my_dist, _ = self.shortest_path(state, my_piece)
        opp_dist, _ = self.shortest_path(state, opponent)

        contact = self.has_contact(state, my_piece, opponent)

        #our_bridges_count = len(self.how_many_bridges(state, my_piece))
        potential_bridges = self.count_potential_bridges(state, my_piece)

        bridge_gap = self.bridge_formation_score(state, my_piece)
        bridge_timing = self.bridge_timing_score(state, my_piece, opponent)

        block_score = self.blocking_score(state, my_piece, opponent)
        center_score = self.center_control_score(state)

        # =========================================
        # 🔴 PHASE 3 — DEFENSE (PRIORITÉ GLOBALE)
        # =========================================
        # if opp_dist <= my_dist:
        #     #print("Phase 3")
        #
        #     return (
        #             (opp_dist - my_dist) * 10  # 🔥 très fort
        #             + block_score * 6
        #     )

        # =========================================
        # 🔴 PHASE 2 — CONTACT
        # =========================================
        if contact:
            #print("Phase 2")

            # =========================================
            # Pre-Phase — Center control
            # =========================================
            # if early_game:
            #     #print("Early Game Contact")
            #     return (
            #             (opp_dist - my_dist) * 2
            #             + self.center_control_score(state) * 5
            #             #+ len(self.how_many_bridges(state, my_piece)) * 3
            #     )
            #Link chains to make bridges
            # print("Bridge Count", our_bridges_count)
            # if our_bridges_count >= 5: #Starting at 3, we letting to much chances to be blocked
            #     print((opp_dist - my_dist))
            #     return (
            #         (opp_dist - my_dist) * 10
            #         #+ our_bridges_count * 2
            #     )

            return (
                    (opp_dist - my_dist) * 4
                    + potential_bridges * 6 #Aggrandi la surface
                    + bridge_timing * 0.4
                    + block_score * 2
            )

        # =========================================
        # 🟢 PHASE 1 — BUILD (PAS DE CONTACT)
        # =========================================
        else:
            #print("Phase 1")
            return (
                    (opp_dist - my_dist) * 3
                    #+ len(our_bridges) * 6  # 🔥 priorité bridges
                    + bridge_gap * 6
                    + bridge_timing
                    #+ center_score * 2
            )

    def has_contact(self, state, my_piece, opponent):

        board = state.get_rep().get_env()

        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue

            for _, (ptype, _) in state.get_neighbours(i, j).items():
                if ptype == opponent:
                    return True

        return False

    def how_many_bridges(self, state, my_piece):
        board = state.get_rep().get_env()
        bridges = set()

        opponent = "B" if my_piece == "R" else "R"

        directions = [
            (1, 1), (2, -1), (1, -2),
            (-1, -1), (-2, 1), (-1, 2)
        ]

        for (i, j), piece in board.items():
            if piece.get_type() != my_piece:
                continue

            for di, dj in directions:
                bridge_end = (i + di, j + dj)

                # vérifier que endpoint est notre pièce
                if bridge_end not in board:
                    continue
                if board[bridge_end].get_type() != my_piece:
                    continue

                dx, dy = di, dj

                if (dx, dy) not in self.BRIDGE_TO_GAPS:
                    continue

                valid_bridge = True
                enemy_in_gaps = 0

                for gx, gy in self.BRIDGE_TO_GAPS[(dx, dy)]:
                    gap = (i + gx, j + gy)

                    if not state.in_board(gap):
                        valid_bridge = False
                        break

                    if gap in board:
                        if board[gap].get_type() == my_piece:
                            valid_bridge = False
                            break
                        elif board[gap].get_type() == opponent:
                            enemy_in_gaps += 1

                if enemy_in_gaps == len(self.BRIDGE_TO_GAPS[(dx, dy)]):
                    valid_bridge = False

                if valid_bridge:
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
        score = 0

        for (i, j), piece in board.items():

            neighbours = state.get_neighbours(i, j)

            # rouge si c'est une pièce adverse
            if piece.get_type() == opponent:

                for _, (ptype, _) in neighbours.items():
                    if ptype == my_piece:
                        score += 1  # on bloque

            # bleu si c'est notre pièce
            if piece.get_type() == my_piece:

                for _, (ptype, _) in neighbours.items():
                    if ptype == opponent:
                        score -= 1  # danger

        return score

    def get_top_actions(self, state):

        my_piece = self.get_piece_type()
        opponent = "B" if my_piece == "R" else "R"
        board = state.get_rep().get_env()
        my_dist, _ = self.shortest_path(state, my_piece)
        opp_dist, opp_path = self.shortest_path(state, opponent)
        # print("My distance", my_dist)
        # print("Opp distance", opp_dist)
        # print(" opp_path ",opp_path)

        emergency = False
        if opp_dist <= 5:
            emergency = True
        critical_blocks = {pos for pos in opp_path if pos not in board}
        for pos in opp_path:
            if pos not in board:
                critical_blocks.add(pos)
        # # 🔥 adaptatif
        if opp_dist < 3:
            k = 15  # danger → explorer plus
        elif my_dist < 3:
            k = 12  # proche victoire
        else:
            k = 8  # normal

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

            new_my_dist, _ = self.shortest_path(next_state, my_piece)
            if new_my_dist < my_dist:
                score += (my_dist - new_my_dist) * 18

            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])
        blocking_moves.sort(
            key=lambda a: self.shortest_path(a.get_next_game_state(), opponent)[0],
            reverse=True
        )
        if emergency and blocking_moves:
            print("Critical blocks:", critical_blocks)
            print("Blocking moves:", blocking_moves)
            return blocking_moves
        return [a for _, a in scored[:k]]

    def get_action_position(self, state, action):
        current_board = state.get_rep().get_env()
        next_board = action.get_next_game_state().get_rep().get_env()

        for pos in next_board:
            if pos not in current_board:
                return pos

        return None

    def minimax(self, state, depth, alpha, beta, maximizing):
        state_key = self.hash_state(state)
        alpha_orig = alpha
        beta_orig = beta

        if state_key in self.transposition_table:
            entry = self.transposition_table[state_key]
            stored_depth = entry["depth"]
            stored_value = entry["value"]
            stored_flag = entry["flag"]

            if stored_depth >= depth:
                if stored_flag == "EXACT":
                    return stored_value
                if stored_flag == "LOWERBOUND":
                    alpha = max(alpha, stored_value)
                elif stored_flag == "UPPERBOUND":
                    beta = min(beta, stored_value)

                if alpha >= beta:
                    return stored_value

        if depth == 0 or state.is_done():
            return self.evaluate(state)

        actions = self.get_top_actions(state)


        #
        # self.transposition_table[state_key] = (depth, min_eval)
        # return min_eval

        if maximizing:
            max_value = float("-inf")

            for action in actions:
                next_state = action.get_next_game_state()

                max_value = max(max_value, self.minimax(next_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, max_value)

                if alpha >= beta:
                    break

            if max_value <= alpha_orig:
                flag = "UPPERBOUND"
            elif max_value >= beta_orig:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"

            self.transposition_table[state_key] = {
                "depth": depth,
                "value": max_value,
                "flag": flag,
            }
            return max_value

        #minimizing
        else:
            min_value = float("inf")

            for action in actions:
                next_state = action.get_next_game_state()

                min_value = min(min_value, self.minimax(next_state, depth - 1, alpha, beta, True))
                beta = min(beta, min_value)

                if beta <= alpha:
                    break

            if min_value <= alpha_orig:
                flag = "UPPERBOUND"
            elif min_value >= beta_orig:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"

            self.transposition_table[state_key] = {
                "depth": depth,
                "value": min_value,
                "flag": flag,
            }
            return min_value

    def get_threatened_bridge_move(self, state: GameStateHex):
        """
        Check if any of our bridges are threatened by opponent pieces.
        Returns the move to defend the bridge, or None if no threat detected.
        Handles multiple simultaneous threats (flower patterns) efficiently.
        """
        current_rep = state.get_rep()
        env = current_rep.env
        dims = current_rep.dimensions

        # Define bridge patterns (same as your existing code)
        if self.piece_type == "R":
            opponent = "B"
            internal_patterns = [
                ((1, -2), (0, -1), (1, -1), "i>0 and i<14"),
                ((-1, -1), (0, -1), (-1, 0), "i>0 and i<14"),
                ((-2, 1), (-1, 0), (-1, 1), "i>0 and i<14"),
                ((-1, 2), (0, 1), (-1, 1), "i>0 and i<14"),
                ((1, 1), (0, 1), (1, 0), "i>0 and i<14"),
                ((2, -1), (1, -1), (1, 0), "i>0 and i<14"),
            ]
            edge_patterns = [
                ((0, 0), (-1, 0), (-1, 1), "i==0"),
                ((0, 0), (1, 0), (1, -1), "i==13"),
            ]
        else:
            opponent = "R"
            internal_patterns = [
                ((1, -2), (0, -1), (1, -1), "j>0 and j<14"),
                ((-1, -1), (0, -1), (-1, 0), "j>0 and j<14"),
                ((-2, 1), (-1, 0), (-1, 1), "j>0 and j<14"),
                ((-1, 2), (0, 1), (-1, 1), "j>0 and j<14"),
                ((1, 1), (0, 1), (1, 0), "j>0 and j<14"),
                ((2, -1), (1, -1), (1, 0), "j>0 and j<14"),
            ]
            edge_patterns = [
                ((0, 0), (0, -1), (1, -1), "j==0"),
                ((0, 0), (0, 1), (-1, 1), "j==13"),
            ]

        # Get our pieces
        if self.piece_type == "R":
            _, our_pieces = current_rep.get_pieces_player(state.players[0])
        else:
            _, our_pieces = current_rep.get_pieces_player(state.players[1])

        # Track threats: for each defensive move, count how many bridges it saves
        defensive_moves = {}  # position -> (bridges_saved, is_critical)

        for (i, j) in our_pieces:
            # Check internal bridge patterns
            for (di2, dj2), (e1i, e1j), (e2i, e2j), condition in internal_patterns:
                if not eval(condition):
                    continue

                piece2 = (i + di2, j + dj2)
                empty1 = (i + e1i, j + e1j)
                empty2 = (i + e2i, j + e2j)

                # Check if this forms a valid bridge
                if not (0 <= piece2[0] < dims[0] and 0 <= piece2[1] < dims[1]):
                    continue
                piece2_cell = env.get(piece2)
                if piece2_cell is None or piece2_cell.piece_type != self.piece_type:
                    continue

                empty1_cell = env.get(empty1)
                empty2_cell = env.get(empty2)

                # Bridge is threatened if exactly one empty is occupied by opponent
                threat_case1 = (empty1_cell is not None and empty1_cell.piece_type == opponent and empty2_cell is None)
                threat_case2 = (empty2_cell is not None and empty2_cell.piece_type == opponent and empty1_cell is None)

                if threat_case1:
                    # Playing in empty2 saves this bridge
                    if 0 <= empty2[0] < dims[0] and 0 <= empty2[1] < dims[1]:
                        if empty2 not in defensive_moves:
                            defensive_moves[empty2] = {'saved': 0, 'critical': False}
                        defensive_moves[empty2]['saved'] += 1
                        # Mark as critical if the bridge is almost complete (one move away from winning)
                        if self.is_critical_bridge(state, (i, j), piece2, empty1, empty2):
                            defensive_moves[empty2]['critical'] = True

                elif threat_case2:
                    # Playing in empty1 saves this bridge
                    if 0 <= empty1[0] < dims[0] and 0 <= empty1[1] < dims[1]:
                        if empty1 not in defensive_moves:
                            defensive_moves[empty1] = {'saved': 0, 'critical': False}
                        defensive_moves[empty1]['saved'] += 1
                        if self.is_critical_bridge(state, (i, j), piece2, empty1, empty2):
                            defensive_moves[empty1]['critical'] = True

            # Check edge bridge patterns
            for (pi, pj), (e1i, e1j), (e2i, e2j), condition in edge_patterns:
                if not eval(condition):
                    continue

                empty1 = (i + e1i, j + e1j)
                empty2 = (i + e2i, j + e2j)

                empty1_cell = env.get(empty1) if 0 <= empty1[0] < dims[0] and 0 <= empty1[1] < dims[1] else None
                empty2_cell = env.get(empty2) if 0 <= empty2[0] < dims[0] and 0 <= empty2[1] < dims[1] else None

                threat_case1 = (empty1_cell is not None and empty1_cell.piece_type == opponent and
                                empty2_cell is None and 0 <= empty2[0] < dims[0] and 0 <= empty2[1] < dims[1])
                threat_case2 = (empty2_cell is not None and empty2_cell.piece_type == opponent and
                                empty1_cell is None and 0 <= empty1[0] < dims[0] and 0 <= empty1[1] < dims[1])

                if threat_case1:
                    if empty2 not in defensive_moves:
                        defensive_moves[empty2] = {'saved': 0, 'critical': False}
                    defensive_moves[empty2]['saved'] += 1
                elif threat_case2:
                    if empty1 not in defensive_moves:
                        defensive_moves[empty1] = {'saved': 0, 'critical': False}
                    defensive_moves[empty1]['saved'] += 1

        if not defensive_moves:
            return None

        # Decision logic - prioritize moves that save multiple bridges
        # First, check for moves that save 2+ bridges (flower pattern defense)
        multi_save_moves = [(pos, data) for pos, data in defensive_moves.items() if data['saved'] >= 2]

        if multi_save_moves:
            # Sort by: critical bridges first, then by number saved
            multi_save_moves.sort(key=lambda x: (x[1]['critical'], x[1]['saved']), reverse=True)
            best_pos = multi_save_moves[0][0]
            print(f"🌼 Flower pattern defense: Playing {best_pos} saves {multi_save_moves[0][1]['saved']} bridges!")
            return StatelessAction({"piece": self.piece_type, "position": best_pos})

        # Check for critical bridges (winning path almost complete)
        critical_moves = [(pos, data) for pos, data in defensive_moves.items() if data['critical']]
        if critical_moves:
            critical_moves.sort(key=lambda x: x[1]['saved'], reverse=True)
            best_pos = critical_moves[0][0]
            print(f"⚠️ Critical bridge defense: Playing {best_pos}")
            return StatelessAction({"piece": self.piece_type, "position": best_pos})

        # Otherwise, save the most bridges (even if just 1)
        best_pos = max(defensive_moves.items(), key=lambda x: x[1]['saved'])[0]
        print(f"🛡️ Defending single bridge: Playing {best_pos}")
        return StatelessAction({"piece": self.piece_type, "position": best_pos})

    def is_critical_bridge(self, state, piece1, piece2, empty1, empty2):
        """
        Determine if a bridge is critical (close to winning).
        For example, if completing this bridge would create a winning path.
        """
        # This is a simplified check - you can enhance based on your game logic
        current_rep = state.get_rep()
        dims = current_rep.dimensions

        if self.piece_type == "R":
            # Check if either piece is on the edge (close to winning)
            on_north_edge = (piece1[0] == 0 or piece2[0] == 0)
            on_south_edge = (piece1[0] == dims[0] - 1 or piece2[0] == dims[0] - 1)
            return on_north_edge or on_south_edge
        else:
            # Blue player
            on_west_edge = (piece1[1] == 0 or piece2[1] == 0)
            on_east_edge = (piece1[1] == dims[1] - 1 or piece2[1] == dims[1] - 1)
            return on_west_edge or on_east_edge

    def hash_state(self, state: GameStateHex):
        board = state.get_rep().get_env()

        items = tuple(sorted((pos, piece.get_type()) for pos, piece in board.items()))
        #Add player_id
        # state.get_player_id()
        return items

    def get_bridge_path_move(self, state):
        my_piece = self.get_piece_type()
        board = state.get_rep().get_env()

        _, path = self.shortest_path(state, my_piece)

        if not path:
            return None

        for k in range(len(path) - 1):
            a = path[k]
            b = path[k + 1]

            # Si une des cases du chemin est vide → jouer là
            if a not in board:
                return a
            if b not in board:
                return b

            # Vérifier si a et b peuvent former un bridge
            middle_cells = self.get_bridge_middle_cells(a, b)

            for cell in middle_cells:
                if cell not in board:
                    return cell  # 🔥 EXACTEMENT ton turquoise

        return None

    def get_blocking_path_move(self, state: GameStateHex):
        my_piece = self.get_piece_type()
        opponent = "B" if my_piece == "R" else "R"
        board = state.get_rep().get_env()

        opp_dist, opp_path = self.shortest_path(state, opponent)

        if not opp_path:
            return None

        # Take the MOST central / impactful blocking cell
        candidates = [pos for pos in opp_path if pos not in board]

        if not candidates:
            return None

        # 🔥 prioritize middle of path (harder to bypass)
        return candidates[len(candidates) // 2]

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
        #print(self.opponent_move_history)
        if len(self.transposition_table) > 50000:
            self.transposition_table.clear()

        # fallback (au cas où timeout direct)
        fallback_actions = list(current_state.generate_possible_stateful_actions())
        if fallback_actions:
            best_action = fallback_actions[0]
        current_rep = current_state.get_rep()
        env = current_rep.get_env()

        if self.previous_board is not None:
            for pos, piece in env.items():
                if pos not in self.previous_board:
                    # Nouveau coup détecté
                    if piece.get_type() != self.piece_type:
                        self.opponent_move_history.append(pos)
        # openings
        random_oppening = random.randint(0,2)

        if random_oppening == 0:
            if current_state.get_step() < 2:
                if self.piece_type == "R":
                    if current_state.get_step() == 0:
                        return StatelessAction({"piece": self.piece_type, "position": (8,4)})
                else:
                    opening = StatelessAction({"piece": self.piece_type, "position": (4,8)})
                    if opening in current_state.generate_possible_stateless_actions():
                        return opening
                    else:
                        return StatelessAction({"piece": self.piece_type, "position": (8,4)})
            elif current_state.get_step() < 4:
                if self.piece_type == "R":
                    if current_state.get_step() == 2:
                        nb_blue_cases, blue_cases = current_rep.get_pieces_player(current_state.players[1])
                        if blue_cases[0] == (3, 10):
                            first_answer = StatelessAction({"piece": self.piece_type, "position": (4, 10)})
                            return first_answer
                        else:
                            nb_red_cases, red_cases = current_rep.get_pieces_player(current_state.players[0])
                            if red_cases[-1] == (9, 8):
                                first_answer = StatelessAction({"piece": self.piece_type, "position": (8, 4)})
                                if first_answer in current_state.generate_possible_stateless_actions():
                                    return first_answer
                                elif StatelessAction({"piece": self.piece_type, "position": (4,8)}) in current_state.generate_possible_stateless_actions():  # If our opening is already taken, try to counter it
                                    return StatelessAction({"piece": self.piece_type, "position": (4, 8)})
                                else:
                                    return StatelessAction({"piece": self.piece_type, "position": (3, 4)})
                            else:
                                first_answer = StatelessAction({"piece": self.piece_type, "position": (10, 3)})
                                if first_answer in current_state.generate_possible_stateless_actions():
                                    return first_answer
        else:
            if current_state.step < 2:
                if self.piece_type == "R":
                    if current_state.step == 0:  # premier tour à jouer pour les rouge (toujours possible de jouer)
                        opening = StatelessAction({"piece": self.piece_type, "position": (5, 7)})
                        return opening
                else:
                    opening = StatelessAction({"piece": self.piece_type, "position": (7, 5)})
                    if opening in current_state.generate_possible_stateless_actions():
                        return opening
                    else:  # If our opening is already taken, try to counter it
                        return StatelessAction({"piece": self.piece_type, "position": (7, 6)})

            # First Answer moves
            elif current_state.step < 4:
                if self.piece_type == "R":
                    if current_state.step == 2:  # second tour à jouer pour les rouge (toujours possible de jouer)
                        nb_blue_cases, blue_cases = current_rep.get_pieces_player(current_state.players[1])
                        if blue_cases[0] == (6, 6):
                            first_answer = StatelessAction({"piece": self.piece_type, "position": (6, 7)})
                            return first_answer
                        elif blue_cases[0] == (6, 7):
                            first_answer = StatelessAction({"piece": self.piece_type, "position": (6, 6)})
                            return first_answer
                        elif blue_cases[0] == (7, 6):
                            first_answer = StatelessAction({"piece": self.piece_type, "position": (6, 9)})
                            return first_answer
                        else:
                            return StatelessAction({"piece": self.piece_type, "position": (7, 6)})
                else:
                    nb_red_cases, red_cases = current_rep.get_pieces_player(current_state.players[0])
                    if red_cases[-1] == (9, 8):
                        first_answer = StatelessAction({"piece": self.piece_type, "position": (8, 4)})
                        if first_answer in current_state.generate_possible_stateless_actions():
                            return first_answer
                        elif StatelessAction({"piece": self.piece_type, "position": (
                        4, 8)}) in current_state.generate_possible_stateless_actions():  # If our opening is already taken, try to counter it
                            return StatelessAction({"piece": self.piece_type, "position": (4, 8)})
                        else:
                            return StatelessAction({"piece": self.piece_type, "position": (3, 4)})
                    else:
                        first_answer = StatelessAction({"piece": self.piece_type, "position": (10, 3)})
                        if first_answer in current_state.generate_possible_stateless_actions():
                            return first_answer
        # if self.has_contact(current_state, self.piece_type, "B" if self.piece_type == "R" else "R"):
            # move = self.get_bridge_path_move(current_state)
            #
            # if move is not None:
            #     print("Bridge pathway move!", move)
            #     return StatelessAction({
            #         "piece": self.piece_type,
            #         "position": move
            #     })
        my_piece = self.piece_type
        opponent = "B" if my_piece == "R" else "R"

        my_dist, _ = self.shortest_path(current_state, my_piece)
        opp_dist, _ = self.shortest_path(current_state, opponent)

        # 🔥 GLOBAL DEFENSE
        if opp_dist <= my_dist + 1:
            move = self.get_blocking_path_move(current_state)
            if move:
                print("🔥 Blocking opponent path!", move)
                return StatelessAction({
                    "piece": self.piece_type,
                    "position": move
                })

        threatened_move = self.get_threatened_bridge_move(current_state)
        if threatened_move is not None:
            print("Bridge threatened! Defending...")
            return threatened_move
        #Handles edges
        # 🔥 PRIORITÉ ABSOLUE : construire le pathway en cas de contact


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

                if score > best_score:
                    best_score = score
                    best_action = action



            depth += 1  # iterative deepening
        if best_action is None: #les blocs vont etre dans les edges
            actions = list(current_state.generate_possible_stateful_actions())
            best_action = random.choice(actions) if actions else None
            print("Random action:", best_action)
        self.previous_board = dict(env)
        return best_action
