from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.game.stateful_action import StatefulAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time



def is_bridge_relevant_to_win(self, state: GameStateHex, bridge_cell, connected_pieces, my_piece):
    """
    Determine if a bridge helps create a winning path.
    Different logic for Red (top-bottom) vs Blue (left-right).
    """
    size = state.get_rep().get_dimensions()[0]
    i, j = bridge_cell

    if my_piece == "R":
        # RED: Connects TOP to BOTTOM
        return self.is_bridge_relevant_red(state, bridge_cell, connected_pieces)
    else:
        # BLUE: Connects LEFT to RIGHT
        return self.is_bridge_relevant_blue(state, bridge_cell, connected_pieces)


def is_bridge_relevant_red(self, state: GameStateHex, bridge_cell, connected_pieces):
    """
    Evaluate bridge relevance for RED player (top-bottom connection).
    """
    size = state.get_rep().get_dimensions()[0]
    i, j = bridge_cell

    relevance = 0

    # 1. Check if bridge extends toward BOTTOM (winning side for Red)
    distance_to_bottom = size - 1 - i
    if distance_to_bottom < size / 2:
        # Closer to bottom = more relevant
        relevance += (size - distance_to_bottom) * 3

    # 2. Check if bridge connects to TOP (starting side)
    for piece1, piece2 in connected_pieces:
        # Check if either piece is connected to top row
        if self.is_connected_to_top(state, piece1):
            relevance += 40
        if self.is_connected_to_top(state, piece2):
            relevance += 40

        # Check if either piece is already near bottom
        if self.is_near_bottom(state, piece1, size):
            relevance += 30
        if self.is_near_bottom(state, piece2, size):
            relevance += 30

    # 3. Check if this bridge creates a path around opponent's blocks
    if self.bypasses_opponent_block(state, bridge_cell, connected_pieces, "R"):
        relevance += 50  # Major bonus for bypassing blocks

    # 4. Check if bridge is in a "must-fill" position (critical for connection)
    if self.is_critical_bridge(state, bridge_cell, connected_pieces, "R"):
        relevance += 100

    return relevance


def is_bridge_relevant_blue(self, state: GameStateHex, bridge_cell, connected_pieces):
    """
    Evaluate bridge relevance for BLUE player (left-right connection).
    """
    size = state.get_rep().get_dimensions()[0]
    i, j = bridge_cell

    relevance = 0

    # 1. Check if bridge extends toward RIGHT (winning side for Blue)
    distance_to_right = size - 1 - j
    if distance_to_right < size / 2:
        # Closer to right = more relevant
        relevance += (size - distance_to_right) * 3

    # 2. Check if bridge connects to LEFT (starting side)
    for piece1, piece2 in connected_pieces:
        # Check if either piece is connected to left column
        if self.is_connected_to_left(state, piece1):
            relevance += 40
        if self.is_connected_to_left(state, piece2):
            relevance += 40

        # Check if either piece is already near right side
        if self.is_near_right(state, piece1, size):
            relevance += 30
        if self.is_near_right(state, piece2, size):
            relevance += 30

    # 3. Check if this bridge creates a path around opponent's blocks
    if self.bypasses_opponent_block(state, bridge_cell, connected_pieces, "B"):
        relevance += 50

    # 4. Check if bridge is in a "must-fill" position
    if self.is_critical_bridge(state, bridge_cell, connected_pieces, "B"):
        relevance += 100

    return relevance


def is_connected_to_top(self, state: GameStateHex, piece_pos):
    """
    Check if a piece is connected to the top row (for Red).
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    my_piece = self.get_piece_type()

    if my_piece != "R":
        return False

    # BFS to check connection to top
    visited = set()
    stack = [piece_pos]

    while stack:
        i, j = stack.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))

        # Reached top row?
        if i == 0:
            return True

        for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
            if not (0 <= ni < size and 0 <= nj < size):
                continue
            if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

    return False


def is_connected_to_left(self, state: GameStateHex, piece_pos):
    """
    Check if a piece is connected to the left column (for Blue).
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    my_piece = self.get_piece_type()

    if my_piece != "B":
        return False

    visited = set()
    stack = [piece_pos]

    while stack:
        i, j = stack.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))

        # Reached left column?
        if j == 0:
            return True

        for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
            if not (0 <= ni < size and 0 <= nj < size):
                continue
            if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

    return False


def is_near_bottom(self, state: GameStateHex, piece_pos, size):
    """
    Check if a piece is near the bottom row (within 3 rows).
    """
    i, j = piece_pos
    return i >= size - 4  # Within 4 rows of bottom


def is_near_right(self, state, piece_pos, size):
    """
    Check if a piece is near the right column (within 3 columns).
    """
    i, j = piece_pos
    return j >= size - 4  # Within 4 columns of right


def bypasses_opponent_block(self, state: GameStateHex, bridge_cell, connected_pieces, player_color):
    """
    Check if this bridge helps bypass opponent blocks.
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    i, j = bridge_cell

    # Look for opponent pieces that might be blocking the direct path
    if player_color == "R":
        # For Red, check if there are opponent pieces blocking downward path
        for piece1, piece2 in connected_pieces:
            # Check the area between these pieces and bottom
            max_row = max(piece1[0], piece2[0], i)

            # Scan for opponent blocks below
            for row in range(max_row, min(size, max_row + 5)):
                for col in range(max(0, j - 2), min(size, j + 3)):
                    if (row, col) in board and board[(row, col)].get_type() != "R":
                        # Opponent piece found - bridge helps bypass
                        return True
    else:
        # For Blue, check if there are opponent pieces blocking rightward path
        for piece1, piece2 in connected_pieces:
            max_col = max(piece1[1], piece2[1], j)

            for col in range(max_col, min(size, max_col + 5)):
                for row in range(max(0, i - 2), min(size, i + 3)):
                    if (row, col) in board and board[(row, col)].get_type() != "B":
                        return True

    return False


def is_critical_bridge(self, state: GameStateHex, bridge_cell, connected_pieces, player_color):
    """
    Check if this bridge is critical for creating a winning path.
    A critical bridge is one where without it, the path would be blocked.
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    my_piece = self.get_piece_type()

    # Check if this bridge connects two large components
    component_sizes = []
    for piece1, piece2 in connected_pieces:
        size1 = self.get_component_size(state, piece1, my_piece)
        size2 = self.get_component_size(state, piece2, my_piece)
        component_sizes.append(size1 + size2)

    # If connecting large components, it's critical
    if max(component_sizes) > size:
        return True

    # Check if this is the only bridge connecting to a side
    if player_color == "R":
        # Count how many bridges connect to top
        top_connections = self.count_connections_to_side(state, "top", my_piece)
        if top_connections <= 2:  # Only few connections to top
            return True
    else:
        right_connections = self.count_connections_to_side(state, "right", my_piece)
        if right_connections <= 2:
            return True

    return False


def get_component_size(self, state: GameStateHex, start_pos, my_piece):
    """
    Get size of connected component containing start_pos.
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    visited = set()
    stack = [start_pos]
    count = 0

    while stack:
        i, j = stack.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        count += 1

        for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
            if not (0 <= ni < size and 0 <= nj < size):
                continue
            if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

    return count


def count_connections_to_side(self, state: GameStateHex, side, my_piece):
    """
    Count how many pieces are connected to a specific side.
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    connected_pieces = set()

    if side == "top":
        # Find all pieces on top row
        for j in range(size):
            if (0, j) in board and board[(0, j)].get_type() == my_piece:
                # Get entire component connected to this top piece
                component = self.get_component_at(state, (0, j), my_piece)
                connected_pieces.update(component)
    elif side == "right":
        for i in range(size):
            if (i, size - 1) in board and board[(i, size - 1)].get_type() == my_piece:
                component = self.get_component_at(state, (i, size - 1), my_piece)
                connected_pieces.update(component)

    return len(connected_pieces)


def get_component_at(self, state: GameStateHex, start_pos, my_piece):
    """
    Get all pieces in the component containing start_pos.
    """
    size = state.get_rep().get_dimensions()[0]
    board = state.get_rep().get_env()
    visited = set()
    stack = [start_pos]
    component = set()

    while stack:
        i, j = stack.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        component.add((i, j))

        for _, (ptype, (ni, nj)) in state.get_neighbours(i, j).items():
            if not (0 <= ni < size and 0 <= nj < size):
                continue
            if (ni, nj) in board and board[(ni, nj)].get_type() == my_piece:
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

    return component