import numpy as np
from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self.nb_moves = 0
        self.bridges = []
        # distance between every point
        self.distances = []

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        #TODO
        # To return at the end
        current_board = current_state.rep.env
        # list[int]
        dimensions = current_state.rep.get_dimensions()
        # player_pieces = (13, [(0, 0), (0, 1), (0, 2)...]) number of pieces and the position of each of them
        player_pieces = current_state.get_rep().get_pieces_player(self)
        print(dimensions) # [14, 14]
        # print(player_pieces[1]) # array of positions
        # print(player_pieces[1][0]) # to get the first position of the from the list of pieces' positions
        
        # check turn number, incrementation is done first
        self.nb_moves += 1
        if self.nb_moves == 1:
            # first move should be in the center
            # board is a square in terms of coords
            self.distances = np.full((dimensions[0], dimensions[1]), np.inf)
            pass
        else:
            # check if other player is close to winning
            # check who controls center
            # check if opponent has bridges
            # if we have bridges, protect them if necessary
            # create bridge toward our end (could create an array to store where they are)
            pass
            
        
            
        for position, piece in current_board.items():
            print(piece.get_type())
            # print(piece)
            # print(position)
            
        self.nb_moves += 1
        return next(current_state.generate_possible_stateful_actions())
        
