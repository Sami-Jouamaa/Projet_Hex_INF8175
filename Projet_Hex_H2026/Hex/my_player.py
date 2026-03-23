import numpy as np
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.game.stateful_action import StatefulAction
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
        self.players = []
        self.bridges = []
        # distance between every point
        self.distances = []
        # list of moves as position in order from first to last
        self.move_history = []
        self.opponent_move_history = []
        self.center = [0, 0]

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        #TODO
        
        # player_pieces = (13, [(0, 0), (0, 1), (0, 2)...]) number of pieces and the position of each of them
        player_pieces = current_state.get_rep().get_pieces_player(self)
        # positions are in (0, 0), tuple, not list or array
        # print(player_pieces[1]) # array of positions
        # print(player_pieces[1][0]) # to get the first position of the from the list of pieces' positions
        
        # To return at the end
        valid_move = False
        
        # just an image when printing
        current_board = current_state.rep.env
        
        # dimensions are [14, 14] with index from 0 to 13
        dimensions = current_state.rep.get_dimensions()
        
        self.players = current_state.get_players()
        if self.players[0] == self:
            opponent_pieces = current_state.get_rep().get_pieces_player(self.players[1])
        else:
            opponent_pieces = current_state.get_rep().get_pieces_player(self.players[0])
        
        for moves in opponent_pieces:
            if moves not in self.opponent_move_history:
                self.opponent_move_history.append(moves)
            else:
                continue
        
        if dimensions[0] % 2 == 0:
            # for the index to be correct
            self.center = (round(dimensions[0]/2) - 1, round(dimensions[1]/2) - 1)
        else:
            # otherwise it'll round up and be offcenter by a bit
            self.center = (round(dimensions[0]/2) - 2, round(dimensions[1]/2) - 2)
        
        # check turn number, incrementation is done first
        self.nb_moves += 1
        while_iteration = 0
        while not valid_move:
            # first move works
            if self.nb_moves == 1:
                # first move should be in the center
                # board is a square in terms of coords
                
                possible_move = self.center
                self.distances = np.full((dimensions[0], dimensions[1]), np.inf)
                if possible_move not in self.move_history and possible_move not in self.opponent_move_history:
                    self.move_history.append(possible_move)
                    valid_move = True
                    return StatelessAction({"piece": self.piece_type, "position": possible_move})
                else:
                    possible_move = self.center
                    if while_iteration % 4 == 0:
                        possible_move[0] += 1
                    elif while_iteration % 4 == 1:
                        possible_move[0] -= 1
                    elif while_iteration % 4 == 2:
                        possible_move[1] += 1
                    else:
                        possible_move [1] -= 1
                    while_iteration = (while_iteration + 1) % 4
            else:
                # until we actually implement it so it can finish its execution and finish the game
                valid_move = True
                
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
        
