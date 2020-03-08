import numpy as np
from math import inf


class AIPlayer:

    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        limit = 3
        bestMove = (-inf, None)

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.minimax(next_board, move, limit, -inf, inf, True)
            if val > bestMove[0]:
                bestMove = (val, move)

        # raise NotImplementedError('Whoops I don\'t know what to do')
        return bestMove[1]

    def minimax(self, board, move, layer, alpha, beta, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or not children:
            return (move, self.evaluating_function(board))

        maxVal = (-inf, None)
        minVal = (inf, None)
        for child in children:
            val = self.minimax(self.apply_move(board, child),
                               child, layer - 1, alpha, beta, not maxMove)
            if val > maxVal[0]:
                maxVal = (val, child)
            elif val < minVal[0]:
                minVal = (val, child)

            if val > alpha:
                alpha = val
            elif val < beta:
                beta = val

            if beta <= alpha:
                break
        return maxVal if maxMove else minVal

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        limit = 3
        bestMove = (-inf, None)

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.expectimax(next_board, move, limit, True)
            if val > bestMove[0]:
                bestMove = (val, move)

        return bestMove[1]

    def expectimax(self, board, move, layer, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or not children:
            return (move, self.evaluating_function(board))

        maxVal = (-inf, None)
        expVal = 0
        for child in children:
            val = self.minimax(self.apply_move(board, child),
                               child, layer - 1, not maxMove)
            if maxMove and val > maxVal[0]:
                maxVal = (val, child)
            else:
                expVal += val / len(children)

        return maxVal if maxMove else expVal

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        return 0

    def possible_moves(self, board):
        return [i for row, i in enumerate(board[0]) if 0 in board[:, col] for col in row]
        # valid_cols = []
        # for col in range(board.shape[1]):
        #     if 0 in board[:, col]:
        #         valid_cols.append((col, layer + 1))
        # return valid_cols

    def apply_move(self, board, col):
        temp = board
        column = board[:, col]
        row = next((r for r in column if r > 0), default=len(column)) - 1
        temp[row][col] = self.player_number
        return temp


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        print(board)
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move