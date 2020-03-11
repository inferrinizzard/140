import numpy as np

BIG_NUMBER = 1000


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
        depth = 100
        bestMove = None
        bestVal = -BIG_NUMBER

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.minimax(next_board, move, depth, -
                               BIG_NUMBER, BIG_NUMBER, False)
            if val > bestVal:
                bestVal = val
                bestMove = move

        return bestMove

    def minimax(self, board, move, layer, alpha, beta, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or len(children) == 0:
            return self.evaluation_function(board)

        if maxMove:
            maxVal = -BIG_NUMBER
            for child in children:
                val = self.minimax(self.apply_move(board, child),
                                   child, layer - 1, alpha, beta, not maxMove)
                if val > maxVal:
                    maxVal = val

                if val > alpha:
                    alpha = val
                elif val < beta:
                    beta = val

                if beta <= alpha:
                    break
            return maxVal

        else:
            minVal = BIG_NUMBER
            for child in children:
                val = self.minimax(self.apply_move(board, child),
                                   child, layer - 1, alpha, beta, not maxMove)
                if val < minVal:
                    minVal = val

                if val > alpha:
                    alpha = val
                elif val < beta:
                    beta = val

                if beta <= alpha:
                    break
            return minVal

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
        depth = 3
        bestMove = None
        bestVal = -BIG_NUMBER

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.expectimax(next_board, move, depth, False)
            if val > bestVal:
                bestVal = val
                bestMove = move

        return bestMove

    def expectimax(self, board, move, layer, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or len(children) == 0:
            return self.evaluation_function(board)

        if maxMove:
            maxVal = -BIG_NUMBER
            for child in children:
                val = self.expectimax(self.apply_move(board, child),
                                      child, layer - 1, not maxMove)
            if val > maxVal:
                maxVal = val
            return maxVal
        else:
            expectedVal = 0
            for child in children:
                val = self.expectimax(self.apply_move(board, child),
                                      child, layer - 1, not maxMove)
                expectedVal += val / len(children)
            return expectedVal

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
        score = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if i <= 3:
                    three = True
                    for k in range(i, i + 3):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 4
                if i > 2:
                    three = True
                    for k in range(i - 3, i):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 4
                if i > 0 and i < 5:
                    three = True
                    for k in range(i - 1, i + 2):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 4
                if i < 6 and i > 1:
                    three = True
                    for k in range(i - 2, i + 1):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 4
                if i <= 4:
                    three = True
                    for k in range(i, i + 2):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 3
                if i > 1:
                    three = True
                    for k in range(i - 2, i):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 3
                if i > 0 and i < 6:
                    three = True
                    for k in range(i - 1, i + 1):
                        if board[k][j] is not 1:
                            three = False
                    if three:
                        score += 3
                if i > 0:
                    if board[i - 1][j] == 1:
                        score += 2
                if i < 5:
                    if board[i + 1][j] == 1:
                        score += 2
                if j > 0 and board[i][j - 1] == 1:
                    score += 1
        return score

    def check_kernel(self, submat, kernel, zeroes, target):
        return np.count_nonzero((submat - kernel) == 0) - zeroes == target

    def possible_moves(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)
        return valid_cols

    def apply_move(self, board, col):
        temp = board
        column = board[:, col]
        row = next((r for r in column if r > 0), len(column)) - 1
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
