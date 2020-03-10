import numpy as np


kernels_3 = [np.diag(np.ones(3)),
             np.flip(np.diag(np.ones(3)), 1),
             np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
             np.rot90(np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))]

kernel_adj = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
kernel_diag = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
kernel_edges = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
kernel_corners = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
kernels_2a = [np.rot90(k, i)
              for k in [kernel_adj, kernel_diag] for i in range(4)]
kernels_2b = [np.rot90(k, i)
              for k in [kernel_edges, kernel_corners] for i in range(2)]
MAX_VAL = len(kernels_3) * 3 + len(kernels_2a) * 2 + len(kernels_2b) * 2


class AIPlayer:

    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        limit = 100
        bestMove = (-MAX_VAL, None)
        pool = []

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.minimax(next_board, move, limit, -
                               MAX_VAL, MAX_VAL, True)
            if val > bestMove[0]:
                bestMove = (val, move)
                pool = []
            elif val == bestMove[0]:
                if pool:
                    pool.append((val, move))
                else:
                    pool = [(val, move), bestMove]

        return np.random.choice([p[1] for p in pool]) if pool and pool[0][0] == bestMove[0] else bestMove[1]

    def minimax(self, board, move, layer, alpha, beta, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or not children:
            return self.evaluation_function(board)

        maxVal = -MAX_VAL
        minVal = MAX_VAL
        for child in children:
            val = self.minimax(self.apply_move(board, child),
                               child, layer - 1, alpha, beta, not maxMove)

            if maxMove and val > maxVal:
                maxVal = val
            elif not maxMove and val < minVal:
                minVal = val

            if val > alpha:
                alpha = val
            elif val < beta:
                beta = val

            if beta <= alpha:
                break
        return maxVal if maxMove else minVal

    def get_expectimax_move(self, board):
        limit = 100
        bestMove = (-MAX_VAL, None)
        start = time()

        for move in self.possible_moves(board):
            next_board = self.apply_move(board, move)
            val = self.expectimax(next_board, move, limit, True)
            if val > bestMove[0]:
                bestMove = (val, move)
                pool = []
            elif val == bestMove[0]:
                if pool:
                    pool.append((val, move))
                else:
                    pool = [(val, move), bestMove]

        return np.random.choice([p[1] for p in pool]) if pool and pool[0][0] == bestMove[0] else bestMove[1]

    def expectimax(self, board, move, layer, maxMove):
        children = self.possible_moves(board)

        if layer == 0 or not children:
            return self.evaluation_function(board)

        maxVal = -MAX_VAL
        expVal = 0
        for child in children:
            val = self.expectimax(self.apply_move(board, child),
                                  child, layer - 1, not maxMove)
            if maxMove and val > maxVal:
                maxVal = val
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
        score = 0
        for i in range(1, board.shape[0] - 1):
            for j in range(1, board.shape[1] - 1):
                submat = board[i-1:i+2, j-1:j+2]
                val = submat[1][1]
                cur_zeroes = np.count_nonzero(submat == 0)
                ones = np.count_nonzero(submat == 1)
                twos = 9 - cur_zeroes - ones
                score += (ones - twos)
                if val:
                    for kernel, size in [(kernels_3, 3), (kernels_2a, 2)]:
                        if self.check_kernel(submat, val * kernel, cur_zeroes, size):
                            score += size if val == 1 else -size
                else:
                    for kernel in kernels_2b:
                        if (self.check_kernel(submat, kernel, cur_zeroes, 2) or
                                self.check_kernel(submat, 2 * kernel, cur_zeroes, 2)):
                            score += 3

        return score

    def check_kernel(self, submat, kernel, zeroes, target):
        return np.count_nonzero((submat - kernel) == 0) - zeroes == target

    def possible_moves(self, board):
        return [i for i in range(board.shape[1]) if 0 in board[:, i:i+1]]

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
