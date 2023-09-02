"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None
pInf = math.inf
nInf = -math.inf


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    x_count = 0
    o_count = 0
    for i in board:
        for j in i:
            if j == X:
                x_count += 1
            if j == O:
                o_count += 1
        
    if terminal(board):
        return "Game Over"
    if x_count > o_count:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return "Game Over"
    
    explored = set()
    for row_idx, i in enumerate(board):
        for col_idx, j in enumerate(i):
            if j == EMPTY:
                explored.add((row_idx, col_idx))
    return explored
    

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise ValueError("Action not valid.")
    
    new_board = deepcopy(board)
    for row_idx, i in enumerate(board):
        for col_idx, j in enumerate(board):
            if action == (row_idx, col_idx):
                new_board[row_idx][col_idx] = player(new_board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """    
    for i in range(3):
        for j in range(3):
            if j == 0:
                if board[i][j] == board[i][j+1] == board[i][j+2] != None:
                    return board[i][j]
            if i == 0:
                if board[i][j] == board[i+1][j] == board[i+2][j] != None:
                    return board[i][j]
            if j == 0 and i == 0:
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] != None:
                    return board[i][j]
            if j == 0 and i == 2:
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] != None:
                    return board[i][j]   
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if not any(EMPTY in row for row in board) or winner(board) in [X, O]:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    if winner(board) == None:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    optimal_move = None

    if terminal(board) == True:
        return None

    def max_value(board):
        if terminal(board):
            return utility(board)
        v = nInf
        for action in actions(board):
            v = max(v, min_value(result(board, action)))
        return v
    
    
    def min_value(board):
        if terminal(board):
            return utility(board)
        v = pInf
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v
    
    if player(board) == X:
        v = nInf
        for action in actions(board):
            new_v = min_value(result(board, action))
            if new_v > v:
                v = new_v
                optimal_move = action
    else:
        v = pInf
        for action in actions(board):
            new_v = max_value(result(board, action))
            if new_v < v:
                v = new_v
                optimal_move = action

    return optimal_move
    
    
