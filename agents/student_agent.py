"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        LIMIT = STEP_TIME_LIMIT - 0.3
        WIN = 10_000
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        class Timeout(Exception):
            pass

        def after(board, reserves, player, move):
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, move)
            return b, r

        def Eval(b, r):
            owners, sizes = top_view(b)
            SIZE_W = np.array([1.0, 1.5, 2.0])          # small, medium, large
            LINE_SCORE = {0: 0, 1: 1, 2: 10, 3: 100, 4: 10_000}

            cells = [[(i, j) for j in range(4)] for i in range(4)]          # rows
            cells += [[(i, j) for i in range(4)] for j in range(4)]         # columns
            cells += [[(d, d) for d in range(4)], [(d, 3 - d) for d in range(4)]]

            score = 0.0
            for line in cells:
                o = np.array([owners[c] for c in line])
                if (o == OBSTACLE).any():
                    continue
                mine, theirs = (o == player).sum(), (o == opponent).sum()
                if mine and not theirs:
                    weight = np.mean([SIZE_W[sizes[c]] for c in line if owners[c] == player])
                    score += LINE_SCORE[mine] * weight
                elif theirs and not mine:
                    weight = np.mean([SIZE_W[sizes[c]] for c in line if owners[c] == opponent])
                    score -= LINE_SCORE[theirs] * weight

            # large pieces still in hand are valuable
            score += 2.0 * (r[player][LARGE] - r[opponent][LARGE])
            return score

        def max_value(b, r, alpha, beta, depth, last_mover):
            if time.perf_counter() - start > LIMIT:
                raise Timeout
            w = check_winner(b, last_mover)
            if w:
                return WIN + depth if w == player else -WIN - depth
            if depth == 0:
                return Eval(b, r)
            moves = get_valid_moves(b, r, player)
            if not moves:
                return Eval(b, r)
            for mv in moves:
                b2, r2 = after(b, r, player, mv)
                alpha = max(alpha, min_value(b2, r2, alpha, beta, depth - 1, player))
                if alpha >= beta:
                    return beta
            return alpha

        def min_value(b, r, alpha, beta, depth, last_mover):
            if time.perf_counter() - start > LIMIT:
                raise Timeout
            w = check_winner(b, last_mover)
            if w:
                return WIN + depth if w == player else -WIN - depth
            if depth == 0:
                return Eval(b, r)
            moves = get_valid_moves(b, r, opponent)
            if not moves:
                return Eval(b, r)
            for mv in moves:
                b2, r2 = after(b, r, opponent, mv)
                beta = min(beta, max_value(b2, r2, alpha, beta, depth - 1, opponent))
                if alpha >= beta:
                    return alpha
            return beta

        root_moves = get_valid_moves(board, reserves, player)
        best_move = root_moves[0]
        try:
            alpha, cur_best = -float("inf"), root_moves[0]
            for mv in root_moves:
                b2, r2 = after(board, reserves, player, mv)
                v = min_value(b2, r2, alpha, float("inf"), 2, player)
                if v > alpha:
                    alpha, cur_best = v, mv
            best_move = cur_best           
        except Timeout:
            pass
        return best_move

        
