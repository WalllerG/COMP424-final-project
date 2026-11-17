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

        def eval(board, reserves, player):
            SIZE_BLUE = (1.0, 2.0, 3.0)
            SIZE_RED = (-1.0, -2.0, -3.0) 
            
            score = 0.0


            value = []
            top = []

            for r in range(4):
                value_row = []
                top_row = []

                for c in range(4):
                    if board[r][c][s] == OBSTACLE:
                        value_row.append(0.0)
                        top_row.append(3)
                        continue
                    v = 0.0
                    owner_on_top = 0
                    for s in (0,1,2):
                        owner = board[r][c][s]
                        if owner == player: 
                            v += SIZE_BLUE[s]
                            owner_on_top = owner
                        elif owner == (3-player):
                            v += SIZE_RED[s]
                            owner_on_top = owner
                    value_row.append(v)
                    top_row.append(owner_on_top)
                value.append(value_row)
                top.append(top_row)



            lines = []

            for r in range(4):
                line = []
                for c in range(4):
                    line.append((value[r][c], top[r][c]))
                lines.append(line)

            for c in range(4):
                line = []
                for r in range(4):
                    line.append((value[r][c], top[r][c]))
                lines.append(line)

            line = []
            for d in range(4):
                line.append((value[d][d], top[d][d]))
            lines.append(line)

            line = []
            for d in range(4):
                line.append((value[d][3-d], top[d][3-d]))
            lines.append(line)


            for line in lines:
                
                blocked = False
                for v, t in line:
                    if t == 3:
                        blocked = True
                    if blocked:
                        continue

                n = 1.5
                m = 1.5
                for v, t in line:
                    if t == player:
                        score += v + n
                        n = n ** 2
                    elif t == (3-player):
                        score += v - m 
                        m = m ** 2

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
                v = min_value(b2, r2, alpha, float("inf"), 3 - 1, player)
                if v > alpha:
                    alpha, cur_best = v, mv
            best_move = cur_best           
        except Timeout:
            pass
        return best_move

        
