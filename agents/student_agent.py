"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES)
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
        self.seen = {}

    def step(self, board, reserves, player, opponent):
        LIMIT = STEP_TIME_LIMIT - 0.3
        WIN = 10_000
        NEAR_WIN = 5_000
        DOUBLE_THREAT = 2_000
        REPEAT_PENALTY = 20
        LINE_SCORE = (0, 1, 10, 100, 10_000)
        start = time.perf_counter()

        class Timeout(Exception):
            pass

        lines = [line for line in LINES[WIN_LENGTH]
                 if all(board[r, c, 0] != OBSTACLE for r, c in line)]
        lines_through = {(i, j): sum((i, j) in line for line in lines)
                         for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)}
        killers = []

        def after(b, r, who, move):
            b2, r2 = copy_state(b, r)
            execute_move(b2, r2, who, move)
            return b2, r2

        def can_bring(p, bigger, skip, own, sz, res):
            if any(res[p][s] for s in range(bigger + 1, NUM_SIZES)):
                return True
            if ALLOW_MOVES:
                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        if own[i][j] == p and sz[i][j] > bigger and (i, j) not in skip:
                            return True
            return False

        def add_threat(p, line, own, sz, res, threats):
            gap = next((i, j) for i, j in line if own[i][j] != p)
            if can_bring(p, sz[gap[0]][gap[1]], line, own, sz, res):
                threats[p].append((line, gap))

        def one_cover_stops_all(threat_list, p, own, sz, res):
            shared = set(threat_list[0][0])
            for line, _ in threat_list[1:]:
                shared &= set(line)
            shared -= {gap for _, gap in threat_list}
            return any(sz[i][j] < LARGE and can_bring(p, sz[i][j], (), own, sz, res)
                       for i, j in shared)

        def Eval(b, r, to_move):
            owners, sizes = top_view(b)
            own = owners.tolist()
            sz = sizes.tolist()
            res = r.tolist()
            score = 0
            done = {player: False, opponent: False}
            threats = {player: [], opponent: []}
            for line in lines:
                mine = theirs = 0
                my_large = their_large = False
                for i, j in line:
                    if own[i][j] == player:
                        mine += 1
                        if sz[i][j] == LARGE:
                            my_large = True
                    elif own[i][j] == opponent:
                        theirs += 1
                        if sz[i][j] == LARGE:
                            their_large = True
                if mine and not their_large:
                    score += LINE_SCORE[mine] * 0.5 ** theirs
                    if mine == WIN_LENGTH:
                        done[player] = True
                    elif mine == WIN_LENGTH - 1:
                        add_threat(player, line, own, sz, res, threats)
                if theirs and not my_large:
                    score -= LINE_SCORE[theirs] * 0.5 ** mine
                    if theirs == WIN_LENGTH:
                        done[opponent] = True
                    elif theirs == WIN_LENGTH - 1:
                        add_threat(opponent, line, own, sz, res, threats)

            mover = 3 - to_move
            if done[to_move]:
                return WIN if to_move == player else -WIN
            if done[mover]:
                return WIN if mover == player else -WIN
            if threats[to_move]:
                return NEAR_WIN if to_move == player else -NEAR_WIN
            gaps = {gap for _, gap in threats[mover]}
            if len(gaps) > 1 and not one_cover_stops_all(threats[mover], to_move, own, sz, res):
                score += DOUBLE_THREAT if mover == player else -DOUBLE_THREAT
            return score

        def ordered(b, r, mover, moves, depth):
            other = 3 - mover
            if depth >= 2:
                keyed = []
                for k, mv in enumerate(moves):
                    b2, r2 = after(b, r, mover, mv)
                    keyed.append((Eval(b2, r2, other), k))
                keyed.sort(reverse=(mover == player))
                return [moves[k] for _, k in keyed]
            own = top_view(b)[0].tolist()
            kill = killers[depth]

            def guess(mv):
                g = 10 * mv.size + 2 * lines_through[mv.dest]
                if own[mv.dest[0]][mv.dest[1]] == other:
                    g += 5
                if mv.src is None:
                    g += 1
                if (mv.src, mv.dest, mv.size) in kill:
                    g += 100
                return g
            return sorted(moves, key=guess, reverse=True)

        def remember(mv, depth):
            kill = killers[depth]
            key = (mv.src, mv.dest, mv.size)
            if key not in kill:
                kill.insert(0, key)
                del kill[2:]

        def max_value(b, r, alpha, beta, depth, last_mover):
            if time.perf_counter() - start > LIMIT:
                raise Timeout
            w = check_winner(b, last_mover)
            if w:
                return WIN + depth if w == player else -WIN - depth
            if depth == 0:
                return Eval(b, r, player)
            moves = get_valid_moves(b, r, player)
            if not moves:
                return Eval(b, r, player)
            for mv in ordered(b, r, player, moves, depth):
                b2, r2 = after(b, r, player, mv)
                alpha = max(alpha, min_value(b2, r2, alpha, beta, depth - 1, player))
                if alpha >= beta:
                    remember(mv, depth)
                    return beta
            return alpha

        def min_value(b, r, alpha, beta, depth, last_mover):
            if time.perf_counter() - start > LIMIT:
                raise Timeout
            w = check_winner(b, last_mover)
            if w:
                return WIN + depth if w == player else -WIN - depth
            if depth == 0:
                return Eval(b, r, opponent)
            moves = get_valid_moves(b, r, opponent)
            if not moves:
                return Eval(b, r, opponent)
            for mv in ordered(b, r, opponent, moves, depth):
                b2, r2 = after(b, r, opponent, mv)
                beta = min(beta, max_value(b2, r2, alpha, beta, depth - 1, opponent))
                if alpha >= beta:
                    remember(mv, depth)
                    return alpha
            return beta

        def record_position(move):
            b2, r2 = after(board, reserves, player, move)
            key = b2.tobytes() + r2.tobytes()
            self.seen[key] = self.seen.get(key, 0) + 1
            return move

        root_moves = get_valid_moves(board, reserves, player)
        if len(root_moves) == 1:
            return record_position(root_moves[0])
        penalty = {}
        for mv in root_moves:
            b2, r2 = after(board, reserves, player, mv)
            penalty[mv] = REPEAT_PENALTY * self.seen.get(b2.tobytes() + r2.tobytes(), 0)
        best_move = root_moves[0]
        depth = 1
        while True:
            killers = [[] for _ in range(depth + 1)]
            alpha, cur_best, scores = -float("inf"), None, {}
            try:
                for mv in root_moves:
                    b2, r2 = after(board, reserves, player, mv)
                    pen = penalty[mv]
                    v = min_value(b2, r2, alpha + pen, float("inf"), depth - 1, player) - pen
                    scores[mv] = v
                    if v > alpha:
                        alpha, cur_best = v, mv
            except Timeout:
                if cur_best is not None:
                    best_move = cur_best
                break
            best_move = cur_best
            if depth == 1:
                root_moves.sort(key=lambda m: scores[m], reverse=True)
            else:
                root_moves.remove(best_move)
                root_moves.insert(0, best_move)
            if abs(alpha) >= WIN:
                break
            if time.perf_counter() - start > LIMIT / 2:
                break
            depth += 1
        return record_position(best_move)
