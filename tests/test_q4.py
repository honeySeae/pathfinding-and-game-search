"""题四：原题回归、独立最短路及随机搜索对拍。"""
import math
import random
from dataclasses import FrozenInstanceError, replace

import pytest

from src.grid_utils import all_distances
from src.q4_treasure import (EVALUATORS, GRID, ORDERS, POINTS, GameState, GameStats,
                             TreasureGame, alphabeta, eval_rollout, eval_score,
                             first_step_values, minimax, move, original_game,
                             principal_variation, solve, terrain_cost)


def test_q4_distances():
    game = original_game()
    assert game.distance('A', 'T2') == 4
    assert game.distance('B', 'T3') == 4
    assert game.distance('T1', 'T4') == 12
    for a, source in POINTS.items():
        distances = all_distances(GRID, source, terrain_cost)
        for b, target in POINTS.items():
            assert game.distance(a, b) == distances[target] == game.distance(b, a)


@pytest.mark.parametrize('algorithm', ['minimax', 'alphabeta'])
@pytest.mark.parametrize('H,target,value', [(2, 'T2', -1), (4, 'T3', 3)])
def test_q4_given_example(algorithm, H, target, value):
    assert solve(H, algorithm) == {'best_target': target, 'best_value': value}
    pv = principal_variation(GameState(depth_left=H))
    assert len(pv) == H and pv[-1]['U'] == value and pv[0]['目标'] == target


def test_q4_rollout_regression_and_terminal():
    assert alphabeta(GameState(depth_left=3), evaluate=eval_rollout)[1] == 'T3'
    state = GameState(remaining=frozenset(), score_a=-2, score_b=5)
    assert eval_rollout(state) == eval_score(state) == -7
    for evaluate in EVALUATORS.values():
        assert alphabeta(state, evaluate=evaluate) == (-7, None)


def test_q4_immutable_and_forced_negative_move():
    state = GameState()
    with pytest.raises(FrozenInstanceError):
        state.turn = 'B'
    child = move(state, 'T1')
    assert len(state.remaining) == 4 and len(child.remaining) == 3
    assert child.turn == 'B' and child.pos_a == 'T1'
    assert child.score_a == -1  # 不能因为远处宝藏收益负而跳过回合。


def test_q4_random_search_matches_minimax():
    # 100个独立距离实例：一维正代价路径的距离保证三角不等式，对所有评价/排序对拍。
    rng = random.Random(0)
    for _ in range(100):
        n = rng.randint(2, 6)
        names = ('A', 'B') + tuple(f'T{i+1}' for i in range(n))
        coords = rng.sample(range(30), n+2)
        game = TreasureGame(names, tuple(tuple(abs(x-y) for y in coords) for x in coords),
                            (0, 0) + tuple(rng.randint(5, 12) for _ in range(n)))
        state = GameState(remaining=frozenset(names[2:]), game=game,
                          depth_left=rng.randint(0, n), turn=rng.choice(('A', 'B')),
                          score_a=rng.randint(-5, 5), score_b=rng.randint(-5, 5))
        for evaluate in EVALUATORS.values():
            expected = minimax(state, GameStats(), evaluate)
            for order in ORDERS:
                stats = GameStats()
                assert alphabeta(state, -math.inf, math.inf, stats, order, evaluate) == expected
                assert stats.nodes >= stats.leaves > 0


def test_q4_scores_shift_only_utility_and_full_depth_evaluations_agree():
    state = GameState()
    values = first_step_values(state)
    for evaluate in EVALUATORS.values():
        assert first_step_values(state, evaluate) == values
        assert alphabeta(replace(state, score_a=10, score_b=-3), evaluate=evaluate)[0] == 16


def test_q4_experiment_truth_cache_matches_uncached_minimax():
    # 实验真值缓存仅消除重复状态，独立用无缓存Minimax核验每个首步。
    from experiments.q4_exp_order_and_eval import exact_first_values, random_instance, wilson
    for seed in range(8):
        state, _ = random_instance(4, seed)
        for turn in ('A', 'B'):
            query = replace(state, turn=turn, score_a=3, score_b=-2)
            truth = exact_first_values(query)
            assert truth == {t: minimax(move(query, t), GameStats())[0] for t in query.remaining}
    assert wilson(0, 30)[0] == 0
    assert wilson(30, 30)[1] == pytest.approx(1)
