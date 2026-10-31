import numpy as np
import pytest

from envs.mock_env import MockTabletopEnv
from envs.scenes import SCENES
from execution.executor import ExecutorConfig, GrammarExecutor
from grammar.dependency import closure, stale_atoms
from grammar.predicates import P
from grammar.programs import TASKS

PERFECT = dict(p_grasp_fail=0, p_slip_transport=0, approach_noise_std=0, align_noise_std=0,
               rot_noise_deg=0, p_disturb=0, obs_noise_std=0, p_collision=0)


@pytest.mark.parametrize("task", list(TASKS))
def test_program_is_symbolically_valid(task):
    assert TASKS[task]().validate() == []


@pytest.mark.parametrize("task", list(TASKS))
def test_dependency_graph_is_acyclic(task):
    import networkx as nx
    assert nx.is_directed_acyclic_graph(TASKS[task]().graph)


def test_closure_of_grasp_excludes_approach():
    prog = TASKS["peg_insert"]()
    c = closure(prog.graph, {"grasp"})
    assert "approach" not in c
    assert {"grasp", "lift", "transport", "align", "insert", "release"} <= c


def test_stack_second_chain_depends_on_first_placement():
    prog = TASKS["stack"]()
    assert prog.graph.has_edge("b_release", "c_transport")
    assert "c_release" in closure(prog.graph, {"b_release"})


@pytest.mark.parametrize("task", list(TASKS))
def test_perfect_env_succeeds_without_repairs(task):
    env = MockTabletopEnv(SCENES[task], PERFECT)
    for seed in range(10):
        env.reset(seed)
        r = GrammarExecutor(TASKS[task](), env).run()
        assert r.success, r.events
        assert r.local_repairs == 0 and r.full_replans == 0
        assert r.steps == len(TASKS[task]().atoms)


@pytest.mark.parametrize("task", list(TASKS))
def test_noisy_env_mostly_recovers(task):
    env = MockTabletopEnv(SCENES[task], dict(p_grasp_fail=0.2, p_slip_transport=0.1))
    wins = 0
    for seed in range(40):
        env.reset(seed)
        wins += GrammarExecutor(TASKS[task](), env).run().success
    assert wins >= 36


def test_stale_detection_after_knock():
    prog = TASKS["stack"]()
    env = MockTabletopEnv(SCENES["stack"], PERFECT)
    env.reset(0)
    for a in prog.atoms[:6]:  # place b on a
        env.execute(a)
    done = {a.id for a in prog.atoms[:6]}
    assert stale_atoms(prog.atoms, done, env.observe()) == set()
    env.knock("b", (0.06, 0.0))
    assert stale_atoms(prog.atoms, done, env.observe()) == {"b_release"}


def test_insert_jam_backs_off_to_align():
    env = MockTabletopEnv(SCENES["peg_insert"], {**PERFECT, "align_noise_std": 0.0015})
    found = False
    for seed in range(60):
        env.reset(seed)
        r = GrammarExecutor(TASKS["peg_insert"](), env).run()
        for e in r.events:
            if (e["event"] == "local_repair" and e.get("reason") == "backoff_after_repeated_failure"
                        and e["atom"] == "insert"):
                assert e["redo"][0] == "align"
                found = True
    assert found


def test_full_replan_baseline_uses_no_local_repair():
    env = MockTabletopEnv(SCENES["stack"], dict(p_grasp_fail=0.2))
    cfg = ExecutorConfig(local_repair=False, max_full_replans=5)
    for seed in range(10):
        env.reset(seed)
        assert GrammarExecutor(TASKS["stack"](), env, cfg).run().local_repairs == 0


def test_predicates_on_and_inserted():
    from grammar.predicates import ObjectState, WorldState
    s = WorldState({"a": ObjectState([0.5, 0, 0.025], [0.05] * 3),
                    "b": ObjectState([0.502, 0, 0.075], [0.05] * 3)}, ee_pos=np.zeros(3))
    assert P("on", "b", "a").holds(s)
    s["b"].pos[0] += 0.03
    assert not P("on", "b", "a").holds(s)
