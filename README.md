# Physics-Grounded Action Grammars: Week 1 codebase

Everything you need for the first week of the project. You get:

- the **atom schema** c_i = (skill, objects, pre, effect, Omega, Sigma)
- **three task programs** with automatically derived dependency graphs
- an **executor** that does dependency-local repair (the WP4 idea, working on day one)
- an **evaluation harness** with the metrics from the concept doc
- **hand-eye calibration** collection and solving
- **teleop atom tagging** for demonstrations
- an **Isaac Lab** grasp scene

```
pgag/
  grammar/       predicates.py  atoms.py  dependency.py  programs.py  show.py
  envs/          base.py (Env interface)  mock_env.py  scenes.py
  execution/     executor.py   (Track A skeleton, local repair, WP4 risk hook)
  eval/          run.py  metrics.py  replay.py
  configs/       eval_default.yaml  stress.yaml
  calibration/   collect_samples.py  handeye_solve.py
  teleop/        atom_tagger.py  align_tags.py
  sim/           isaaclab_grasp.py
  tests/         test_grammar.py   (18 tests)
  docs/          week1_decisions.md
```

### What was tested before handing over
| Part | Status |
|------|--------|
| Grammar, executor, mock env, eval harness | Tested: 18/18 unit tests pass; full harness runs |
| Hand-eye solver | Tested on synthetic data for both eye-in-hand and eye-to-hand |
| Atom tagger and aligner | Tested |
| `collect_samples.py` | Needs RealSense and ROS2; syntax-checked only |
| `isaaclab_grasp.py` | Needs Isaac Lab; syntax-checked only |

The two hardware scripts are written against Isaac Lab 2.x and OpenCV 4.7+/5. If an API name differs on your install, see *Troubleshooting* at the end.

---

## Day 0: Setup (15 min, any Linux laptop)

```bash
unzip pgag_week1.zip && cd pgag
python3 -m venv .venv && source .venv/bin/activate
pip install numpy networkx pyyaml pytest opencv-contrib-python
python -m pytest -q          # expect: 18 passed
```

All commands below are run from the `pgag/` folder. Put it under git now:

```bash
git init && git add . && git commit -m "Week 1 scaffold"
```

Then push it to a shared GitHub repo with Jyotindra.

---

## Day 1 (Mon): Decisions meeting

**Goal:** fill in `docs/week1_decisions.md` with Jyotindra (1–2 h). It covers:
- the horizon
- the three tasks and their numeric success thresholds
- who owns what
- the hardware details

Before the meeting, show him the programs and dependency graphs:

```bash
python -m grammar.show                     # all three tasks, prints VALID/INVALID
python -m grammar.show --task stack --dot stack.dot
dot -Tpng stack.dot -o stack.png           # needs: sudo apt install graphviz
```

**What to look at in the output:**
- Edges such as `grasp -> lift : holding(peg)` mean "lift needs what grasp produced". Repair always follows these edges.
- In `stack`, there is an edge `b_release -> c_transport : on(b,a)`. That edge is why knocking block b off later invalidates c's placement too.

**After the meeting:**
- If you change a success threshold, edit the matching predicate default in `grammar/predicates.py` or the parameter in `grammar/programs.py`.
- Re-run `pytest`.

**Done when:** the decisions doc is filled in and committed.

---

## Day 2 (Tue): Atom schema and grammar review

**Goal:** both of you agree the schema is right *before* anyone builds on it.

**1. Read the three files in this order:**
- `grammar/atoms.py`: the `Atom` and `Guards` dataclasses (about 100 lines)
- `grammar/predicates.py`: what every predicate means numerically
- `grammar/programs.py`: the three task programs

**2. Run the evaluation harness on the mock env.**

The mock is a kinematic tabletop with injected failures, not physics. It lets you test grammar logic with no robot or GPU.

```bash
python -m eval.run --episodes 100
```

All three tasks should be at or near 1.00 success. Some local repairs are expected, because grasps fail 8% of the time on purpose.

**3. Run the first real ablation: local repair vs full replanning.**

```bash
python -m eval.run --config configs/stress.yaml --tag local
python -m eval.run --config configs/stress.yaml --no-local-repair --tag full
```

On my run, success was:

| Task | Local repair | Full replan |
|------|--------------|-------------|
| stack | 1.00 | 0.90 |
| peg_insert | 1.00 | 0.88 |

Your numbers will be identical for the same seed.

**4. Inspect a single episode's decisions:**

```bash
python -m eval.replay runs/<run_folder> --failures
python -m eval.replay runs/<run_folder> --task stack --episode 1
```

The `stack` episode 1 trace in the `local` run shows:
1. block b gets knocked
2. a `stale_effect` is detected on `b_release`
3. only b's chain is redone
4. then c continues

**5. Answer the schema question for each task:** "If GRASP fails, what gets invalidated?" The answer is the closure printed in the replay.

**Done when:** the schema is agreed (tick section 5 of the decisions doc) and `pytest` passes after any edits.

---

## Day 3 (Wed): Real robot (calibration and teleop)

### A. Hand–eye calibration (about 2 h)

1. **Print a board.** Generate it:
   ```bash
   python calibration/collect_samples.py --make-board board.png
   ```
   Print it at 100% scale on stiff card. **Measure** the printed square and marker sizes with calipers.

2. **Mount it.**
   - Eye-in-hand (camera on the hand): fix the board flat on the table.
   - Eye-to-hand: mount the board rigidly on the gripper.

3. **Start the robot driver** so TF is published, e.g. the `franka_ros2` bringup. Check it:
   ```bash
   ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand
   ```
   Use `panda_link0 / panda_hand` on a Panda.

4. **Collect 15–25 poses.** Use guiding mode, and vary the rotation by at least ±30° about each axis while keeping the full board in view:
   ```bash
   python calibration/collect_samples.py --setup eye_in_hand \
       --base-frame fr3_link0 --ee-frame fr3_hand \
       --square 0.0400 --marker 0.0300 --out calib/samples.json
   ```
   Press ENTER at each pose, and `q` + ENTER to finish.

5. **Solve:**
   ```bash
   python -m calibration.handeye_solve calib/samples.json --out calib/handeye.json \
       --parent-frame fr3_hand --child-frame camera_color_optical_frame
   ```
   This prints the residual per method, saves the best result, and prints the exact `static_transform_publisher` command to use.

6. **Check the result.**
   - **Target: residual below 2 mm RMS.**
   - Above 3 mm, re-measure the board, add poses with bigger rotations, and check that the board is flat.
   - **Write the residual into the decisions doc.** It's your calibration baseline for objective O5.

### B. Teleop with atom tags (about 1 h)

1. Bring up your teleop pipeline (the OpenTeach → Franka migration) as usual.

2. **Option 1: second terminal.** Run:
   ```bash
   python teleop/atom_tagger.py --task stack --out data/tags/stack.jsonl
   ```
   - Press `n` when you start recording an episode.
   - Press `1`–`8` each time a new skill starts.
   - Press `s`/`f` to end the episode as success or failure.
   - Press `u` to undo a wrong key.

3. **Option 2 (better): inside your recorder.**
   ```python
   from teleop.atom_tagger import AtomTagger
   tagger = AtomTagger("stack", "data/tags/stack.jsonl")
   tagger.start_episode()      # same moment recording starts
   tagger.tag("APPROACH")      # bind to a VR controller button
   tagger.end_episode("success")
   ```

4. **Record 2–3 test demos per task,** then check them:
   ```bash
   python teleop/align_tags.py data/tags/stack.jsonl --fps 30 --check stack \
       --out data/tags/stack_frames.json
   ```
   `--check` flags any episode whose skill sequence differs from the grammar program.

**Done when:** `calib/handeye.json` exists with its residual recorded, and 2–3 tagged demos per task pass `--check`.

> **If Wednesday overruns:** calibration and teleop often do. Let Day 4 slip to the following Monday, but keep Day 5 on schedule.

---

## Day 4 (Thu): Isaac Lab scene

```bash
cd ~/IsaacLab
./isaaclab.sh -p ~/pgag/sim/isaaclab_grasp.py --episodes 10                 # watch it in the GUI
./isaaclab.sh -p ~/pgag/sim/isaaclab_grasp.py --episodes 50 --headless \
    --out ~/pgag/runs/isaaclab_grasp.csv                                     # batch
```

Each episode:
1. randomises the cube position (±5 cm)
2. runs APPROACH → GRASP → LIFT with differential IK
3. checks each effect with the same thresholds as `grammar/predicates.py`
4. appends one CSV row

**Expected result:** success on most episodes.

**If grasps fail systematically:**

| Symptom | Likely fix |
|---------|-----------|
| Fingers close above the cube | Adjust `HAND_TO_TCP` (top of the script) by ±5 mm |
| Cube slips during lift | Raise the friction in `physics_material`, or lengthen the grasp phase |
| Arm doesn't reach | Reduce `--xy_noise` |

**Then match the real camera.** Copy the camera extrinsic from `calib/handeye.json` into your notes for Week 2, when a camera sensor is added to the scene.

**Done when:** at least 10 consecutive successful episodes are logged to the CSV.

---

## Day 5 (Fri): Harness, results, review

1. **Run the full mock evaluation and keep the folders:**
   ```bash
   python -m eval.run --episodes 100 --tag week1_baseline
   python -m eval.run --config configs/stress.yaml --tag week1_local
   python -m eval.run --config configs/stress.yaml --no-local-repair --tag week1_full
   ```

2. **Make an initial results table.** Put the three `summary.json` files plus `runs/isaaclab_grasp.csv` in one table. This is the first row of your eventual ablation table.

3. **30-minute review with Jyotindra.** Cover:
   - what is blocked
   - the Week 2 targets: wire Isaac Lab into `envs/` as an `Env`, start real perception for `observe()`, and record 20+ demos per task

### Friday checklist
- [ ] `docs/week1_decisions.md` filled in and committed
- [ ] Schema agreed, `pytest` passes
- [ ] `calib/handeye.json` with residual below 2 mm (or a known reason)
- [ ] Atom-tagged teleop demos for each task, passing `align_tags.py --check`
- [ ] Isaac Lab grasp scene: 10+ successful episodes in the CSV
- [ ] Mock baseline, plus local-vs-full ablation run folders saved

---

## How the pieces fit (and what changes later)

The `Env` interface (`envs/base.py`) is the seam the whole project is built around. Its methods are:
- `reset(seed)`
- `observe()`: returns what the robot believes, i.e. the perception output
- `true_state()`: ground truth, used only for scoring
- `execute(atom)`

What sits behind each method changes by week:

| | Week 1 | Later |
|---|---|---|
| `observe()` | ground truth (with optional noise) | WP1 perception: Grounded-SAM 2 + 6D pose + covariance, filling `Atom.sigma` |
| `execute(atom)` | scripted skills | Track A: MoveIt / cuRobo per atom. Track B: a diffusion policy conditioned on the atom |
| `GrammarExecutor(risk_fn=...)` | `None` | WP4: short simulator rollouts that predict the probability of a guard violation |

The executor, grammar and eval harness don't change when these are swapped in. So every ablation in the concept doc becomes a config flag plus a run folder.

---

## Troubleshooting

**Isaac Lab import errors.** Isaac Lab 1.x uses `omni.isaac.lab` and `omni.isaac.lab_assets`. Replace `isaaclab` → `omni.isaac.lab` and `isaaclab_assets` → `omni.isaac.lab_assets`.

**Isaac Lab attribute errors.** Newer releases may deprecate `body_state_w` and `root_state_w` in favour of `body_pose_w` and `root_pose_w`. Change those two lines in `ee_pose_base()`.

**Calibration: no `calibrateHandEye`.** Some OpenCV 5 builds lack it. The NumPy Park–Martin solver runs regardless, so nothing breaks; you just get one method instead of six.

**ChArUco not detected.**
- Check that `--dict` matches the printed board.
- Avoid glare.
- Keep the board 0.3–0.8 m from the camera.

**Tagger keys not registering.** The terminal window running the tagger must have focus.
