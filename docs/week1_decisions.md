# Week 1 decisions (fill in at the Day 1 meeting)

Date: ______  Attendees: Sajjad, Jyotindra

## 1. Project horizon
- [ ] 5 months   - [ ] 12 months
- Milestones to put in the concept doc: ____________________

## 2. Tasks (start with three)
| # | Task | Objects (real) | Success threshold | Owner |
|---|------|----------------|-------------------|-------|
| 1 | pick_place | cube 40 mm, bin | cube fully inside bin footprint, below rim | |
| 2 | stack | 3 blocks 50 mm | each block xy within 15 mm of block below, resting on it (5 mm) | |
| 3 | peg_insert | peg 20x20x80 mm, hole block, clearance __ mm | peg bottom >= 15 mm below hole top | |

Thresholds above are the defaults coded in `grammar/predicates.py` and `grammar/programs.py`.
If you change one here, change it there too.

## 3. Ownership
| Area | Owner |
|------|-------|
| WP1 perception (grounding, 6D pose, uncertainty) | |
| Grammar programs and learning productions | |
| Track A execution (MoveIt / cuRobo) | |
| Track B diffusion policy | |
| Sim (Isaac Lab scenes) | |
| Real robot, calibration, demos | |
| Eval harness and ablations | |

## 4. Hardware
- Robot: Franka ___ (Panda / FR3), firmware ___, ROS2 distro ___
- Cameras: RealSense model ___ x __; mounting: eye-in-hand / eye-to-hand / both
- Gripper: Franka Hand / other ___

## 5. Atom schema review (Day 2)
- [ ] Skill list agreed (APPROACH, GRASP, LIFT, TRANSPORT, ALIGN, INSERT, RELEASE, RECOVER)
- [ ] Guard fields (Omega) agreed: clearance, max force, reach, speed, pos/rot tolerance
- [ ] Sigma representation agreed: 6x6 pose covariance per object
- [ ] "If GRASP fails, which atoms are invalidated?" answered for all 3 tasks (see `python -m grammar.show`)

## 6. Open questions / risks
-
