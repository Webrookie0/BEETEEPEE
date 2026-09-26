# Implementation contract

## Simulation and task

World coordinates are meters, seconds, z-up. Default control frequency is 20 Hz with five 10 ms integration substeps. Action components lie in [-1, 1]. The action is scaled to a desired velocity and vector-clipped to 3 m/s. The stabilized model computes acceleration as `(desired_velocity - velocity) / 0.28`, vector-clipped to 5 m/s², then integrates velocity and position. The gust scenario adds a known external acceleration. The actual velocity may temporarily exceed commanded speed under disturbance; no artificial state clamp is applied.

This approximates a closed-loop flight response. It does not integrate gravity, rotor thrust or angular motion because these are abstracted by stabilization. These limits are benchmark parameters, not identified drone characteristics.

The reference target moves at 0.55 m/s in nominal scenarios. The formation center follows 3 m behind its x coordinate. Slots form a centered grid with 1.25 m spacing; the grid stays world-aligned even when the target turns. Slot assignment is immutable until reset. All agents start close to their slots with target velocity; the initial benchmark does not include takeoff or recovery from random orientations.

Crossing objects are constant-velocity spheres. They start outside the formation envelope, traverse it, then leave the modeled scene. Default crossing geometry shares the target's forward velocity and adds lateral relative velocity; the head-on case adds negative x relative velocity. Injection speed is therefore speed **relative to the target**, not necessarily world speed. Object insertion follows a reproducible path through the formation, not an arbitrary mouse-drawn path. The scenario seed changes spawn jitter and crossing radius/speed. Target motion and scripted event times are not randomized yet.

Drone/drone and drone/object swept distances are evaluated on each straight integration segment, including partially active object intervals. Surface clearance subtracts both radii. Any overlap terminates the team. World limits are ±60 m in x/y and 0–10 m altitude. No collisions are repaired or excluded from metrics.

## Observation tensors

Each agent receives a dictionary:

- `self`, 17 floats: slot error xyz (m), own velocity xyz (m/s), target velocity xyz (m/s), previous velocity command xyz (m/s), persistent slot offset xyz (m), altitude (m), remaining episode fraction.
- `neighbors`, K×8: relative position xyz (m), relative velocity xyz (m/s), neighbor radius (m), validity mask.
- `obstacles`, L×8: the same schema for external objects.

K=L=6 by default. Objects are selected by center distance within 7 m, then padded. Noise is independent Gaussian noise on observed relative positions; velocity/self/target estimates are ideal. Dropout independently masks neighbor and obstacle rows. Missing rows contribute neither to the masked mean nor maximum in the neural set encoders. The normalization constants are fixed inside the policy; raw spaces preserve physical units.

Training and inference receive the same actor observations. The critic shares the local encoder. The privileged simulation state is used for rewards, scenario construction, visualization, collision checking and evaluation only. The classical policy also consumes this same local observation dictionary.

## Controllers and learning

The baseline samples velocity candidates and predicts its own finite-response motion over 2 s. It assumes observed objects and neighbors continue at their observed velocities, penalizes clearance violations and tracks its own moving slot. This is a receding-horizon sampled heuristic with no formal safety guarantee. It has no access to future target or obstacle paths. It may wait, take a detour or fail under difficult configurations. The slot-only controller ignores obstacles and is used to verify that obstacle scenarios are meaningful.

The actor combines a self MLP and separate masked set encoders for neighbors/obstacles. Each set contributes mean and max pooling, preserving row-order invariance. A tanh-squashed Gaussian generates velocity actions. PPO stores unsquashed samples and uses the corresponding transformed log probability for both sampling and optimization. Entropy regularization uses latent Gaussian entropy as a proxy. The environment's vector speed limit remains part of the action mapping.

Optional cloning uses baseline rollouts, followed by fresh on-policy PPO trajectories. GAE bootstraps the value of final observations on time limits, masks terminal failures and stops its recurrence at every reset. Advantage normalization, gradient clipping, ratio clipping and approximate-KL early stopping are implemented. This is a compact single-environment trainer batching agents together, not a distributed Sample Factory reproduction. Actor and critic share an encoder; centralized MAPPO, attention, recurrent memory, failure-state resets and automatic curriculum promotion are not implemented.

Every update saves `latest.pt` atomically. Logs contain step counts, losses, collisions and measured throughput. Environment configuration and algorithm settings accompany checkpoints. Learned evaluation loads the saved configuration by default, with explicit CLI overrides for stress tests. Network weights use local observations exclusively; no baseline or safety filter runs behind the learned actor.

## Measurement semantics

Slot RMSE averages squared 3D slot distance across drones and integrates over simulated time. Centroid RMSE compares swarm centroid with intended formation centroid. Minimum surface clearance covers the whole episode, including failed trials. Command smoothness is RMS finite-difference commanded acceleration, not mechanical energy or motor efficiency.

Recovery starts no earlier than an object's scheduled scene-exit time. At control samples, every drone must meet both its own position and velocity tolerance. The clock starts at the first qualifying sample, resets if any drone fails the condition and requires one full second. Reported recovery time includes that confirmation interval and is quantized to the control period. Thus a swarm already back in formation at scene exit still records approximately 1 s. This is deliberately a scene-exit metric; future work should also measure recovery from actual closest approach and clearance of the last affected drone.

Events that started but did not recover remain `null`; aggregate recovery excludes events from failed episodes. Events scheduled after a premature failure are not encountered and do not enter that denominator, while the episode still counts as failed. Always read event recovery together with episode failure rate. Dynamic scenarios must have enough duration for an event and its recovery window; late manual injection can run into the episode time limit.

## Viewer

The Python server binds only to 127.0.0.1. The browser sends serialized step/reset/inject requests. The browser canvas projects the actual 3D simulation positions; drone arm tilt is decorative and must not be read as simulated attitude. UI controls identify the active policy, model class, units and failure state. Static assets ship inside the Python package; no external runtime script is needed. PNG export captures the current flight view with scenario, seed, controller and time labels.
