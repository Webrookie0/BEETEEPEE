# Project context and equations

[Implementation sequence](README.md) · [Configuration](configuration.md) · [All files](file_inventory.md)

This page records project-owned mathematics and runtime behavior. Research influences are linked explicitly. A formula appearing here is not automatically a formula taken from a paper. Source of truth: the checked-out Python/JavaScript code on 29 September 2026, checkpoint format `swarmflow-v4`, curriculum version 4.

## Notation

- $i,j$: drones or an observed object; $N$: number of drones; $t$: control time.
- $p_i,v_i,u_i$: world position (m), velocity (m/s), commanded velocity (m/s).
- $a_i\in[-1,1]^3$: normalized policy action; acceleration is written $\dot v_i$ or $a^{cmd}_i$ to avoid confusion.
- $q,v_q$: target position/velocity; $s_i$: desired slot; $\delta_i$: persistent slot offset.
- $r_i,r_j$: geometric radii when used in a clearance equation; reward is written $r_{i,t}$ or explicitly described.
- $\Delta t=0.05$ s; $h=\Delta t/5=0.01$ s; $T=24$ s default duration.
- $[x]_+=\max(0,x)$; $\operatorname{clip}(x,l,u)$ is componentwise clipping; $\operatorname{limit}_b(x)=x\min(1,b/\max(\|x\|,10^{-12}))$ limits vector norm.
- $\theta,\phi$: actor and critic parameters; $\gamma=0.99$; $\lambda=0.95$; $\hat A$: GAE advantage; $\hat R$: critic return target.
- $f\in[0,1]$: stage progress; $\beta(f)$: demonstration coefficient; $\epsilon$: PPO clip parameter where specified.

## Task and boundaries

The swarm follows a prescribed reference target, yields to spherical objects and recovers the **same assigned target-relative slots**. The target does not respond to the swarm. The standard experiment learns coordination above stabilized flight. It does not model takeoff, motor control, interception or a pursuit-evasion objective. The viewer additionally offers user-steered reference motion (`pilot`).

The runtime is independent NumPy/PyTorch code; it is not a wrapper around QuadSwarm or PyBullet. Full flight-physics validation is [future work](paper_05.md#future-physical-validation).

## Formation geometry

Let $g=\max(spacing,0.75)$ m and center all offsets by subtracting their mean. `formation()` supplies four geometries:

- **Grid:** $c=\lceil\sqrt N\rceil$, raw offsets $g(\lfloor i/c\rfloor,i\bmod c,0)$, then center them.
- **Circle:** radius $R=g/(2\sin(\pi/N))$; offsets $(R\cos(2\pi i/N),R\sin(2\pi i/N),0)$, then center them.
- **Square/triangle:** distribute counts across $m=4$ or $m=3$ edges, with $n_s=\lfloor N/m\rfloor+\mathbf1_{s<N\bmod m}$. Polygon side length is $(\max_s n_s+1)g$. Equally angled vertices are scaled to that side length. Place each edge's drones at interior fractions $(k+1)/(n_s+1)$, avoiding shared corner positions.
- All geometries rescale if the smallest nonzero planar pair distance is below $g$, then recenter. Geometric spacing validation also requires `spacing > 2*radius + 0.2`.

Slots are $s_i=q-(stand\_off,0,0)+\delta_i$. They remain world-aligned and each drone retains its index until reset. Default stand-off is 3 m; spacing 1.25 m; altitude 3 m; drone radius 0.18 m. Reset adds independent uniform jitter in $[-0.07,0.07]$ m per coordinate, sets velocity to target velocity and initializes commands to those velocities.

Code: [`scenarios.formation`](../src/swarmflow/scenarios.py#L57), [`scenarios._polygon_ring`](../src/swarmflow/scenarios.py#L36), [`scenarios._enforce_min_spacing`](../src/swarmflow/scenarios.py#L24), [`env.SwarmEnv.reset`](../src/swarmflow/env.py#L82). Research relation: [Paper 1](paper_01.md#persistent-formation-and-moving-obstacles).

## Scenarios and worlds

Base target motion is $q(t)=(2+v_tt,0,z_0)$ with velocity $(v_t,0,0)$, where $v_t=0$ for hold/pilot and `target_speed=0.55` otherwise. Additional scripted components:

$$q_y^{turn}=1.8\sin(0.22t),\quad v_y^{turn}=0.396\cos(0.22t),$$
$$q_z^{turn}=z_0+0.35\sin(0.3t),\quad v_z^{turn}=0.105\cos(0.3t),$$
$$q_y^{terrain}=1.2\sin(0.18t),\quad v_y^{terrain}=0.216\cos(0.18t).$$

The 11 registered scenarios are hold, follow, crossing, head_on, double_crossing, slalom, dense, turn, gust, pilot and terrain_follow.

For a crossing at $t_0$, horizontal unit direction $d=(\cos\alpha,\sin\alpha,0)$, radius $r_o$ and relative speed $v_r$:

$$D=\max(5,\max_i\|\delta_i\|+r_o+2),\quad c_0=q(t_0)-(stand\_off,0,0),$$
$$p_o(t_0)=c_0-Dd,\quad v_o=v_q(t_0)+v_rd,\quad
p_o(t)=p_o(t_0)+v_o(t-t_0),\quad t_{end}=t_0+2D/v_r.$$

Object speed is relative to the target at spawn; the object then keeps constant world velocity, even if the target later turns. Activity is $t_{start}\le t<t_{end}$.

- Crossing: first time uniform 3–5.5 s, angle $\pi/2\pm0.35$, relative speed 1.4–1.8 m/s, radius 0.6–0.8 m.
- Double crossing: adds a second crossing 6–10 s after the first start, opposite lateral direction $-\pi/2\pm0.35$.
- Head-on: time 3–5.5 s, angle $\pi\pm0.2$, relative speed 1.8–2.2 m/s, radius 0.7–0.9 m.
- Slalom: three radius-0.7 static spheres near $(4,0.3)$, $(7,-0.8)$ and $(10,0.6)$ at formation altitude, with x jitter ±0.3 and y jitter ±0.4 m.
- Dense: 8–12 static spheres along x≈4–14 m with x jitter ±0.25, alternating staggered y offsets with ±0.2 jitter and radii 0.45–0.7 m. This is outside the default learned benchmark.
- Gust: external acceleration $(0,1.2,0.2)$ m/s² during $7\le t<9$ s.
- Pilot: user changes reference velocity/position. Velocity is norm-limited; reference integration clamps x/y to ±55 m and altitude to allowed limits. This is viewer interaction, not default training.

World options are empty, hills, canyon and pillars; they compose with scenarios. Canyon sides are **spheres**, not solid walls: centers at $y_c(x)\pm3.2$, $y_c(x)=1.6\sin(0.22x)+0.55\sin(0.48x+0.7)$, radius 0.55 m. Initial canyon has 24 x samples from 2 to 22 m. Pillars are seven static spheres with x in 3.5–13, y in −2.8–2.8, radii 0.35–0.6 m at formation altitude.

Code: [`scenarios.target_state`](../src/swarmflow/scenarios.py#L141), [`scenarios.crossing`](../src/swarmflow/scenarios.py#L156), [`scenarios.make_obstacles`](../src/swarmflow/scenarios.py#L171), [`scenarios.make_terrain`](../src/swarmflow/scenarios.py#L127).

## Terrain and continuous viewing

For hills, the floor is

$$H(x,y)=\max\left(0,1.15\sin(0.55x)(0.7+0.3\cos(0.3y))
+0.75\sin(0.33x+1)+0.5\sin(0.95x)\cos(0.45y)
+0.28\sin(0.2x-0.28y)+\sum_kA_ke^{-((x-x_k)^2+(y-y_k)^2)/(2w_k^2)}\right).$$

Generated peaks come in pairs on either side of the path: amplitude 1.4–2.5 m, width 1.05–1.7 m, centers separated in x by 2.4–3.6 m, with seeded position jitter. `terrain_follow` forces hills regardless of configured world.

Hills slots raise their z coordinate to at least floor + radius +0.22 m at look-ahead times 0, 1, 2 and 3.2 s along the target's horizontal velocity. If horizontal target speed is below 0.3 m/s, the fallback look-ahead velocity is $(\max(target\_speed,1.2),0)$. This uses privileged terrain in the **goal construction**; it is not a learned terrain perception/planning module. Scripted target altitude does not itself follow the hills. Pilot reference altitude has a similar floor constraint.

Continuous mode disables time truncation. Every three seconds it may inject traffic in traffic scenarios, extend canyon/hills ahead and add pillar spheres. Old finished movers and distant peaks are pruned according to the code. Viewer reset can enlarge obstacle capacity to at least 16 in continuous or terrain worlds. These flights differ from the fixed 24 s, six-obstacle-slot benchmark.

Code: [`scenarios.ground_height`](../src/swarmflow/scenarios.py#L74), [`scenarios.make_peaks`](../src/swarmflow/scenarios.py#L90), [`env.SwarmEnv.slots`](../src/swarmflow/env.py#L64), [`env.SwarmEnv._maybe_stream_world`](../src/swarmflow/env.py#L114).

## Flight dynamics

At each 20 Hz control step:

$$u_i=\operatorname{limit}_{3}(3a_i),\qquad h=\Delta t/substeps.$$

For five substeps:

$$a^{cmd}_i=\operatorname{limit}_{5}((u_i-v_i)/0.28),$$
$$v_i\leftarrow v_i+(a^{cmd}_i+w)h,\qquad p_i\leftarrow p_i+v_i h.$$

Position uses the **updated velocity** (semi-implicit Euler). The gust $w$ is added after limiting commanded acceleration; total acceleration may exceed 5 m/s². Actual velocity is not clamped to 3 m/s after disturbance. These are benchmark parameters, not calibrated vehicle characteristics. No gravity, attitude integration, motor torque, downwash or contact dynamics are simulated. Inputs must provide one finite 3D action per active agent, within component bounds ±1.

Code: [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307), [`geometry.limit_norm`](../src/swarmflow/geometry.py#L4). For physical context see [Paper 5](paper_05.md#stabilized-velocity-interface).

## Collision and world failure

Drone pairs and sphere objects use [swept relative geometry](paper_06.md#swept-collision-geometry) on every integration substep. For partially active objects, intersect the substep with the object's active interval and interpolate drone positions to the clipped interval endpoints. Any surface gap ≤0 marks collision and records the involved drones. Detection is continuous along the straight segment approximation, not exact continuous curved dynamics.

Ground failure tests $z_i\le H(x_i,y_i)+radius$ at substep endpoints. Boundary failure tests $z_i\ge10-radius$ or $|x_i|$/$|y_i|\ge60-radius$. Ground contact counts as collision; a boundary-only failure does not. `culprits` accumulate during substeps. If any drone is grounded, the code's `elif` skips adding simultaneous outside-only culprits in that substep. Integration continues through the control step even after a failure flag is set. No drone overlap or position is repaired.

`min_surface_clearance_m` tracks drone–drone and drone–sphere gaps only; it does not include ground/world-boundary distance. Episode termination and all reward calculations occur after the substeps. Code: [`env.SwarmEnv._clearance`](../src/swarmflow/env.py#L261), [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307).

## Observations

Each drone receives a Gymnasium dictionary with:

- `self`, 17 floats, in this order: $s_i-p_i$ (3), $v_i$ (3), $v_q$ (3), previous commanded velocity (3), persistent offset $\delta_i$ (3), own altitude (1), $\max(0,1-t/T)$ (1).
- `neighbors`, $K\times8$, default $K=6$: relative position (3), relative velocity (3), neighbor radius (1), validity mask (1).
- `obstacles`, $L\times8$, default $L=6$: same schema for real objects and optional virtual ray hits.

Real rows are stably sorted by center distance, filtered to ≤7 m, truncated to capacity, then optionally dropped. Dropout does not backfill with farther real objects. Noise affects relative positions only; self/target/velocity estimates remain ideal. A dropped or padded row is zero. Actor normalization is [fixed inside the encoder](paper_19.md#complete-encoder), not a modification of raw physical-unit spaces.

The baseline and actor see the same local observation dictionary. Only reward, collision checks, scenario construction, metrics and the optional team critic use additional aggregate/state information. Code: [`env.SwarmEnv._observe`](../src/swarmflow/env.py#L240), [`env.SwarmEnv._object_rows`](../src/swarmflow/env.py#L189).

## Ray intersections

Optional rays use unit horizontal directions $d_k=(\cos(2\pi k/R),\sin(2\pi k/R),0)$. For ray origin $o$, sphere center $c$ and radius $r$, set

$$b=2(o-c)\cdot d,\quad c_0=\|o-c\|^2-r^2,\quad D=b^2-4c_0,$$
$$t_\pm=(-b\pm\sqrt{\max(D,0)})/2.$$

Use the entry root if >$10^{-6}$, otherwise the exit root; accept only $D\ge0$ and a positive hit within `ray_range=5` m. Choose the nearest sphere hit per ray, sort hits by range and fill empty obstacle rows only, with relative hit position $td$, zero relative velocity, artificial radius 0.05 m and mask 1. Rays do not sense the analytic hills floor and do not create a full signed-distance field. The default ray count is zero.

Code: [`env.SwarmEnv._ray_hit`](../src/swarmflow/env.py#L205), [`env.SwarmEnv._fill_ray_rows`](../src/swarmflow/env.py#L220). Research motivation: [Paper 2](paper_02.md#bounded-obstacle-sensing).

## Predictive controller

The nominal velocity is $u_{nom}=\operatorname{limit}_{v_{max}}(v_q+1.8(s_i-p_i))$. Candidates include that command; gains 0.35, 0.6, 0.9 and 1.2 on slot error; target velocity; current velocity; zero; and the 26 normalized nonzero vectors in $\{-1,0,1\}^3$ at speeds 0.8, 1.6 and $v_{max}$. There are 86 candidates before possible duplicates, all norm-limited.

Each candidate is held constant during a 2 s prediction sampled every 0.1 s, using the same response/acceleration model but the coarser predictor step and no wind forecast. For relative own displacement $P_k(t)$ and desired displacement $G(t)=s_i-p_i+tv_q$:

$$E_k(t)=\|P_k(t)-G(t)\|^2,$$
$$C_k=1.5\operatorname{mean}_t E_k(t)+2E_k(2)
+0.12\|u_k-u_{nom}\|^2+0.06\|u_k-u_{previous}\|^2.$$

For each group separately (`neighbors`, `obstacles`), compute predicted surface gaps and add

$$180\left(\max_{j,t}[m_s-g_{kj}(t)]_+\right)^2
+20000\left(\max_{j,t}[m_h-g_{kj}(t)]_+\right)^2.$$

Altitude adds $5000\{(\max_t[0.6-z_k(t)]_+)^2+(\max_t[z_k(t)-9.4]_+)^2\}$. That guard uses fixed flat altitude limits, not a terrain-aware feasibility calculation. The minimizing command is divided by maximum speed to form the normalized action. Optional [temperature averaging](paper_29.md#temperature-weighted-commands) replaces the argmin, but defaults off.

Baseline margins are 0.35/0.08 m; current training teacher margins 0.7/0.3 m. This is soft-cost sampled receding-horizon planning, inspired by [relative-motion reasoning](paper_06.md#relative-motion-reasoning) and [margin allowance](paper_28.md#clearance-margin-for-imitation-error), not ORCA or robust MPC. Prediction tests sampled points, while the environment checks swept substeps; the planner can miss a collision between its prediction samples.

The slot-only ablation simply returns $u_{nom}/v_{max}$ without obstacle avoidance. Learned evaluation uses only the neural controller, with no hidden predictive override. Code: [`baseline.PredictiveController.action`](../src/swarmflow/baseline.py#L19), [`baseline.SlotController.__call__`](../src/swarmflow/baseline.py#L69), [`policy.ModelController.__call__`](../src/swarmflow/policy.py#L179).

## Reward

For each drone after the control step, define slot error $e=\|p-s\|$, speed error $b=\|v-v_q\|$, command derivative $d=\|u-u_{prev}\|/\Delta t$, nearest-drone surface gap $g_{nn}$, and active-object surface gaps $g_o$:

$$e_c=\min(e,5),\quad C_{slot}=\begin{cases}e_c^2&e_c\le1\\2e_c-1&e_c>1,\end{cases}$$
$$P=[0.45-g_{nn}]_+^2,\qquad D=\sum_{o\ active}[1-g_o/1]^2_+,$$
$$C=C_{slot}+0.12\min(b,3)^2+6P+3\min(D,2)
+0.005\|u\|^2+0.001\min(d,20)^2,$$
$$r_{i,t}=\Delta t(1-C_i)-100\mathbf1_{failure}\mathbf1_{culprit,i}.$$

The notation $[1-g_o/1]^2_+$ means $\max(0,1-g_o/1)^2$, with gap measured in metres and warning distance 1 m. The collision reward is an event cost, not multiplied by $\Delta t$. Other drones still lose the episode's remaining future rewards when any drone fails. The reward uses all active objects and all drones, not just visible observation rows. Proximity uses post-step distance; the collision flag uses swept distance.

The 5 m cap, Huber-like linear tail, speed/derivative caps, obstacle cap and bounded command make costs bounded. With default radius 0.18 and maximum speed 3, even allowing zero center distance, $C\le9+1.08+6(0.45+0.36)^2+6+0.045+0.4=20.4616$. Hence the conservative bound $\sum\gamma^tr_t\ge-97.308$ for bounded per-step nonterminal rewards at default $\Delta t$ and $\gamma$. This is a sanity check relative to a culpable immediate −100 event penalty, **not** a proof that every agent always prefers every possible survival trajectory: penalties are per culprit, terminal timing matters, and configuration can change the bound. Older assertions that the reward mathematically prevents all intentional failure are too broad.

Source influences: [Batra reward structure](paper_01.md#reward-components-and-collision-attribution) and [Hopper alive reward](references.md#gymnasium-and-pettingzoo). Code: [`env.SwarmEnv._reward`](../src/swarmflow/env.py#L392). PPO usually trains on [scaled rewards](paper_11.md#reward-scaling), while episode returns/logged raw reward retain original units.

## Recovery

Every recovery-enabled object gets an event once its start time is reached. Its original clear time is its scheduled end. Once the measurement time has arrived, the entire team must satisfy

$$\max_i\|p_i-s_i\|\le0.35\text{ m},\qquad
\max_i\|v_i-v_q\|\le0.35\text{ m/s}$$

continuously for at least 1 s of control-time observations, without failure. Leaving either threshold resets the settling timer. Recorded recovery duration includes the full hold:

$$recovery_s=t_{confirmed}-measured\_from_s.$$

For an unrecovered event, another currently active recovery object whose start is before `measured_from_s + 5` extends `measured_from_s` to that object's end and clears the timer. The code applies this only after the original event's end and only to active overlapping events; already recovered events are not revised. Thus v4 measures settling after overlapping disturbances. It does not stop tracking at 5 s: the acceptance check separately tests whether recovery was within 5 s. An episode may end with `recovery_s=null`.

Code: [`env.SwarmEnv._update_recovery`](../src/swarmflow/env.py#L280). Tests check original slot identity, the full hold and overlapping objects. The numerical threshold/overlap rule is project-owned, not taken as a theorem from a research paper.

## Metrics and selection

For actual elapsed time $T_e$, rectangular quadrature at control-step endpoints gives

$$RMSE_{slot}=\sqrt{\frac1{T_e}\sum_t\Delta t\frac1N\sum_i\|p_{i,t}-s_{i,t}\|^2},$$
$$RMSE_{centroid}=\sqrt{\frac1{T_e}\sum_t\Delta t\|\bar p_t-\bar s_t\|^2},$$
$$RMS_{command}=\sqrt{\frac1{T_e}\sum_t\Delta t\frac1N\sum_i(\|u_{i,t}-u_{i,t-1}\|/\Delta t)^2}.$$

Command derivative is smoothness in m/s², not energy or actual motor effort. Failed episodes use their shorter actual duration; a low slot RMSE in a crash is not a successful flight. Mean return is the mean across drones of undiscounted episode rewards. Reports also include collision/failure, minimum swept sphere clearance and each event's timing.

Per-scenario aggregation reports success $1-F/n$, mean episode slot RMSE, sample standard deviation, a normal-approximation 95% interval, minimum clearance and recovery counts/times. It does not pool all episode squared errors before taking a root. See [uncertainty limits](paper_26.md#uncertainty-and-complete-reporting).

Checkpoint selection, equally weighting scenarios, is

$$score=\operatorname{mean}_s(success_s)-0.25\operatorname{mean}_s(\min(RMSE_s,2))
-0.25(1-recovered/events).$$

With no events, recovery fraction is treated as 1. This score counts any completed recovery on successful episodes, including ones slower than 5 s. It is a ranking heuristic; it is **not the acceptance verdict**.

Final learned checks require: zero failures; at least 95% of encountered events recovered within 5 s on successful episodes (if events exist); and mean episode slot RMSE <0.35 m on any hold/follow/turn scenarios present. The implementation uses whole-episode RMSE, though the research plan calls the desired property "steady-state" error. No warm-up portion is removed. Current showcase uses seven scenarios ×20 seeds =140 episodes, not the original proposed 100 episodes per scenario. The baseline suite also tests slalom, so its nominal suite has eight scenarios.

Code: [`env.SwarmEnv.summary`](../src/swarmflow/env.py#L411), [`evaluate.aggregate`](../src/swarmflow/evaluate.py#L37), [`evaluate.selection_score`](../src/swarmflow/evaluate.py#L66), [`evaluate.learning_checks`](../src/swarmflow/evaluate.py#L79), [`suite._acceptance`](../src/swarmflow/suite.py#L68).

## Training execution and storage

The execution sequence is configure stage → initialize/resume weights/optimizer/scaler → noisy-teacher cloning → student DAgger → validate starting policy → create workers → sample policy → advance environments → compute final-state values/GAE → normalize advantages → critic warm-up or PPO+imitation → save latest → periodic selection/snapshots → final evaluation → completion marker → next stage → showcase.

Eight environments are independent instances whose **inference is batched**. Python advances them sequentially in a list comprehension; there are no eight worker processes. PyTorch may use up to four CPU threads. Team step count increases by the number of environments on each tick; agent transitions multiply that count by $N$. Cloning/evaluation steps are outside the PPO counter. The final batch may overshoot a requested budget by fewer than `envs` team steps.

Checkpoint format v4 records model/optimizer states, environment config, cumulative lineage step count, training metadata, critic mode, network spec and reward-scaler moments. `latest.pt` is replaced via a temporary file after each completed update. `best.pt` is saved only for a strictly improved selection score, including the post-cloning starting evaluation. Optional step checkpoints and `finished.json` preserve budget state.

`--resume` restores weights, optimizer and reward moments, but uses fresh seeded episodes; `--continue` restores original stage settings/budget and rebuilds teacher labels when needed. RNG state, current workers and imitation dataset are not serialized, so continuation is not bitwise replay. Inference accepts checkpoint formats v1–v4; training resume requires current v4 and identical environment configuration. Pipeline version (1–4) and checkpoint format version are distinct.

Code: [`train.train`](../src/swarmflow/train.py#L228), [`policy.LearnedController.__init__`](../src/swarmflow/policy.py#L186), [`curriculum.train_stage`](../src/swarmflow/curriculum.py#L75), [`curriculum.build_showcase`](../src/swarmflow/curriculum.py#L178). The exact knobs are in [configuration](configuration.md).

## Viewer and application flow

Python's `ThreadingHTTPServer` binds to localhost port 8765 by default. GET `/api/state` serves simulation state, `/api/model` training/model evidence, `/api/tests` the numbered catalog. POST `/api/reset`, `/api/step`, `/api/inject` and `/api/steer` control the demo. Requests use JSON, same-origin checking, a 16 KiB body limit and a state lock. Step requests allow 1–4 control steps. The frontend normally requests two steps per tick.

The web app uses plain HTML/CSS/JavaScript and Canvas2D, with no npm build, framework or CDN. It provides scenario, geometry, spacing, terrain, seed, drone-count and controller selection; manual traffic injection; pilot keyboard/buttons; orbit and orthographic axis views; trails, telemetry, training curves, validation/test results and PNG export. Projection/rotor tilt are illustrative rendering, not physical attitude estimation. The live slot-error chart is instantaneous RMS across drones, distinct from the integrated episode RMSE.

Default viewer model discovery prefers a showcase, otherwise the highest available stage's best selection score. It ignores root-level legacy `models/best.pt` and `models/latest.pt`. Follow mode checks for changed best models roughly every 5 s; model/test panels refresh periodically. An explicit checkpoint can be viewed without follow mode. Opening a viewer does not train. Training with `pipeline --view` is a separate orchestration path.

Code: [`server.make_server`](../src/swarmflow/server.py#L32), [`curriculum.default_model`](../src/swarmflow/curriculum.py#L307), [`curriculum.training_status`](../src/swarmflow/curriculum.py#L391), [web/app.js](../src/swarmflow/web/app.js), [web/index.html](../src/swarmflow/web/index.html), [web/style.css](../src/swarmflow/web/style.css). These are project engineering features without a research-paper derivation.

## Tests and unimplemented experiments

The numbered suite covers unit checks, baseline, slot-only collisions, 16-drone scaling, held-out baseline, noisy sensing, dense spheres, smoke training, and learned-vs-baseline stages 1–3. Source test files cover seeded behavior, immutable slots, multiple formations, pilot/hills behavior, swept collision, action limits, masks, recovery, reward attribution/bounds, attention invariance, GAE, checkpoint/resume, retained teacher loss and viewer/showcase behavior. The [file inventory](file_inventory.md#tests) lists every test function with a source link.

Research-plan proposals not implemented as complete studies include: automatic mastery-based promotion; free slot reassignment; adaptive formation cost; a matched-budget BC-only/PPO-only/BC+PPO comparison; recurrence; systematic multi-seed uncertainty analysis; full-height dense-wall navigation; full multirotor/downwash/hardware validation. Existing stress-test options do not establish that learned checkpoints pass them.
