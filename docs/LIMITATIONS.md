# Limitations

PickSort-VLA is a small research platform. Its clean package structure and
end-to-end scripts do not make it a production manipulation system.

## Simulation fidelity

- The default five-axis arm is built from simple MuJoCo primitives, not a
  validated dynamics model of a named commercial robot.
- Actuator gains, joint friction, compliance, contact material, camera model,
  and timing are approximate until fitted from identified hardware.
- The optional contact-gated equality constraint stabilizes grasps. Results
  using it are not directly comparable to unassisted physical grasping.
- Collision classification depends on named MuJoCo contacts and cannot capture
  cable strain, torque overload, pinch hazards, or workspace obstacles absent
  from the scene.
- Simulation cycle time is not wall-clock throughput and does not include real
  perception, communication, PLC, conveyor, or recovery latency.

## Task and visual semantics

- `accepted`, `scratch`, and `unknown` are synthetic task labels represented by
  simple visual cues. The project does not implement industrial defect
  inspection or establish inspection accuracy.
- Training scenes have a small number of objects, trays, viewpoints, and
  instruction intents. Performance does not imply robustness to clutter,
  transparent/reflective parts, deformable objects, occlusion, or unseen tools.
- Domain randomization spans configured nuisance ranges; it cannot cover all
  real optical or dynamic variation.
- Detection-coordinate noise is an interface perturbation. It does not replace
  evaluation with a real detector.

## Model capacity and language

- Compact VLA uses a small CNN and byte-level Transformer trained on project
  demonstrations. It is not a pretrained semantic language model and should
  not be expected to understand unrestricted language.
- Paraphrase generalization is limited by the training data and must be measured
  on held-out templates. Byte tokenization alone is not evidence of semantic
  understanding.
- Action queries provide chunked prediction, but the model is not the full ACT
  CVAE and does not model multimodal demonstrations in the same way.
- A small model may exploit tray color, object location, or wording shortcuts.
  Split design and nuisance randomization reduce but do not eliminate this risk.

## Imitation and residual RL

- The waypoint/IK expert has privileged access to object and goal geometry.
  Expert success is not evidence that the RGB policy learned the task.
- Behavior cloning is vulnerable to covariate shift and compounding errors.
- Dense simulation rewards may be exploitable and can encode privileged state.
- Bounded residual SAC can degrade a good base policy, overfit a perturbation
  range, or learn corrections that are unsafe outside simulation. A hard bound
  limits magnitude but does not prove safety.
- Short smoke training is expected to be noisy and may show no ordering among
  methods.

## Reproducibility

- Fixed seeds improve traceability but do not guarantee bitwise-equivalent GPU
  kernels or rendered pixels across OS, driver, MuJoCo, and PyTorch versions.
- Small evaluation sets have wide uncertainty. Always inspect episode counts and
  intervals instead of ranking methods by a few percentage points.
- Checkpoints can depend on dependency versions and preprocessing metadata;
  loading a weight file without its resolved configuration is unsupported.

## Real2Sim2Real and hardware

- No real-robot success rate is established by the initial project release.
- Real-log replay measures consistency with recorded data and simulation, not
  closed-loop transfer or hardware safety.
- Calibration drift, time synchronization, controller dynamics, tool-frame
  errors, network loss, and emergency-stop integration require hardware-specific
  validation.
- ROS 2 defaults to dry-run. A preview message is not a command authorization.
- The software has no safety certification and must not be connected to moving
  machinery without an independent risk assessment, guarded workspace,
  hardware limits, emergency stop, and qualified supervision.

## Appropriate interpretation

The repository can support claims such as “the recorded smoke pipeline ran” or
“method A achieved the stored simulation result on suite B.” It cannot by
itself support claims about factory readiness, defect-inspection accuracy,
human-safe collaboration, or real-robot task success.
