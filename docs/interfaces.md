# RoboRoach and RL interfaces

This guide explains how the Bluetooth hardware interface in
[interface/roboroach.py](../interface/roboroach.py) connects to the teaching
environment in [rl_control/teach.py](../rl_control/teach.py) and the command-line
entry point in [rl_control/run.py](../rl_control/run.py).

## System flow

The control loop has four replaceable parts:

    pose tracker -> movement state -> policy -> stimulation action
          ^                                      |
          |______________________________________|

- A pose tracker reports position and time.
- AntiHabituationEnv converts consecutive poses into speed, heading, turn rate,
  and still-step count.
- A policy chooses left, right, or wait plus pulse frequency and duration.
- The environment updates its simulated animal when present. A stimulator
  either stays silent, pauses without hardware output, or sends the action
  through the live stimulation gate.
- The resulting pose produces a reward and the policy receives an online
  update.

This is a small online control interface rather than a Gym-compatible API.
Policies implement act and update, while stimulators implement the asynchronous
pulse method.

## Bluetooth hardware interface

The RoboRoach class owns one BleakClient and provides an asynchronous API for
the BLE backpack.

### Connection lifecycle

RoboRoach may receive a discovered BLEDevice or scan by the advertised name
RoboRoach. The preferred lifecycle is an asynchronous context manager:

    async with RoboRoach() as roach:
        settings = await roach.read_settings()
        await roach.turn_left()

Entering the context scans and connects. Exiting always disconnects. Connection
also verifies that the device exposes the RoboRoach service before allowing
commands. It reads and caches the board's current settings so a turn is never
sent against an unknown waveform. BLE reads and writes are serialized with one
asynchronous lock.

### Main operations

- connect and disconnect manage the BLE connection.
- read_settings returns frequency, pulse width, duration, gain, and random mode
  as a StimulationSettings value.
- read_battery_percent reads the standard battery characteristic when
  available.
- configure validates and writes the waveform settings.
- turn, turn_left, and turn_right request exactly one firmware-timed stimulus.
- keep_alive periodically rewrites the unchanged frequency setting. It keeps
  the board awake without stimulating either antenna.

The duration characteristic is encoded in 5 ms units. A left request writes
one to B2B5 and stimulates the right antenna; a right request writes one to
B2B6 and stimulates the left antenna.

### Safety behavior

Configure enforces the living-animal waveform envelope, and turn refuses cached
board settings that exceed its upper safety caps:

- Frequency: 1 to 10 Hz
- Pulse width: 1 ms
- Duration: 200 to 300 ms in 5 ms increments
- Gain: 0 to 10 percent
- Minimum interval: 2 seconds between trains
- Rolling limit: at most 30 trains or 9 seconds of train time per 60 seconds

Recent stimulation is recorded in a temporary shared file so separate
processes use the same rolling budget. Before each turn, the cached board
settings are also checked against the frequency, pulse width, duration, and
gain caps. This catches a backpack that was previously configured by another
client with values outside the safe envelope.

Unknown settings, an out-of-envelope waveform, or an exhausted rolling budget
raise RuntimeError before any turn characteristic is written. Turn commands
are never retried automatically because a failed response does not prove that
the stimulus was not delivered.

### Hardware CLI

Run these from the repository root:

    python interface/roboroach.py scan
    python interface/roboroach.py info
    python interface/roboroach.py session

The left and right commands each send one stimulus:

    python interface/roboroach.py left
    python interface/roboroach.py right

Scan and info do not stimulate. The persistent session accepts left, right,
info, and quit. The left, right, and session commands first configure the board
to the safe defaults of 10 Hz, 1 ms pulse width, 250 ms duration, and 10 percent
gain. Direct users of RoboRoach.turn must call configure first or rely on the
settings cached during connection and checked by turn.

## Teaching environment

[rl_control/teach.py](../rl_control/teach.py) defines the feedback loop shared
by simulation, camera-only observation, and live backpack control.

### Stimulator interface

Stimulator is a protocol with one asynchronous operation:

    async def pulse(self, action: StimAction) -> None

SilentStim discards the action. PauseStim in run.py waits for the configured
cooldown and is used for camera-only teaching. BackpackStim in live.py converts
the action to StimulationSettings and requests it through StimGate.

StimGate serializes requests, applies the refractory period, configures the
requested waveform, calls RoboRoach.turn, and records accepted or rejected
requests in stim_gate.jsonl. Live RL requests use the source name rl and a
unique request identifier. A gate rejection is printed so the policy output is
not mistaken for a pulse that reached the board.

### AntiHabituationEnv

AntiHabituationEnv joins a PoseTracker and a Stimulator.

- simulated creates a HabituatingAnimal with no hardware stimulation. It uses
  SimulatedCamera by default, but may observe a supplied real tracker while
  continuing to step the simulated animal.
- wired accepts a stimulator and optionally a real camera tracker. Without a
  tracker it uses SimulatedCamera.
- reset reads the first pose and returns an initial MovementState.
- bind_action clamps frequency and pulse width and snaps duration to a 5 ms
  boundary.
- step applies one action, reads the next pose, computes movement, and returns
  the next state, reward, and done flag.

For a turn action, reward is signed turn progress plus 0.3 times speed, with a
0.6 penalty when the animal is still. A wait action receives only the speed
term. The episode ends when the configured still-step limit is reached.
Stillness uses per-step displacement despite the still-speed option name. Its
default threshold is 0.02 for simulation and 0.01 when a tracker is supplied.

### Policy interface

A teaching policy supplies:

    def act(self, state: MovementState) -> StimAction

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None

The available command-line choices are:

- static: repeats one frequency and duration as a baseline.
- irregular: cycles through a hand-built set of pulse pairs.
- bandit: learns online from reward while varying pulse parameters.

### Teaching tasks

teach runs two phases: a 180 degree turn in the starting direction and a 180
degree turn in the opposite direction. Each phase begins with ten fixed warmup
steps. Warmup steps receive no policy update. Later steps credit only plausible
turns: the animal must be moving and the observed heading change must not
exceed 35 degrees in one step.

follow_path combines a PathPolicy with the same environment to visit a sequence
of waypoints. reversal is a compatibility wrapper around the shared two-phase
teaching loop.

Every step produces a StepLog containing the movement state, selected action,
reward, turn progress, and warmup status.

## RL command-line interface

Run the CLI as a module from the repository root:

    python -m rl_control.run COMMAND [OPTIONS]

### teach

Simulation is the default and cannot stimulate hardware:

    python -m rl_control.run teach --policy bandit --no-plot
    python -m rl_control.run teach --compare --no-plot

Camera-only mode uses real poses but no backpack output:

    python -m rl_control.run teach --camera --source phone

Live mode connects the camera and backpack:

    PYTHONPATH=src python -m rl_control.run teach --live --source phone \
        --policy bandit --run-dir runs/teach-001

The live path is dispatched to run_live_teach in rl_control/live.py. It opens
one RoboRoach connection, starts the non-stimulating keepalive task, creates one
StimGate, and uses BackpackStim for each policy action. Every command using
--live requires --run-dir for the gate log. Use --gain-percent to change the
gain from its 10 percent default.

### Pose sources

Teach and reversal support two camera implementations:

- The interface tracker is selected with --source and --tracker. Source may be
  phone, a camera index, a JPEG path, or an HTTP or RTSP URL. This remains the
  default camera path.
- The calibrated overhead tracker in src/traj is selected with --pose-source
  camera. It requires PYTHONPATH=src, --run-dir, and every --traj option:
  camera index, HSV bounds, arena calibration, minimum contour area, position
  noise, acceleration noise, and heading cutoff speed.

Use --pose-source sim to retain the default interface tracker behavior. Add
--sim-pose to use simulated pose instead of opening that tracker. The
--still-speed option overrides the displacement threshold for either source.
The live dashboard is skipped for the src/traj tracker because it does not
provide the display methods expected by interface.plot.

### reversal

Reversal is a bounded left-then-right success test:

    python -m rl_control.run reversal --policy bandit --max-steps 30
    PYTHONPATH=src python -m rl_control.run reversal --live --source phone \
        --max-steps 30 --run-dir runs/reversal-001

### goal

Goal runs a heading controller toward one point:

    python -m rl_control.run goal --goal-x 1.0 --goal-y 0.4

Adding --live sends real backpack stimulation, but this older live goal loop
still uses the simulated world for pose rather than the camera. Live goal also
requires PYTHONPATH=src and --run-dir because its stimulation passes through
StimGate:

    PYTHONPATH=src python -m rl_control.run goal --live \
        --run-dir runs/goal-001

### path

Path follows either the default three-point route or repeated waypoint values:

    python -m rl_control.run path --policy bandit
    python -m rl_control.run path --waypoint 0.4,0.2 --waypoint 0.8,0.5

Path currently runs in simulation only.

### spam

Spam is a bounded open-loop hardware check. It sends the requested number of
left pulses followed by the same number of right pulses:

    python -m rl_control.run spam --pulses 6 --cooldown 2

This command configures safe defaults and calls RoboRoach directly rather than
using a policy, pose tracker, or StimGate. The RoboRoach interval and rolling
budget guards still apply.

## Mode summary

- Default teach or reversal: simulated pose and silent stimulation.
- teach with --camera: interface camera pose and silent stimulation.
- teach with --camera --pose-source camera: calibrated src/traj pose and silent
  stimulation.
- teach or reversal with --live: interface camera pose by default and real
  backpack stimulation through StimGate.
- teach or reversal with --live --pose-source camera: calibrated src/traj pose
  and real backpack stimulation through StimGate.
- teach or reversal with --live --sim-pose: simulated pose and real backpack
  stimulation through StimGate.
- goal with --live: simulated pose and real backpack stimulation through
  StimGate.
- path: simulated pose and silent stimulation.
- spam: no pose or policy and direct bounded backpack stimulation.

Live stimulation requires a cooldown of at least two seconds and a run
directory. Keep max-steps bounded, verify the tracker before enabling live
mode, and stop immediately if the connector is loose, the board becomes hot,
or the animal is not recovering.
