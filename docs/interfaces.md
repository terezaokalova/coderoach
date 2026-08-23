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
- A stimulator either updates the simulator, pauses without hardware output, or
  sends the action to a connected RoboRoach backpack.
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
commands.

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

The hardware interface enforces the living-animal envelope before every turn:

- Frequency: 1 to 10 Hz
- Pulse width: 1 ms
- Duration: 200 to 300 ms in 5 ms increments
- Gain: 0 to 10 percent
- Minimum interval: 2 seconds between trains
- Rolling limit: at most 30 trains or 9 seconds of train time per 60 seconds

Recent stimulation is recorded in a temporary shared file so separate
processes use the same rolling budget. A rejected action raises RuntimeError
before any turn characteristic is written. Turn commands are never retried
automatically because a failed response does not prove that the stimulus was
not delivered.

### Hardware CLI

Run these from the repository root:

    python interface/roboroach.py scan
    python interface/roboroach.py info
    python interface/roboroach.py session

The left and right commands each send one stimulus:

    python interface/roboroach.py left
    python interface/roboroach.py right

Scan and info do not stimulate. The persistent session accepts left, right,
info, and quit.

## Teaching environment

[rl_control/teach.py](../rl_control/teach.py) defines the feedback loop shared
by simulation, camera-only observation, and live backpack control.

### Stimulator interface

Stimulator is a protocol with one asynchronous operation:

    async def pulse(self, action: StimAction) -> None

SilentStim discards the action. PauseStim in run.py waits for the configured
cooldown and is used for camera-only teaching. BackpackStim in live.py maps an
action to RoboRoach.configure followed by RoboRoach.turn.

### AntiHabituationEnv

AntiHabituationEnv joins a PoseTracker and a Stimulator.

- simulated creates a HabituatingAnimal and SimulatedCamera with no hardware
  stimulation.
- wired accepts a stimulator and optionally a real camera tracker.
- reset reads the first pose and returns an initial MovementState.
- bind_action clamps frequency and pulse width and snaps duration to a 5 ms
  boundary.
- step applies one action, reads the next pose, computes movement, and returns
  the next state, reward, and done flag.

For a turn action, reward is signed turn progress plus 0.3 times speed, with a
0.6 penalty when the animal is still. A wait action receives only the speed
term. The episode ends when the configured still-step limit is reached.

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

    python -m rl_control.run teach --live --source phone --policy bandit

The live path is dispatched to run_live_teach in rl_control/live.py. It opens
one RoboRoach connection, starts the non-stimulating keepalive task, and uses
BackpackStim for each policy action.

### reversal

Reversal is a bounded left-then-right success test:

    python -m rl_control.run reversal --policy bandit --max-steps 30
    python -m rl_control.run reversal --live --source phone --max-steps 30

### goal

Goal runs a heading controller toward one point:

    python -m rl_control.run goal --goal-x 1.0 --goal-y 0.4

Adding --live sends real backpack stimulation, but this older live goal loop
still uses the simulated world for pose rather than the camera.

### path

Path follows either the default three-point route or repeated waypoint values:

    python -m rl_control.run path --policy bandit
    python -m rl_control.run path --waypoint 0.4,0.2 --waypoint 0.8,0.5

Path currently runs in simulation only.

## Mode summary

- Default teach or reversal: simulated pose and silent stimulation.
- teach with --camera: camera pose and silent stimulation.
- teach or reversal with --live: camera pose by default and real backpack
  stimulation.
- teach or reversal with --live --sim-pose: simulated pose and real backpack
  stimulation.
- goal with --live: simulated pose and real backpack stimulation.
- path: simulated pose and silent stimulation.

Live stimulation requires a cooldown of at least two seconds. Keep max-steps
bounded, verify the tracker before enabling live mode, and stop immediately if
the connector is loose, the board becomes hot, or the animal is not recovering.
