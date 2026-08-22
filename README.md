# CodeRoach

Research workspace for experiments involving a Backyard Brains RoboRoach.

Hardware (backpack BLE and the iPhone or webcam pose stream) lives in
[`interface/`](interface/README.md). RL policy and teaching loops live in
`rl_control/`.

## Environment

```bash
conda env create -f environment.yml
conda activate axohack
```

See [`interface/README.md`](interface/README.md) before connecting to hardware.
