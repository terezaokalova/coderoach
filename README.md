# CodeRoach

Research workspace for experiments involving a Backyard Brains RoboRoach.

The Bluetooth control code is isolated in [`interface/`](interface/README.md).
That folder contains its own usage guide and contributor instructions so the
hardware-facing code can evolve independently of analysis and experiment code.

## Environment

```bash
conda env create -f environment.yml
conda activate axohack
```

See [`interface/README.md`](interface/README.md) before connecting to hardware.
