# Generated datasets

Generated training archives are intentionally excluded from Git history so the
source repository stays lightweight. The exact generation settings live in
`configs/data/`, and the release provenance manifest is retained as
`generated/vision_six_axis_release_v2.manifest.json`.

To recreate the current six-axis vision dataset, run:

```powershell
& .\.venv\Scripts\python.exe -m smartpick_vla generate-perception `
  --config configs/data/vision_six_axis.yaml `
  --output datasets/generated/vision_six_axis_release_v2.npz
```

The output is synthetic MuJoCo data. It must not be represented as physical
camera, factory inspection, or real-robot demonstration data.
