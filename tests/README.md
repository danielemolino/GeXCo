# Tests

Run from the repository root after [installation](../README.md#installation):

```bash
python -m pytest -q
python scripts/smoke.py --output-dir outputs/smoke
```

## Regression coverage

The regression suite covers geometry/ray consistency, encoder/fusion interfaces,
checkpoint loading, TIFF/HDF5 normalization, DICOM polarity and physical
calibration, radiograph export/inference consistency,
SIGReg, patient-level splits, resume, seed/cache consistency, HU metrics
and surface distances. Optional reconstruction monitoring is checked for
five-sample selection, unchanged training updates/resume and state restoration.

## Pipeline smoke

The CPU smoke generates a procedural CT128 fixture, projects it, exports a 13-view HDF5 bank,
loads K=8 targets and checks identity volume metrics and the symmetric HD95
penalty when exactly one structure mask is empty. It needs no external checkpoints.
Use a fresh output directory
because completed shards are not overwritten.

The fixture contains no patient data and tests numerical I/O contracts. To use
the bundled LIDC-IDRI CT128 example instead, pass
`--input-ct examples/ct128.nii.gz`. See [example provenance and license](../examples/README.md).

GitHub Actions runs both commands on CPU for Python 3.11 and 3.12. Pretrained generative inference
requires the [model checkpoints](../gexco/weights/README.md).
