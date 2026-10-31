# LIDC-IDRI CT example

`ct128.nii.gz` is derived from LIDC-IDRI subject **LIDC-IDRI-0001**, hosted by
[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
The source CT was reoriented to RAS, clipped to [−1000, 1000] HU, trilinearly
resampled to a 128³ grid at 3 mm isotropic spacing, and stored as int16. Free-text
NIfTI header fields and extensions are empty. SHA-256 of this processed example:
`fbfaa53aa46f27222b9bdb63cbf6c5b95f71df4181848713998fec99307f9ba2`.

## License and attribution

TCIA lists the LIDC-IDRI image data under the
[Creative Commons Attribution 3.0 Unported license (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/).
This example is an adapted copy of one CT volume. The CT data retains its source
license; the repository's Apache-2.0 license applies to code, not to the CT.

Suggested attribution: “LIDC-IDRI, The Cancer Imaging Archive (TCIA), subject
LIDC-IDRI-0001, https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX, CC BY 3.0.
Example volume reoriented, clipped, resampled to 128³ at 3 mm, and converted to int16.”

Dataset citation: Armato SG III et al. *Data From LIDC-IDRI* [Data set]. The
Cancer Imaging Archive (2015). https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX.

TCIA also requests this acknowledgement in publications and grant applications:

> The authors acknowledge the National Cancer Institute and the Foundation for
> the National Institutes of Health, and their critical role in the creation of
> the free publicly available LIDC/IDRI Database used in this study.

Run the [inference example](../README.md#inference) to preprocess this CT onto
the reconstruction reference grid, render frontal and lateral DRRs, reconstruct
the volume, and compare prediction with reference. The offline [smoke test](../tests/README.md#pipeline-smoke)
remains procedural and does not require this clinical example.
