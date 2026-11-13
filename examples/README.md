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

Run the [inference example](../README.md#inference) to preprocess this CT onto
the reconstruction reference grid, render frontal and lateral DRRs, reconstruct
the volume, and compare prediction with reference.
