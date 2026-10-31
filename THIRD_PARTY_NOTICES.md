# Third-party notices

GeXCo code is Apache-2.0 (see LICENSE). External libraries remain under their
own licenses. MAISI weights are not included and remain subject to NVIDIA's
MAISI license; CT-RATE-trained GeXCo/CT-Net weights inherit the data's
non-commercial terms. The trained CT-Net evaluation checkpoint is bundled at
`gexco/weights/ctnet.pt` under these non-commercial terms. Its weights are not
covered by the Apache-2.0 code license. Competitor implementations are not vendored.

## CT-RATE qualitative figure

The reference CT content in `assets/qualitative_comparison.png` originates from
CT-RATE (Ibrahim Ethem Hamamci and collaborators; Istanbul Medipol University
Mega Hospital). The two-column figure is adapted from the authors' revised
manuscript Figure 2, retaining only its CT-RATE Reference and GeXCo panels and
their matched detail views. Image values and display window are unchanged.
The code license does not grant rights to the underlying medical images. Consult
CT-RATE's [declared CC BY-NC-SA 4.0 license and access conditions](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## LIDC-IDRI inference example

`examples/ct128.nii.gz` is adapted from subject LIDC-IDRI-0001 in the LIDC-IDRI
collection. TCIA lists the images under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
The dataset citation, requested acknowledgement, and processing description are
in [examples/README.md](examples/README.md).

## Dependencies

The rectified-flow scheduler follows MONAI's Apache-2.0 MAISI implementation.
MAISI AutoencoderKlMaisi and DiffusionModelUNetMaisi are imported from MONAI;
their implementations are not copied here. ResNet-18 is imported from torchvision.
DiffDRR, TotalSegmentator, nnU-Net, frd-score and PyRadiomics are external
runtime dependencies; consult their distribution licenses and weight terms.

SIGReg follows the LeWorldModel implementation:
https://github.com/lucas-maes/le-wm/blob/main/module.py
The following upstream MIT notice applies to the SIGReg-derived implementation:

MIT License

Copyright (c) 2026 Lucas Maes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
