"""DICOM decoding, physical calibration and exported inference inputs."""
import json
import sys

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import tifffile
import torch
import torch.nn.functional as F

from gexco.config import default_config, load_config
from gexco.data.radiographs import preprocess_dicom, resolve_geometry, main
from gexco.reconstruction.inference import load_input_image


def write_dicom(path, pixels, **tags):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.ImagerPixelSpacing = [0.8, 0.8]
    ds.DistanceSourceToDetector = 1800
    ds.DistanceSourceToPatient = 1750
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds


@pytest.fixture
def pixels():
    image = np.zeros((72, 96), dtype=np.uint16)
    image[10:65, 20:80] = np.linspace(500, 4000, 55 * 60).reshape(55, 60)
    return image


def test_polarity_modality_and_voi_do_not_change_paper_input(tmp_path, pixels):
    a, b = tmp_path / 'a.dcm', tmp_path / 'b.dcm'
    write_dicom(a, pixels)
    write_dicom(b, pixels.max() - pixels, PhotometricInterpretation='MONOCHROME1',
                RescaleSlope=2, RescaleIntercept=-500, WindowCenter=1, WindowWidth=1)
    normal, _ = preprocess_dicom(a, 'frontal')
    inverted, decisions = preprocess_dicom(b, 'frontal')
    assert decisions['monochrome1_inverted']
    np.testing.assert_allclose(normal[128], inverted[128], atol=2e-7)
    expected = F.avg_pool2d(torch.from_numpy(normal[512])[None, None], 4)[0, 0].numpy()
    np.testing.assert_array_equal(normal[128], expected)
    assert np.unique(normal[512]).size > 256
    assert normal[128].min() >= 0 and normal[128].max() <= 1
    direct = load_input_image(str(a), 128, input_kind='acquired', view='frontal')
    np.testing.assert_array_equal(direct[0].numpy(), normal[128])


def test_geometry_tiers_and_view_specific_magnification():
    config = load_config(default_config('radiographs'))
    ds = pydicom.Dataset()
    ds.Manufacturer = 'GE MEDICAL SYSTEMS'
    ds.ManufacturerModelName = 'Revolution XQi ADS_28.2'
    ds.Rows = ds.Columns = 2022
    d = resolve_geometry(ds, 'frontal', config)
    assert d['pixel_spacing_tier'].startswith('b_')
    assert d['detector_spacing_row_col_mm'] == [0.19444444] * 2
    assert d['source_detector_imputed'] and d['source_support_imputed']
    assert d['magnification_factor'] == pytest.approx(1800 / 1625)
    ds.ImagerPixelSpacing = [0.2, 0.3]
    d = resolve_geometry(ds, 'lateral', config)
    assert d['pixel_spacing_tier'].startswith('a_')
    assert d['isocenter_spacing_row_col_mm'] == pytest.approx([0.2 / (1800 / 1597), 0.3 / (1800 / 1597)])
    del ds.ImagerPixelSpacing
    ds.ManufacturerModelName = 'unknown'
    d = resolve_geometry(ds, 'frontal', config)
    assert d['pixel_spacing_tier_c']
    assert d['detector_spacing_row_col_mm'] == [0.194444] * 2
    with pytest.raises(ValueError, match='explicitly'):
        resolve_geometry(ds, None, config)
    ds.DistanceSourceToPatient = 100
    with pytest.raises(ValueError, match='midplane'):
        resolve_geometry(ds, 'frontal', config)


def test_degenerate_dicom_is_rejected(tmp_path):
    path = tmp_path / 'flat.dcm'
    write_dicom(path, np.zeros((32, 32), dtype=np.uint16))
    with pytest.raises(ValueError, match='Degenerate'):
        preprocess_dicom(path, 'frontal')


def test_prepare_cli_exports_ready_inference_manifest(tmp_path, monkeypatch, pixels):
    source = tmp_path / 'xray.dcm'
    write_dicom(source, pixels)
    out = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare-xray', '--frontal', str(source), '--lateral', str(source),
                                    '--case-id', 'example', '--output-dir', str(out)])
    main()
    records = json.loads((out / 'manifest.json').read_text())['samples']
    assert len(records) == 1
    assert [r['angle_deg'] for r in records[0]['views']] == [0, 90]
    assert len(list(out.rglob('*.tiff'))) == 4
    for item, view in zip(records[0]['views'], ['frontal', 'lateral']):
        expected = load_input_image(source, 128, input_kind='acquired', view=view)
        actual = load_input_image(item['path'], 128, input_kind='normalized')
        assert torch.equal(actual, expected)
        assert tifffile.imread(item['path']).dtype == np.float32
    log = json.loads((out / 'preprocessing.json').read_text())
    assert all('source_sha256' in d for d in log['views'])
    with pytest.raises(FileExistsError):
        main()
