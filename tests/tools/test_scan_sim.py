"""Scan simulator: image-only output, page count, determinism and profile behaviour."""

from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pytest
from tools.synth.layouts import LAYOUTS
from tools.synth.ledger import LedgerGenerator
from tools.synth.scan_sim import PROFILES, ScanSimulator, _perspective_coefficients


@pytest.fixture(scope="module")
def digital_pdf() -> bytes:
    ledger = LedgerGenerator(21).generate(months=2)
    return LAYOUTS["bank_b"]().render(ledger)


def _pages(pdf: bytes) -> list[pdfium.PdfPage]:
    doc = pdfium.PdfDocument(pdf)
    return [doc[i] for i in range(len(doc))]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_scan_has_same_pages_and_no_text_layer(digital_pdf: bytes, profile: str) -> None:
    scan = ScanSimulator(PROFILES[profile], seed=1).simulate(digital_pdf)
    source_pages, scan_pages = _pages(digital_pdf), _pages(scan)
    assert len(scan_pages) == len(source_pages)
    for page in scan_pages:
        assert page.get_textpage().get_text_range().strip() == ""


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_scan_is_deterministic(digital_pdf: bytes, profile: str) -> None:
    sim = ScanSimulator(PROFILES[profile], seed=3)
    assert sim.simulate(digital_pdf) == sim.simulate(digital_pdf)


def test_different_seeds_give_different_scans(digital_pdf: bytes) -> None:
    profile = PROFILES["phone_photo"]
    assert ScanSimulator(profile, 1).simulate(digital_pdf) != ScanSimulator(profile, 2).simulate(
        digital_pdf
    )


def test_fax_profile_is_black_and_white(digital_pdf: bytes) -> None:
    scan = ScanSimulator(PROFILES["fax"], seed=5).simulate(digital_pdf)
    image = _pages(scan)[0].render(scale=1).to_pil().convert("L")
    values = np.asarray(image)
    # JPEG compression smears edges a little; most pixels must still be near black or white
    near_extremes = np.mean((values < 60) | (values > 195))
    assert near_extremes > 0.9


def test_page_size_matches_original(digital_pdf: bytes) -> None:
    scan = ScanSimulator(PROFILES["clean"], seed=1).simulate(digital_pdf)
    w0, h0 = _pages(digital_pdf)[0].get_size()
    w1, h1 = _pages(scan)[0].get_size()
    # rotation expands the canvas slightly; physical size must stay close to A4
    assert abs(w1 - w0) / w0 < 0.03
    assert abs(h1 - h0) / h0 < 0.03


def test_perspective_coefficients_map_corners() -> None:
    out = [(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)]
    src = [(3.0, -2.0), (98.0, 4.0), (104.0, 197.0), (-1.0, 203.0)]
    a, b, c, d, e, f, g, h = _perspective_coefficients(out, src)
    for (x, y), (u, v) in zip(out, src, strict=True):
        den = g * x + h * y + 1
        assert (a * x + b * y + c) / den == pytest.approx(u)
        assert (d * x + e * y + f) / den == pytest.approx(v)
