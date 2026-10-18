"""Synthetic KYC specimens: identifier formats, date logic, rendering and ground truth."""

from __future__ import annotations

import re

import pypdfium2 as pdfium
import pytest
from tools.synth.kyc import KYC_TYPES, KycGenerator, KycType
from tools.synth.layouts.kyc_layouts import WATERMARK, render
from tools.synth.ledger import LATEST_PERIOD_END
from tools.synth.truth import kyc_expected

ID_FORMATS: dict[KycType, str] = {
    "pan": r"[A-Z]{3}P[A-Z]\d{4}[A-Z]",
    "aadhaar": r"[2-9]\d{11}|X{8}\d{4}",
    "passport": r"[A-Z]\d{7}",
    "driving_licence": r"[A-Z]{2}\d{2}\d{4}\d{7}",
    "voter_id": r"[A-Z]{3}\d{7}",
}


def _text(pdf: bytes) -> str:
    doc = pdfium.PdfDocument(pdf)
    try:
        return "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    finally:
        doc.close()


@pytest.mark.parametrize("kind", KYC_TYPES)
def test_identifier_formats(kind: KycType) -> None:
    for seed in range(25):
        doc = KycGenerator(seed).generate(kind)
        assert doc.document_type == kind
        assert re.fullmatch(ID_FORMATS[kind], doc.id_number), doc.id_number
        assert doc.id_display.replace(" ", "") == doc.id_number


@pytest.mark.parametrize("kind", KYC_TYPES)
def test_dates_are_consistent(kind: KycType) -> None:
    for seed in range(25):
        doc = KycGenerator(seed).generate(kind)
        assert doc.date_of_birth is not None
        age_years = (LATEST_PERIOD_END - doc.date_of_birth).days / 365.25
        assert 18 <= age_years <= 71
        if doc.issue_date:
            assert doc.date_of_birth < doc.issue_date <= LATEST_PERIOD_END
        if doc.issue_date and doc.expiry_date:
            assert doc.issue_date < doc.expiry_date


def test_masked_aadhaar_keeps_last_four_only() -> None:
    docs = [KycGenerator(seed).generate("aadhaar") for seed in range(40)]
    masked = [d for d in docs if d.masked]
    assert masked, "some Aadhaar specimens should be masked"
    for d in masked:
        assert d.id_number.startswith("XXXXXXXX")
        assert d.id_display.startswith("XXXX XXXX ")


@pytest.mark.parametrize("kind", KYC_TYPES)
def test_digital_render_contains_every_value(kind: KycType) -> None:
    doc = KycGenerator(7).generate(kind)
    text = _text(render(doc, "digital"))
    assert doc.id_display in text
    for value in (
        doc.father_or_spouse_name,
        doc.date_of_birth and doc.date_of_birth.strftime("%d/%m/%Y"),
    ):
        if value:
            assert value in text
    surname = doc.name.split()[-1]
    assert surname in text
    assert WATERMARK in text


@pytest.mark.parametrize("kind", KYC_TYPES)
def test_image_render_has_no_text_layer(kind: KycType) -> None:
    pdf = render(KycGenerator(7).generate(kind), "image")
    assert _text(pdf).strip() == ""


@pytest.mark.parametrize("renderer", ["image", "digital"])
def test_kyc_render_is_deterministic(renderer: str) -> None:
    doc = KycGenerator(3).generate("passport")
    assert render(doc, renderer) == render(doc, renderer)  # type: ignore[arg-type]


def test_expected_json_shape() -> None:
    expected = kyc_expected(KycGenerator(1).generate("driving_licence"))
    assert set(expected) == {
        "document_type", "id_number", "name", "father_or_spouse_name", "date_of_birth",
        "gender", "address", "issue_date", "expiry_date",
    }  # fmt: skip
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected["date_of_birth"])
