"""Choose a parser by the file's magic bytes (never by its extension)."""

from __future__ import annotations

from collections.abc import Sequence

from bhoot.domain.errors import UnsupportedFormatError
from bhoot.ports.parsing import DocumentInput, DocumentParser

_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def sniff_mime(head: bytes) -> str:
    if b"%PDF-" in head[:1024]:  # some PDFs carry junk before the header
        return "application/pdf"
    for signature, mime in _SIGNATURES:
        if head.startswith(signature):
            return mime
    return "application/octet-stream"


class MimeParserSelector:
    def __init__(self, parsers: Sequence[DocumentParser]) -> None:
        self.parsers = sorted(parsers, key=lambda p: -p.priority)

    def sniff_mime(self, head: bytes) -> str:
        return sniff_mime(head)

    def for_input(self, inp: DocumentInput) -> DocumentParser:
        mime = sniff_mime(inp.data[:1024])
        for parser in self.parsers:
            if parser.supports(mime):
                return parser
        supported = "PDF, PNG, JPEG, TIFF"
        raise UnsupportedFormatError(
            f"{inp.filename or 'file'} ({mime}) is not supported; supported formats: {supported}"
        )
