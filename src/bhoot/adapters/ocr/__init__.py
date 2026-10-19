"""Local OCR adapters (``bhoot[ocr]``): RapidOCR engine, scan preprocessing, text assembly."""

from bhoot.adapters.ocr.assembler import OcrTextAssembler
from bhoot.adapters.ocr.preprocess import OpenCvScanPreprocessor
from bhoot.adapters.ocr.rapidocr_engine import RapidOcrEngine

__all__ = ["OcrTextAssembler", "OpenCvScanPreprocessor", "RapidOcrEngine"]
