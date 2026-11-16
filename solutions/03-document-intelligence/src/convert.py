"""convert.py: PDF / image -> text (Markdown when Docling is available).

Engines:
  docling : layout + tables + reading order, OCR through the Tesseract CLI (ara+eng). Optional extra.
  basic   : pdftotext for born-digital PDFs; pdftoppm + tesseract for scans and images. Always available.

    python src/convert.py data/samples/scan.pdf
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Arabic-Indic and Extended Arabic-Indic (Persian/Kurdish) digits -> ASCII. Iraqi invoices use both.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def normalize_digits(text):
    return text.translate(_DIGITS)


def _run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _tesseract(image, langs):
    # --psm 3: automatic page segmentation. Tesseract handles right-to-left Arabic lines itself.
    return _run(["tesseract", str(image), "stdout", "-l", "+".join(langs), "--psm", "3"])


def convert_basic(path, langs=("ara", "eng"), dpi=300):
    path = pathlib.Path(path)
    if path.suffix.lower() in IMAGE_EXT:
        return _tesseract(path, langs)
    text = _run(["pdftotext", "-layout", str(path), "-"])
    if len(text.strip()) > 50:  # born-digital PDF: it has a text layer
        return text
    with tempfile.TemporaryDirectory() as tmp:  # scanned PDF: render pages, OCR each
        _run(["pdftoppm", "-r", str(dpi), "-png", str(path), f"{tmp}/page"])
        pages = sorted(pathlib.Path(tmp).glob("page*.png"))
        return "\n\n".join(_tesseract(p, langs) for p in pages)


def convert_docling(path, langs=("ara", "eng")):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    opts.ocr_options = TesseractCliOcrOptions(lang=list(langs))
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                                             InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
    return conv.convert(str(path)).document.export_to_markdown()


def convert(path, engine="auto", langs=("ara", "eng"), dpi=300):
    path = pathlib.Path(path)
    if path.suffix.lower() in (".md", ".txt"):  # already text (samples, re-runs)
        return normalize_digits(path.read_text(encoding="utf-8"))
    if engine in ("auto", "docling"):
        try:
            return normalize_digits(convert_docling(path, langs))
        except ImportError:
            if engine == "docling":
                raise
    for tool in ("pdftotext", "tesseract"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not found: run common/scripts/os_prep.sh (poppler-utils, tesseract)")
    return normalize_digits(convert_basic(path, langs, dpi))


if __name__ == "__main__":
    print(convert(sys.argv[1]))
