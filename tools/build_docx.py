"""build_docx.py: generate the Word (.docx) versions of the Markdown documentation with pandoc.

For each solution folder:
    docs/<solution>.docx   = README.md + IMPLEMENTATION.md + RESOURCES.md   (the solution guide)
    docs/AGENTS.docx       = AGENTS.md                                       (the agent file)
For the repository root:
    docs/IBM-Power-AI-Solutions.docx = README + COMPLIANCE + MODELS + RESOURCES + common/README
    docs/AGENTS.docx                 = AGENTS.md

    pip install -r tools/requirements-docs.txt      # pypandoc_binary bundles pandoc
    python tools/build_docx.py                      # everything
    python tools/build_docx.py solutions/02-enterprise-rag
Arabic blocks written as <div dir="rtl"> become right-to-left paragraphs in Word.
"""
import pathlib
import re
import sys

import pypandoc

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANDOC_ARGS = ["--toc", "--toc-depth=2", "--standalone"]


def title_of(md_text, fallback):
    m = re.search(r"^# (.+)$", md_text, re.M)
    return m.group(1).strip() if m else fallback


def to_docx(parts, out_path, title):
    text = "\n\n\\newpage\n\n".join(p.read_text(encoding="utf-8") for p in parts if p.exists())
    # Links to other repo files do not work inside Word: keep the text, drop relative targets.
    text = re.sub(r"\]\((?!https?://|#)([^)]+)\)", r"] (\1)", text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_text(text, "docx", format="markdown+raw_html+native_divs", outputfile=str(out_path),
                          extra_args=PANDOC_ARGS + ["--metadata", f"title={title}",
                                                    "--metadata", "subtitle=IBM Power AI Solutions for Iraqi Enterprises"])
    print(f"  {out_path.relative_to(ROOT)}")


def build_solution(folder):
    folder = folder.resolve()
    readme = folder / "README.md"
    title = title_of(readme.read_text(encoding="utf-8"), folder.name)
    to_docx([readme, folder / "IMPLEMENTATION.md", folder / "RESOURCES.md"], folder / "docs" / f"{folder.name}.docx", title)
    to_docx([folder / "AGENTS.md"], folder / "docs" / "AGENTS.docx", f"AGENTS — {title}")


def build_root():
    to_docx([ROOT / "README.md", ROOT / "COMPLIANCE.md", ROOT / "MODELS.md", ROOT / "RESOURCES.md",
             ROOT / "common" / "README.md"], ROOT / "docs" / "IBM-Power-AI-Solutions.docx",
            "IBM Power AI Solutions for Iraqi Enterprises")
    to_docx([ROOT / "AGENTS.md"], ROOT / "docs" / "AGENTS.docx", "AGENTS — repository-wide rules")


def main():
    targets = [pathlib.Path(a) for a in sys.argv[1:]]
    if not targets:
        print("root:")
        build_root()
        targets = sorted(p for p in (ROOT / "solutions").iterdir() if (p / "README.md").exists())
    for t in targets:
        print(f"{t.name}:")
        build_solution(t if t.is_absolute() else ROOT / t)


if __name__ == "__main__":
    main()
