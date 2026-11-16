"""check_repo.py: fast structural checks (runs on any OS, no model downloads).

  * every .py compiles
  * every .sh passes `bash -n` (when bash is available)
  * every solution has the required files
  * every requirements*.txt pins IBM's dated index with /+simple/

    python tools/check_repo.py          # exit code 0 = OK
"""
import pathlib
import py_compile
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = ["README.md", "IMPLEMENTATION.md", "RESOURCES.md", "AGENTS.md", "config.yaml", "requirements.txt",
            "docs/AGENTS.docx"]
SKIP = {"venv", "venv-ft", "venv-docling", "__pycache__", ".git", "out"}


def files(pattern):
    return [p for p in ROOT.rglob(pattern) if not SKIP.intersection(p.relative_to(ROOT).parts)]


def main():
    problems = []
    for p in files("*.py"):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            problems.append(f"compile: {e.msg}")
    bash = shutil.which("bash")
    for p in files("*.sh"):
        if bash and subprocess.run([bash, "-n", str(p)], capture_output=True).returncode != 0:
            problems.append(f"bash -n failed: {p.relative_to(ROOT)}")
    for sol in sorted((ROOT / "solutions").iterdir()):
        for f in REQUIRED:
            if not (sol / f).exists():
                problems.append(f"{sol.name}: missing {f}")
        if not list((sol / "docs").glob(f"{sol.name}.docx")):
            problems.append(f"{sol.name}: missing docs/{sol.name}.docx (run tools/build_docx.py)")
    for req in files("requirements*.txt"):
        text = req.read_text(encoding="utf-8")
        if "wheels.developerfirst.ibm.com/ppc64le/" in text and "/+simple/" not in text:
            problems.append(f"{req.relative_to(ROOT)}: IBM index URL must end with /+simple/")
    print(f"python files: {len(files('*.py'))}, shell scripts: {len(files('*.sh'))}, "
          f"solutions: {len(list((ROOT / 'solutions').iterdir()))}")
    for p in problems:
        print("PROBLEM:", p)
    print("RESULT:", "FAIL" if problems else "PASS")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
