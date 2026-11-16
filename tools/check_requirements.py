"""check_requirements.py: prove every requirements*.txt installs on RHEL 9 ppc64le / CPython 3.12 using
binary wheels only, WITHOUT a Power machine (pip resolves for a foreign platform).

    python tools/check_requirements.py                     # all files
    python tools/check_requirements.py solutions/05-time-series-forecasting/requirements.txt
Needs pip >= 23 and network access to IBM's index and PyPI. torch wheels are downloaded once (pip cache).
"""
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
# RHEL 9 has glibc 2.34: accept every manylinux tag up to 2.34, plus plain linux_ppc64le (some IBM builds).
PLATFORMS = ["manylinux_2_34_ppc64le", "manylinux_2_28_ppc64le", "manylinux_2_17_ppc64le",
             "manylinux2014_ppc64le", "linux_ppc64le"]


def check(req):
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "report.json"
        cmd = [sys.executable, str(ROOT / "tools" / "_pip_linux_ppc64le.py"), "install", "--dry-run", "--ignore-installed", "--quiet",
               "--only-binary=:all:", "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312",
               "--target", str(pathlib.Path(tmp) / "t"), "--report", str(report), "-r", str(req)]
        for p in PLATFORMS:
            cmd += ["--platform", p]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=req.parent)
        if r.returncode != 0:
            err = [line for line in r.stderr.splitlines() if "ERROR" in line or "No matching" in line]
            return False, err[-3:] or r.stderr.splitlines()[-3:]
        items = json.loads(report.read_text(encoding="utf-8"))["install"]
        ibm = [f"{i['metadata']['name']}=={i['metadata']['version']}" for i in items if "+ppc64le" in i["metadata"]["version"]]
        return True, [f"{len(items)} packages, {len(ibm)} IBM Power builds"]


def main():
    reqs = [pathlib.Path(a).resolve() for a in sys.argv[1:]] or sorted(
        p for p in ROOT.rglob("requirements*.txt") if "venv" not in str(p) and p.name != "requirements-docs.txt" and ("solutions" in p.parts or p.parent.name == "common"))
    failed = 0
    for req in reqs:
        ok, info = check(req)
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {req.relative_to(ROOT)}: {'; '.join(info)}", flush=True)
    print("RESULT:", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
