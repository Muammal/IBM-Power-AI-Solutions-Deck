"""tools_ops.py: the IT-operations tools. Read-only diagnostics run freely; anything that changes the system
is registered with requires_approval=True and only runs after a human says yes.

Security rules implemented here (keep them when you add tools):
  * commands are fixed argument lists, never a shell string, never text from the model pasted into a shell
  * service names and HTTP hosts must be on the allow-lists in config.yaml
  * outputs are truncated before they go back to the model
"""
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from agent import tool  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
T = CFG["tools"]
_SERVICE = re.compile(r"^[A-Za-z0-9@._-]{1,80}$")

DIAGNOSTICS = {
    "disk": ["df", "-hT", "-x", "tmpfs", "-x", "devtmpfs"],
    "memory": ["free", "-g"],
    "load": ["uptime"],
    "top_cpu": ["ps", "-eo", "pid,user,comm,%cpu,%mem", "--sort=-%cpu"],
    "top_memory": ["ps", "-eo", "pid,user,comm,%mem,rss", "--sort=-%mem"],
    "failed_units": ["systemctl", "--failed", "--no-pager"],
    "cpu_info": ["lscpu"],
    "lpar_info": ["lparstat", "-i"],          # IBM Power: entitlement, mode, SMT
    "smt": ["ppc64_cpu", "--smt"],
    "network": ["ss", "-tlnp"],
}


def _run(argv, lines=40):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return f"{argv[0]}: not installed on this host"
    text = (out.stdout or out.stderr).strip().splitlines()
    return "\n".join(text[:lines]) + (f"\n... ({len(text) - lines} more lines)" if len(text) > lines else "")


def _check_service(name):
    if not _SERVICE.match(name) or name not in T["allowed_services"]:
        raise ValueError(f"service '{name}' is not on the allow-list: {T['allowed_services']}")


@tool("Run a read-only diagnostic on this Linux LPAR.",
      {"check": {"type": "string", "enum": list(DIAGNOSTICS)}})
def run_diagnostic(check):
    return _run(DIAGNOSTICS[check])


@tool("Status of a systemd service (active, since when, last log lines).",
      {"service": {"type": "string", "description": "systemd unit name"}})
def service_status(service):
    _check_service(service)
    return _run(["systemctl", "status", service, "--no-pager", "-n", "15"])


@tool("Recent log lines of a systemd service.",
      {"service": {"type": "string"}, "lines": {"type": "integer", "minimum": 10, "maximum": 200}, "optional": ["lines"]})
def service_logs(service, lines=60):
    _check_service(service)
    return _run(["journalctl", "-u", service, "-n", str(int(lines)), "--no-pager"], lines=int(lines))


@tool("HTTP GET a health endpoint and report status code and latency.",
      {"url": {"type": "string", "description": "http://127.0.0.1:PORT/health"}})
def check_http(url):
    host = urllib.parse.urlparse(url).hostname
    if host not in T["allowed_http_hosts"]:
        raise ValueError(f"host {host} not allowed: {T['allowed_http_hosts']}")
    t = dt.datetime.now()
    r = httpx.get(url, timeout=10)
    ms = (dt.datetime.now() - t).total_seconds() * 1000
    return f"HTTP {r.status_code} in {ms:.0f} ms\n{r.text[:800]}"


@tool("Search the operations runbooks and knowledge base. Returns matching passages.",
      {"query": {"type": "string"}})
def search_runbooks(query):
    try:  # Solution 02's retrieval endpoint, if it is running
        r = httpx.post(T["rag_search_url"], json={"question": query, "top_k": 4}, timeout=30)
        r.raise_for_status()
        return "\n\n".join(f"[{h['source']}] {h['text']}" for h in r.json()["hits"]) or "no match"
    except httpx.HTTPError:
        pass
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    scored = []
    for p in sorted((ROOT / T["runbooks_dir"]).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        score = sum(text.lower().count(w) for w in words)
        if score:
            scored.append((score, p.name, text))
    return "\n\n".join(f"[{n}]\n{t[:1500]}" for _, n, t in sorted(scored, reverse=True)[:2]) or "no runbook matched"


@tool("Restart a systemd service.", {"service": {"type": "string"}, "reason": {"type": "string"}},
      requires_approval=True)
def restart_service(service, reason):
    _check_service(service)
    return _run(["sudo", "-n", "systemctl", "restart", service]) or f"{service} restarted"


@tool("Open an incident ticket for the operations team.",
      {"summary": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
       "details": {"type": "string"}}, requires_approval=True)
def create_ticket(summary, severity, details):
    path = ROOT / T["tickets_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    ticket = {"id": f"INC-{dt.datetime.now():%Y%m%d%H%M%S}", "summary": summary, "severity": severity,
              "details": details, "created": dt.datetime.now().isoformat(timespec="seconds")}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return f"created {ticket['id']}"   # replace with your ITSM API (ServiceNow, Jira, GLPI ...)
