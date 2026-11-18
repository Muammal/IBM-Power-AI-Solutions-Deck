"""Build docs/SIZING.pdf from the capacity model in tools/sizing_model.py.

    pip install -r tools/requirements-docs.txt
    python tools/build_sizing_pdf.py [--out docs/SIZING.pdf]

All numbers in the PDF are computed by sizing_model.py; edit the planning factors there, not here.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sizing_model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-11-18"
NAVY, TEAL, GREY, LINE = (colors.HexColor("#16324f"), colors.HexColor("#0b6e69"),
                          colors.HexColor("#f2f4f7"), colors.HexColor("#c9d1db"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, leading=19, textColor=NAVY, spaceBefore=4,
                    spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=8,
                    spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12.2, alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.6, leading=9.6, spaceAfter=0)
SMALLB = ParagraphStyle("SB", parent=SMALL, fontName="Helvetica-Bold")
HEAD = ParagraphStyle("HD", parent=SMALL, fontName="Helvetica-Bold", textColor=colors.white)
NOTE = ParagraphStyle("N", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#44505c"))
BUL = ParagraphStyle("BL", parent=BODY, leftIndent=11, bulletIndent=2)
CODE = ParagraphStyle("C", parent=BODY, fontName="Courier", fontSize=7.6, leading=9.6, leftIndent=6,
                      backColor=GREY, borderPadding=4, spaceBefore=2, spaceAfter=7)

W = A4[0] - 32 * mm
SC = M.SCALES
KEYS = [s["key"] for s in SC]


def n(x, d=0):
    """Thousands separators; small numbers keep one decimal when asked."""
    return f"{x:,.{d}f}"


def esc(t):
    return str(t).replace("&", "&amp;")


def P(t, st=SMALL):
    return Paragraph(str(t), st)


def table(rows, widths, head_rows=1, bold_first_col=False, total_row=False, align_right_from=1):
    data = []
    for i, r in enumerate(rows):
        st = HEAD if i < head_rows else (SMALLB if (total_row and i == len(rows) - 1) else SMALL)
        out = []
        for j, c in enumerate(r):
            s = st
            if i >= head_rows and j == 0 and bold_first_col:
                s = SMALLB
            if j >= align_right_from:
                s = ParagraphStyle("R", parent=s, alignment=2)
            out.append(c if hasattr(c, "wrap") else P(c, s))
        data.append(out)
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=head_rows)
    style = [("BACKGROUND", (0, 0), (-1, head_rows - 1), NAVY),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    for i in range(head_rows, len(rows)):
        if (i - head_rows) % 2 == 1:
            style.append(("BACKGROUND", (0, i), (-1, i), GREY))
    if total_row:
        style.append(("LINEABOVE", (0, len(rows) - 1), (-1, len(rows) - 1), 0.9, NAVY))
    t.setStyle(TableStyle(style))
    return t


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def on_page(c, doc):
    if doc.page == 1:
        return
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#5b6773"))
    c.drawString(16 * mm, 10 * mm, f"Sizing guide · IBM Power AI Solutions for Iraqi enterprises · {DATE}")
    c.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc.page}")
    c.setStrokeColor(LINE)
    c.line(16 * mm, A4[1] - 12 * mm, A4[0] - 16 * mm, A4[1] - 12 * mm)
    c.drawString(16 * mm, A4[1] - 10.5 * mm, "Planning estimate: calibrate the factors on your own LPAR before you buy")
    c.restoreState()


# ---------------------------------------------------------------------------------------------------------------
def build(out: Path):
    best = {s["key"]: M.size(s, "best") for s in SC}
    lean = {s["key"]: M.size(s, "lean") for s in SC}
    alone = {k: {s["key"]: M.size_solution(k, s, "best") for s in SC} for k in M.SOLUTIONS}
    alone_l = {k: {s["key"]: M.size_solution(k, s, "lean") for s in SC} for k in M.SOLUTIONS}
    cap = {m: M.llm_capacity(m) for m in M.LLM_NODES}
    story = []

    # ---------------- cover
    story += [Spacer(1, 40 * mm),
              Paragraph("Sizing guide", ParagraphStyle("t0", parent=H1, fontSize=13, textColor=TEAL)),
              Paragraph("IBM Power AI Solutions for Iraqi enterprises",
                        ParagraphStyle("t1", parent=H1, fontSize=24, leading=29)),
              Paragraph("Cores, memory, storage, network and servers for the twelve solutions, in their best-quality "
                        "production configuration, from a 300-person organisation to a country-wide programme "
                        f"for all {M.IRAQ_POPULATION / 1e6:.1f} million residents of Iraq.",
                        ParagraphStyle("t2", parent=BODY, fontSize=11.5, leading=15.5)),
              Spacer(1, 10 * mm)]
    n2 = best["N2"]
    story.append(table([
        ["Scale", "Who", "Total cores", "Servers*"],
        *[[s["name"], s["who"], n(best[s["key"]]["total"]),
           n(best[s["key"]]["central_servers"] + best[s["key"]]["hub_servers"])] for s in SC]],
        [0.25, 0.45, 0.14, 0.16], bold_first_col=True, align_right_from=2))
    story += [Spacer(1, 3 * mm),
              Paragraph("* All twelve solutions on one shared estate, with high availability, disaster recovery, "
                        "test, training and platform capacity. Servers counted as 60-core IBM Power S1124 (Power11) "
                        "equivalents.", NOTE),
              Spacer(1, 18 * mm),
              Paragraph(f"Version 1.0 · {DATE} · generated by <font face='Courier'>tools/build_sizing_pdf.py"
                        "</font> from <font face='Courier'>tools/sizing_model.py</font>", NOTE),
              Paragraph("<b>Status: planning estimate.</b> Power10/Power11 hardware was not available while this was "
                        "written. Throughput factors are derived from Power memory bandwidth and MMA behaviour and from "
                        "runs on x86 with the same software. Section 9 shows how to measure them on your LPAR; the PDF "
                        "rebuilds from your numbers in one command.", NOTE),
              PageBreak()]

    # ---------------- 1. executive summary
    pools_n2 = {r["pool"]: r for r in n2["pools"]}
    vis_share = sum(pools_n2[k]["prod_nodes"] * pools_n2[k]["node"] for k in M.SITE_BOUND if k in pools_n2) / n2["prod"]
    llm_share = sum(r["prod_nodes"] * r["node"] for r in n2["pools"] if r["pool"] in M.LLM_NODES) / n2["prod"]
    cut = 1 - lean["N2"]["total"] / n2["total"]
    rag_s = (M.BEST["02"]["prompt"] - M.BEST["02"]["cached"]) / cap["g8"]["prefill"]
    rag_core_s = rag_s * 16
    s_alone = [alone[k]["S"]["cores"] for k in M.SOLUTIONS]
    tops = M.sensitivity("llm_tops_per_core", [0.125, 0.25, 0.5])
    vis = M.sensitivity("vision_core_s_per_frame", [0.3, 0.6, 1.2])
    up, down = tops[0.125][4] / n2["total"] - 1, 1 - tops[0.5][4] / n2["total"]
    story += [Paragraph("1. Executive summary", H1),
              Paragraph("This guide sizes the twelve solutions in this repository for five scales: three private "
                        "organisations (small, medium, large) and two national programmes (one sector, and the whole "
                        "country). Every figure comes from a capacity model that turns business drivers (users, "
                        "documents, call hours, transactions, cameras, lanes) into CPU cores on IBM Power10/Power11, "
                        "using the <b>best-quality</b> configuration of each solution (section 2), CPU-only, on-premises "
                        "in Iraq.", BODY)]
    rows = [["Scale", "Production cores", "Total cores", "Memory (TB)", "Storage (TB)", "Camera network (Gbps)",
             "Servers (central + camera hubs)", "Lean profile: total cores"]]
    for s in SC:
        b, l_ = best[s["key"]], lean[s["key"]]
        rows.append([s["name"], n(b["prod"]), n(b["total"]), n(b["ram_tb"], 1), n(b["storage_tb"], 1),
                     n(b["net_gbps"], 1), f"{n(b['central_servers'])} + {n(b['hub_servers'])}", n(l_["total"])])
    story += [table(rows, [0.19, 0.11, 0.1, 0.1, 0.1, 0.11, 0.15, 0.14], bold_first_col=True), Spacer(1, 2 * mm),
              Paragraph("Production cores: the load at the busy hour, rounded to whole LPARs. Total cores add N+1 spares, "
                        "disaster recovery, test/staging, model training and platform services (section 5). Storage is "
                        "primary usable capacity before replication and backup.", NOTE),
              Paragraph("Key findings", H2)]
    story += bullets([
        f"<b>Private organisations fit on a handful of servers.</b> A small organisation running all twelve solutions "
        f"needs about {n(best['S']['total'])} cores ({best['S']['central_servers']} servers); a large enterprise about "
        f"{n(best['L']['total'])}. Most organisations start with two to four solutions: section 6 gives the footprint of "
        f"each one deployed alone: {min(s_alone)} to {max(s_alone)} cores for a small organisation.",
        f"<b>Country-wide is a data-centre programme.</b> The whole-country scale needs about {n(n2['total'])} cores "
        f"across two central data centres and {SC[4]['hubs']} governorate camera hubs. It is the sum of very large "
        "workloads: a public assistant for every resident, a knowledge assistant for 500,000 civil servants, "
        "20,000 institutional cameras and 2,500 ANPR lanes.",
        f"<b>Cameras and LLM prompts dominate.</b> At country-wide scale, camera analytics (Solutions 10 and 12) are "
        f"{vis_share:.0%} of production cores and the LLM pools {llm_share:.0%}. On CPUs the cost of an LLM answer is "
        f"mostly <i>reading the prompt</i>: one grounded answer with a 1,800-token prompt costs about "
        f"{rag_core_s:.0f} core-seconds of prompt processing, about {rag_s:.0f} s on a whole 16-core node before the "
        "first word appears. Keep prompts tight and cache repeated answers.",
        f"<b>The lean profile cuts the country-wide estate by {cut:.0%}</b> (section 8): a 3B model and answer cache for "
        "the public assistant, analysing only entrance cameras, 5 fps on slower roads, warm-standby disaster recovery. "
        "Each lever has a quality cost, listed next to it.",
        "<b>The biggest uncertainty is LLM speed per Power core.</b> No public llama.cpp benchmark on Power10/Power11 "
        f"was found. If Power cores reach half the planned speed the country-wide estate grows {up:.0%}; at double "
        f"the speed it shrinks {down:.0%} (section 8). Measure it first (section 9).",
    ])
    story.append(PageBreak())

    # ---------------- 2. best version
    story += [Paragraph("2. What “best version” means", H1),
              Paragraph("Each solution is sized in its highest-quality production form allowed by the repository rules "
                        "(open, fine-tunable, Apache-2.0/MIT models; CPU-only; data stays in Iraq; humans decide "
                        "consequential actions). The models are fine-tuned on the organisation's own data and pass the "
                        "solution's release gate before going live.", BODY)]
    cfg = [
        ["#", "Solution", "Best-version configuration (sized)", "Service level"],
        ["01", esc(M.SOLUTIONS["01"]), "Granite 4.2-8B Q8_0 grounded on the service catalogue (RAG), Arabic/Kurdish/English, "
         "approved-answer cache, thinking off", "first words in ~8–10 s (prompt processing), then ~7 tokens/s streaming"],
        ["02", esc(M.SOLUTIONS["02"]), "Granite 4.2-8B Q8_0 + Granite Embedding 278M multilingual (fine-tuned); thinking on "
         "for the hardest 20% of questions; citations", "first words in ~8–12 s; every answer cites its sources"],
        ["03", esc(M.SOLUTIONS["03"]), "Tesseract ara+eng + Docling, Granite 4.2-3B extraction, Granite 8B re-check of the "
         "least confident 20%", "same-day processing, field accuracy gate"],
        ["04", esc(M.SOLUTIONS["04"]), "Whisper large-v3-turbo (q8_0, beam 5) fine-tuned on Iraqi calls + Granite 8B "
         "summaries and quality scores", "all calls processed within the day"],
        ["05", esc(M.SOLUTIONS["05"]), "Granite TimeSeries TTM-R2 fine-tuned per segment, hourly re-forecast", "hourly"],
        ["06", esc(M.SOLUTIONS["06"]), "XGBoost + Isolation Forest on every transaction; Granite 8B explanations for "
         "flagged cases", "decision ≤ 50 ms; peak = 4× the daily average"],
        ["07", esc(M.SOLUTIONS["07"]), "Granite Embedding 278M classifier, int8 ONNX; Granite 3B teacher for the 5% "
         "least confident", "≤ 1 s per ticket"],
        ["08", esc(M.SOLUTIONS["08"]), "MADLAD-400-3B (bf16) fine-tuned on translation memory + Granite 8B post-edit of 30% "
         "(sensitive documents)", "same day for documents"],
        ["09", esc(M.SOLUTIONS["09"]), "Granite 4.2-8B with thinking for every incident; Granite 4.2-30B for the hardest 30%; "
         "human approval for every change", "triage within minutes"],
        ["10", esc(M.SOLUTIONS["10"]), "RT-DETRv2 trained on site footage, ONNX; entrances/lobbies (40%) at 2 analysed fps, "
         "other cameras at 1 fps; triage agent on every alert", "alert ≤ 3 s after the weapon is visible"],
        ["11", esc(M.SOLUTIONS["11"]), "Granite 8B event extraction from public feeds (no social-media account monitoring); "
         "hourly briefings with thinking", "items enriched within the hour"],
        ["12", esc(M.SOLUTIONS["12"]), "RT-DETRv2 vehicles + plates and CRNN plate OCR trained on Iraqi plates, 10 fps per "
         "lane, PTP/chrony clock gate", "every passing vehicle read; violations reviewed by people"],
    ]
    story += [table(cfg, [0.04, 0.2, 0.5, 0.26], align_right_from=9), Spacer(1, 3 * mm),
              Paragraph("Not sized: OWLv2 zero-shot detection (only a pilot and labelling teacher for Solution 10), "
                        "GPU fine-tuning (optional, see COMPLIANCE.md), network switches, NVR/VMS, SAN arrays, software "
                        "licences and staff.", NOTE),
              PageBreak()]

    # ---------------- 3. method + LLM nodes
    story += [Paragraph("3. Method", H1)]
    story += bullets([
        "<b>Drivers to work.</b> Interactive services are sized at the busy hour (10–15% of a day's requests in one "
        "hour); batch work (documents, calls, translation, news items) over a 20–24 hour window.",
        "<b>LLM pools.</b> Requests from all solutions share llama-server pools per model. For each request: time = "
        "prompt tokens / prefill speed + generated tokens / aggregate decode speed of a node. Nodes = load / target "
        "utilisation (60% interactive, 75% batch), rounded up.",
        "<b>Node speed from first principles.</b> Prefill is compute-bound: 2 × parameters operations per token on "
        "the MMA units. Generation is memory-bound: each step reads all weights once and produces one token for every "
        "busy slot.",
        "<b>Vision.</b> Cores per camera = analysed fps × share of time with motion × core-seconds per frame + "
        "video decoding, at 80% utilisation. Camera pools run in regional hubs next to the cameras.",
        "<b>Resilience.</b> N+1: every pool at every location gets at least one spare LPAR (10% for big pools). The 30B "
        "node fails over to the 8B pool instead. Disaster recovery copies the central pools: none for small/medium "
        "(off-site backups only), 50% warm standby for large, a full active-active second data centre nationally.",
        "<b>Around the production estate:</b> test/staging 10% (at least 16 cores), a training pool for fine-tuning "
        "and evaluation, platform services (VIOS, proxy and single sign-on, monitoring, databases, backup) 5% + 8 "
        "cores.",
    ])
    story.append(Paragraph("LLM node capacity used by the model", H2))
    rows = [["Node (one llama-server per LPAR)", "Cores / RAM", "Weights", "Parallel slots", "Prompt (tok/s)",
             "Generation, all slots (tok/s)", "Per user (tok/s)"]]
    for m, node in M.LLM_NODES.items():
        c = cap[m]
        rows.append([node["name"], f"{node['cores']} / {node['ram_gb']} GB", f"{c['weights_gb']:.1f} GB",
                     str(node["slots"]), n(c["prefill"]), n(c["decode"]), n(c["stream"], 1)])
    rows.append(["MADLAD-400-3B (transformers, bf16)", "16 / 128 GB", "6 GB", "batched", "—",
                 n(M.HW["madlad_tok_s_per_16c"]), "—"])
    story += [table(rows, [0.3, 0.12, 0.09, 0.1, 0.11, 0.17, 0.11]), Spacer(1, 2 * mm),
              Paragraph("Keep every LLM LPAR inside one processor socket (NUMA): generation speed follows memory "
                        "bandwidth. Dedicated cores, SMT4, threads = physical cores (common/README.md C5).", NOTE)]

    # ---------------- 4. planning factors
    hw = M.HW
    pf = [["Planning factor", "Value", "Basis"],
          ["LLM matrix maths per core (int8 MMA, effective)", f"{hw['llm_tops_per_core']} tera-ops/s",
           "assumption: no public llama.cpp result on Power10/11 found; llama-bench pp512"],
          ["Memory bandwidth per core / per LLM LPAR", f"{hw['mem_bw_per_core_gbs']} GB/s / ≤ {n(hw['mem_bw_cap_gbs'])} GB/s",
           "share of one socket's OMI memory bandwidth; calibrate with llama-bench tg128"],
          ["RT-DETRv2-r18 640×640, ONNX Runtime fp32", f"{hw['vision_core_s_per_frame']} core-s / frame",
           "measured 0.86 on x86 laptop (108 ms × 8 threads); benchmark.py"],
          ["Video decode per camera (720p sub-stream / 1080p)", f"{hw['decode_core_720p']} / {hw['decode_core_1080p']} cores",
           "PyAV H.264 software decode"],
          ["Whisper large-v3-turbo q8_0, beam 5", f"{hw['whisper_core_h_per_audio_h']} core-h per audio hour",
           "assumption; time whisper-cli on the LPAR"],
          ["MADLAD-400-3B bf16, 16 cores", f"{n(hw['madlad_tok_s_per_16c'])} tok/s", "assumption; translate a test set"],
          ["Embedding a 400-token chunk (278M, fp32)", f"{hw['embed_core_s_per_chunk']} core-s", "assumption"],
          ["OCR + layout per page (Solution 03)", f"{hw['ocr_core_s_per_page']} core-s", "assumption, 300 dpi scans"],
          ["Fraud scoring per transaction (features + 2 models + API)", f"{hw['fraud_core_ms_per_tx']} core-ms",
           "assumption; x86 API measured 28–40 ms end to end; load-test"],
          ["Ticket classification (int8 ONNX)", f"{hw['ticket_core_s']} core-s", "x86 measured ~0.1 s per ticket (multi-threaded); load-test"],
          ["Utilisation targets: interactive / batch / vision / transactions",
           f"{hw['util_interactive']:.0%} / {hw['util_batch']:.0%} / {hw['util_vision']:.0%} / {hw['util_tx']:.0%}",
           "headroom for latency and bursts"],
          ["Server building block", f"{hw['server_cores']} cores (S1124)",
           f"Power11 S1124: up to 60 cores, 8 TB; E1180: up to {hw['big_server_cores']} cores, 64 TB"]]
    story += [Paragraph("4. Planning factors", H1), table(pf, [0.42, 0.22, 0.36], align_right_from=9), PageBreak()]

    # ---------------- 5. scales and drivers
    drv = [("Staff using internal assistants (02, 09)", "staff", 0), ("Public users of the assistant (01)", "public_users", 0),
           ("  daily active share", "public_dau", "%"), ("Pages to digitise per day (03)", "pages_day", 0),
           ("Knowledge-base pages (02)", "corpus_pages", 0), ("Call-centre agents, 5 talk-hours a day (04)", "agents", 0),
           ("Time series forecast hourly (05)", "series", 0), ("Transactions per day (06)", "tx_day", 0),
           ("Tickets per day (07)", "tickets_day", 0), ("Words translated per day (08)", "words_day", 0),
           ("IT incidents per day (09)", "incidents_day", 0), ("CCTV cameras analysed (10)", "cameras", 0),
           ("Public news/feed items per day (11)", "osint_day", 0), ("ANPR lane cameras (12)", "lanes", 0),
           ("Central data centres", "central_sites", 0), ("Regional camera hubs", "hubs", 0),
           ("Disaster recovery copy of central pools", "dr", "%")]
    rows = [["Driver"] + [f"{s['name']}<br/>({s['key']})" for s in SC]]
    for label, k, f in drv:
        rows.append([label] + [f"{s[k]:.1%}".replace(".0%", "%") if f == "%" else n(s[k]) for s in SC])
    story += [Paragraph("5. Scales and demand drivers", H1),
              Paragraph("The drivers below are planning assumptions for a fully adopted, best-version deployment. "
                        "Replace them with your own figures in <font face='Courier'>SCALES</font> in "
                        "tools/sizing_model.py. Country-wide figures are anchored on the 2024 census "
                        f"({n(M.IRAQ_POPULATION)} residents). Cameras are those of public institutions "
                        "(ministries, universities, hospitals, oil and gas sites), not street surveillance. ANPR lanes are "
                        "highway and city enforcement points nationally, and gates and internal roads (ports, oil "
                        "fields) for private organisations; only the authorities issue fines.", BODY),
              table(rows, [0.3] + [0.14] * 5), Spacer(1, 3 * mm)]

    # estate build-up
    rows = [["Estate build-up (cores)"] + KEYS]
    lines = [("Production (busy hour, whole LPARs)", "prod"), ("+ N+1 spares", None), ("+ disaster recovery", "dr"),
             ("+ test / staging", "nonprod"), ("+ training and evaluation", "training"), ("+ platform services", "platform")]
    for label, k in lines:
        if k is None:
            rows.append([label] + [n(best[x]["ha"] - best[x]["prod"]) for x in KEYS])
        else:
            rows.append([label] + [n(best[x][k]) for x in KEYS])
    rows.append(["Total cores"] + [n(best[x]["total"]) for x in KEYS])
    story += [Paragraph("Estate build-up, best profile", H2),
              table(rows, [0.4] + [0.12] * 5, total_row=True), PageBreak()]

    # ---------------- 6. private organisations: standalone
    story += [Paragraph("6. Per solution", H1),
              Paragraph("<b>Deployed alone</b> (most organisations start here): cores for one solution with its own "
                        "LLM pool and an N+1 spare, without disaster recovery, test or training capacity. The second figure "
                        "is the lean profile. Solutions that share one LLM pool need less than the sum of these.", BODY)]
    rows = [["#", "Solution"] + [f"{s['key']}" for s in SC]]
    for k, name in M.SOLUTIONS.items():
        rows.append([k, esc(name)] + [f"{n(alone[k][x]['cores'])} <font color='#6b7785'>/ {n(alone_l[k][x]['cores'])}</font>"
                                 for x in KEYS])
    story += [table(rows, [0.05, 0.3] + [0.13] * 5, align_right_from=2), Spacer(1, 2 * mm),
              Paragraph("Read: “40 / 24” = 40 cores best profile, 24 cores lean. Minimum LPAR sizes: LLM node 16 "
                        "cores (8 for the 3B, 32 for the 30B), translation and vision 16, services 4.", NOTE)]
    rows = [["#", "Solution"] + KEYS]
    for k, name in M.SOLUTIONS.items():
        rows.append([k, esc(name)] + [n(best[x]["solutions"][k].total_cores()) for x in KEYS])
    rows.append(["", "Sum of production demand"] + [n(sum(d.total_cores() for d in best[x]["solutions"].values()))
                                                     for x in KEYS])
    story += [Paragraph("Share of the shared estate (production demand, cores, before rounding and spares)", H2),
              table(rows, [0.05, 0.3] + [0.13] * 5, align_right_from=2, total_row=True), PageBreak()]

    # ---------------- 7. pools, topology, storage
    rows = [["Pool (LPAR size)"] + KEYS]
    names = []
    for s in SC:
        for r in best[s["key"]]["pools"]:
            if r["pool"] not in names:
                names.append(r["pool"])
    for pname in names:
        cells = []
        for x in KEYS:
            r = next((r for r in best[x]["pools"] if r["pool"] == pname), None)
            cells.append("—" if r is None else
                         f"{n(r['ha_nodes'] + r['dr_nodes'])} × {r['node']}"
                         + (f"<br/><font color='#6b7785'>{r['locations']} hubs</font>" if r["locations"] > 1 else ""))
        node = M.LLM_NODES[pname]["cores"] if pname in M.LLM_NODES else M.POOL_NODE[pname]
        rows.append([f"{M.POOL_NAMES[pname]} ({node} cores)"] + cells)
    story += [Paragraph("7. Deployable estate", H1),
              Paragraph("LPARs per pool including spares and the disaster-recovery copy (count × cores). Camera "
                        "pools are spread over regional hubs with a spare in every hub.", BODY),
              table(rows, [0.34] + [0.132] * 5)]
    topo = [["Scale", "Topology", "Servers (60-core S1124 equivalents)"],
            *[[s["name"],
               {"S": "One server room; nightly off-site backup of data and models",
                "M": "One data centre, two server racks; off-site backup; cameras on the central site",
                "L": "Two data centres (warm standby for 50% of the central pools) + 3 regional camera hubs",
                "N1": "Two active-active data centres + 6 regional camera hubs",
                "N2": "Two active-active national data centres + one camera hub per governorate (19)"}[s["key"]],
               f"{best[s['key']]['central_servers']} central"
               + (f" + {best[s['key']]['hub_servers']} in hubs" if s["hubs"] else "")
               + (f"<br/><font color='#6b7785'>or {best[s['key']]['big_servers']} × E1180 (256 cores) centrally</font>"
                  if s["key"] in ("L", "N1", "N2") else "")] for s in SC]]
    story += [Paragraph("Topology and servers", H2), table(topo, [0.22, 0.48, 0.3], align_right_from=9),
              Spacer(1, 2 * mm),
              Paragraph("Power10 equivalents: S1024 (up to 48 cores) for S1124, E1080 (up to 240 cores) for E1180. Every "
                        "location has at least two servers so a hardware failure never stops a pool (one is enough for a "
                        "small camera hub). Large and national estates should consolidate the LLM pools on high-end "
                        "servers (E1150/E1180) and keep scale-out servers in the camera hubs.", NOTE)]
    store = [["Storage (primary usable TB)"] + KEYS]
    for k, name in M.SOLUTIONS.items():
        store.append([f"{k} {esc(name)}"] + [n(best[x]["solutions"][k].storage_tb, 1) for x in KEYS])
    store.append(["Total incl. model files"] + [n(best[x]["storage_tb"], 1) for x in KEYS])
    story += [Paragraph("Storage", H2), table(store, [0.4] + [0.12] * 5, total_row=True), Spacer(1, 2 * mm),
              Paragraph("Retention from the solutions' config.yaml: audio 90 days, documents 90 days, weapon events 30 "
                        "days (confirmed 365), plate reads 24 hours, violations 365 days, transactions 400 days. Plan "
                        "×2 for replication and a separate backup tier; continuous video stays on your NVR/VMS.",
                        NOTE),
              Paragraph(f"One-off knowledge-base ingestion (Solution 02): about {n(best['N2']['ingest_core_h'])} "
                        f"core-hours for the country-wide corpus (~{best['N2']['ingest_core_h'] / M.SCALES[4]['training_cores']:.0f} "
                        "hours on the training pool); "
                        f"{n(best['L']['ingest_core_h'])} core-hours for a large enterprise.", NOTE),
              PageBreak()]

    # ---------------- 8. sensitivity and levers
    rows = [["Total cores if …", *KEYS]]
    for v, tot in tops.items():
        rows.append([f"LLM matrix maths = {v} tera-ops/s per core" + (" (planned)" if v == 0.25 else "")]
                    + [n(t) for t in tot])
    for v, tot in vis.items():
        rows.append([f"Vision = {v} core-s per frame" + (" (planned)" if v == 0.6 else "")] + [n(t) for t in tot])
    story += [Paragraph("8. Sensitivity and cost levers", H1),
              Paragraph("Each factor halved and doubled, all else at the best profile. These two factors decide most of "
                        "the national bill, so they are the first things to measure.", BODY),
              table(rows, [0.4] + [0.12] * 5)]
    lev = [["Lever (lean profile)", "Saving", "Quality cost"],
           ["Public assistant on Granite 4.2-3B, shorter prompts, 60% answers from the approved-answer cache",
            f"01: {n(alone['01']['N2']['cores'])} → {n(alone_l['01']['N2']['cores'])} cores (N2)",
            "weaker Arabic reasoning on unusual questions; keep 8B for escalations"],
           ["RAG without thinking, 1,200 fewer prompt tokens", f"02: {n(alone['02']['N2']['cores'])} → "
            f"{n(alone_l['02']['N2']['cores'])}", "lower answer quality on multi-step questions"],
           ["Analyse only entrance / high-risk cameras", f"10: {n(alone['10']['N2']['cores'])} → "
            f"{n(alone_l['10']['N2']['cores'])}", "no detection on 60% of cameras"],
           ["5 fps per ANPR lane", f"12: {n(alone['12']['N2']['cores'])} → {n(alone_l['12']['N2']['cores'])}",
            "fewer reads of fast vehicles; fine below ~80 km/h"],
           ["Whisper beam 1", f"04: {n(alone['04']['N2']['cores'])} → {n(alone_l['04']['N2']['cores'])}",
            "slightly higher word error rate"],
           ["Warm-standby instead of active-active DR (national)", f"DR: {n(best['N2']['dr'])} → "
            f"{n(lean['N2']['dr'])}", "longer recovery after losing a data centre"],
           ["No 30B model, no 8B re-check, less post-editing", "09, 03, 08", "more human review work"]]
    story += [Paragraph("Cost levers", H2), table(lev, [0.46, 0.24, 0.3], align_right_from=9), Spacer(1, 2 * mm)]
    story += bullets([
        "<b>Phase it.</b> National programmes rarely start with every user. Size year one at 10–20% adoption and "
        "add servers as the measured busy-hour load grows (the model is linear in the drivers).",
        "<b>Quantise the vision models to int8</b> after checking recall on your own footage; it can cut the camera "
        "pools substantially but must pass the release gates again.",
        "<b>Accelerators</b> (for example IBM's Spyre accelerator for Power11) could change the LLM and vision figures; "
        "screen any accelerator against COMPLIANCE.md and benchmark it before planning on it.",
    ])
    story.append(PageBreak())

    # ---------------- 9. calibrate
    story += [Paragraph("9. Calibrate on your LPAR (one day of work)", H1),
              Paragraph("Run these on a dedicated-core LPAR with the repository installed (common/README.md C1–C6), "
                        "put the results into <font face='Courier'>HW</font> in tools/sizing_model.py and rebuild this "
                        "PDF. Use the same LPAR size you plan to deploy (16 cores for an LLM node).", BODY),
              Paragraph("LLM: prompt and generation speed", H2),
              Paragraph("/opt/llama.cpp/build/bin/llama-bench -m /opt/models/granite-4.2-8b-Q8_0.gguf -t 16 -p 512 -n 128<br/>"
                        "cd /opt/llama.cpp && cmake --build build --target llama-batched-bench   # not built by C4<br/>"
                        "build/bin/llama-batched-bench -m /opt/models/granite-4.2-8b-Q8_0.gguf -t 16 "
                        "-c 32768 -npp 1800 -ntg 250 -npl 1,4,8,16", CODE),
              Paragraph("llm_tops_per_core = pp512 tok/s × 2 × 8.8e9 / 16 / 1e12. mem_bw_per_core_gbs = tg128 "
                        "tok/s × 9.3 GB / 16. Check the batched run against the node table in section 3.", BODY),
              Paragraph("Vision, speech, translation", H2),
              Paragraph("python common/python/vision/benchmark.py --model weapons.onnx --threads 16<br/>"
                        "time /opt/whisper.cpp/build/bin/whisper-cli -m /opt/models/ggml-large-v3-turbo-q8_0.bin "
                        "-t 16 -bs 5 -l ar -f one_hour_of_calls.wav<br/>"
                        "time python solutions/08-translation-ar-ku-en/eval/eval_chrf.py", CODE),
              Paragraph("vision_core_s_per_frame = seconds per frame × 16. whisper_core_h_per_audio_h = wall-clock "
                        "hours × 16 / audio hours. madlad_tok_s_per_16c = generated tokens / wall-clock seconds.", BODY),
              Paragraph("Rebuild", H2),
              Paragraph("pip install -r tools/requirements-docs.txt<br/>python tools/sizing_model.py        "
                        "# print the tables<br/>python tools/build_sizing_pdf.py     # write docs/SIZING.pdf", CODE),
              Paragraph("Then load-test the real service at the planned busy-hour rate (for example with locust "
                        "against the solution API) and confirm the service levels in section 2 at 60% utilisation.",
                        BODY)]

    # ---------------- 10. assumptions and limits
    story += [Paragraph("10. Assumptions, limits and compliance", H1)]
    story += bullets([
        "<b>CPU-only, in Iraq.</b> Current fast GPUs are export-controlled for Iraq (US EAR, Country Group D:1); this "
        "estate needs none. No IBM Cloud region is in Iraq: use IBM Cloud Power Virtual Server only for pilots, "
        "training on non-personal data, or disaster recovery where your regulator allows data to leave the country.",
        "<b>Surveillance limits.</b> Solutions 10–12 are sized without face recognition, person tracking, profiling "
        "or plate search, and with a person reviewing every alert and violation. Adding those capabilities is outside "
        "this repository's rules.",
        "<b>Not included:</b> camera and NVR/VMS hardware, network switching and WAN links (camera hubs need the uplink "
        "in the summary table plus margin), SAN/backup systems, power and cooling, software licences, support "
        "contracts and staff.",
        "<b>Per-core figures</b> assume dedicated Power10/Power11 cores in SMT4 with the IBM ppc64le wheel stack pinned in "
        "this repository. Shared-processor pools and SMT8 change results; measure before relying on them.",
        "<b>Business drivers</b> (adoption rates, busy-hour shares, cameras, lanes) are planning assumptions, not survey "
        "data. They are the inputs to replace first with your own figures.",
    ])
    story += [Paragraph("Sources", H2)]
    story += bullets([
        "Iraq 2024 census, final results, Ministry of Planning, 24 February 2025: 46,118,793 residents "
        "(reported by NBC News and PUKmedia).",
        "IBM Power11 server models and limits (S1122, S1124, E1150, E1180): TechChannel, “Power11 server models”.",
        "Measured in this repository on x86 with the same software versions: README.md “Validation status” "
        "and the Solution 10 zero-shot run (2026-11-17).",
        "Model sizes and licences: MODELS.md (verified on Hugging Face 2026-11-15/17).",
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=17 * mm,
                            bottomMargin=16 * mm, title="Sizing guide: IBM Power AI Solutions for Iraqi enterprises",
                            author="IBM Power AI Solutions repository", subject="Capacity planning")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "SIZING.pdf"))
    print("wrote", build(Path(ap.parse_args().out)))
