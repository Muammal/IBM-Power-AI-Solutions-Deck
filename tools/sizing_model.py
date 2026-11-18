"""Capacity model for the twelve solutions: business drivers (users, cameras, calls, transactions, ...) ->
cores, memory, storage, network and servers on IBM Power10 / Power11.

Every number that is not a business driver is a planning factor in HW or PROFILES. They are ESTIMATES: Power
hardware was not available when this model was written. Calibrate them on your own LPAR (commands in
docs/SIZING.pdf, "Calibrate on your LPAR"), edit HW below, and rebuild the PDF:

    python tools/sizing_model.py          # print the tables
    python tools/build_sizing_pdf.py      # write docs/SIZING.pdf
"""
from __future__ import annotations

import copy
import math

# --------------------------------------------------------------------------------------------------------------
# 1. Hardware planning factors (one dedicated Power10/Power11 core, SMT4)
# --------------------------------------------------------------------------------------------------------------
HW = {
    # llama.cpp matrix maths on the MMA units (Q8_0/Q4_0 use int8 MMA kernels). Effective tera-ops/s per core
    # during prompt processing. THE most important factor: prompt processing (prefill) dominates LLM cost on CPUs.
    # Calibrate: llama-bench pp512 tok/s x 2 x parameters / cores.
    "llm_tops_per_core": 0.25,
    # Memory bandwidth an LLM LPAR really gets (token generation reads all weights once per step).
    # Calibrate: llama-bench tg128 tok/s (1 stream) x weight bytes.
    "mem_bw_per_core_gbs": 9.0,
    "mem_bw_cap_gbs": 250.0,          # ceiling for one LPAR inside one socket (keep LLM LPARs on one socket)
    # RT-DETRv2-r18, 640x640, ONNX Runtime fp32 (MLAS). Measured 0.86 core-seconds on an x86 laptop
    # (108 ms x 8 threads); Power10 MMA fp32 assumed a little better. Calibrate: common/python/vision/benchmark.py
    "vision_core_s_per_frame": 0.6,
    "decode_core_720p": 0.05,          # H.264 sub-stream decode per camera (PyAV)
    "decode_core_1080p": 0.25,         # main stream, needed for plate reading
    "whisper_core_h_per_audio_h": 2.0, # whisper.cpp large-v3-turbo q8_0, beam 5 (beam 1 is ~40% cheaper)
    "madlad_tok_s_per_16c": 50.0,      # MADLAD-400-3B bf16, transformers, batched, per 16-core LPAR
    "embed_core_s_per_chunk": 0.3,     # Granite Embedding 278M fp32, ~400-token chunk
    "convert_core_s_per_page": 2.0,    # Docling conversion for RAG ingestion
    "ocr_core_s_per_page": 6.0,        # Tesseract ara+eng 300 dpi + Docling layout (Solution 03)
    "fraud_core_ms_per_tx": 3.0,       # features + XGBoost + Isolation Forest + API (Solution 06)
    "ticket_core_s": 0.4,              # int8 ONNX classifier, 278M encoder (Solution 07)
    "ttm_core_s_per_series": 0.02,     # TTM-R2 batched forecast incl. data I/O (Solution 05)
    # Utilisation targets at the design point (headroom keeps latency stable)
    "util_interactive": 0.60, "util_batch": 0.75, "util_vision": 0.80, "util_tx": 0.50,
    # Building block for the bill of materials
    "server_cores": 60,                # Power S1124 (Power11, 4U, up to 60 cores / 8 TB). S1024 (Power10): 48
    "big_server_cores": 256,           # Power E1180 (Power11, up to 256 cores / 64 TB). E1080 (Power10): 240
}

# LLM inference nodes (one llama-server per LPAR; all GGUF files are IBM's official Granite 4.2 GGUFs)
LLM_NODES = {
    "g8": dict(name="Granite 4.2-8B Q8_0", params_b=8.8, bytes_per_param=1.06, cores=16, ram_gb=128, slots=16),
    "g3": dict(name="Granite 4.2-3B Q8_0", params_b=3.7, bytes_per_param=1.06, cores=8, ram_gb=64, slots=16),
    "g30": dict(name="Granite 4.2-30B Q4_0", params_b=29.0, bytes_per_param=0.56, cores=32, ram_gb=256, slots=8),
}


def llm_capacity(key: str, hw: dict = HW) -> dict:
    """Prefill tok/s, aggregate decode tok/s (all slots busy) and per-stream decode tok/s of one node."""
    n = LLM_NODES[key]
    ops = n["cores"] * hw["llm_tops_per_core"] * 1e12
    ops_per_token = 2 * n["params_b"] * 1e9                       # one multiply-add per weight per token
    bw = min(n["cores"] * hw["mem_bw_per_core_gbs"], hw["mem_bw_cap_gbs"]) * 1e9
    weights = n["params_b"] * 1e9 * n["bytes_per_param"]
    step = weights / bw + n["slots"] * ops_per_token / ops        # read weights once, compute every slot's token
    return dict(prefill=ops / ops_per_token, decode=n["slots"] / step, stream=1 / step,
                weights_gb=weights / 1e9)


# --------------------------------------------------------------------------------------------------------------
# 2. Scales: private organisations and country-wide programmes (business drivers)
# --------------------------------------------------------------------------------------------------------------
IRAQ_POPULATION = 46_118_793      # 2024 census, final results (Ministry of Planning, Feb 2025)

SCALES = [
    dict(key="S", name="Small organisation", who="~300 staff, 1 site",
         staff=300, public_users=20_000, public_dau=0.02, pages_day=300, corpus_pages=20_000, agents=10,
         series=500, tx_day=20_000, tickets_day=200, words_day=10_000, incidents_day=10, cameras=32,
         osint_day=1_000, lanes=4, central_sites=1, hubs=0, dr=0.0, training_cores=16),
    dict(key="M", name="Medium organisation", who="~3,000 staff, regional",
         staff=3_000, public_users=200_000, public_dau=0.02, pages_day=3_000, corpus_pages=200_000, agents=60,
         series=5_000, tx_day=300_000, tickets_day=2_000, words_day=100_000, incidents_day=50, cameras=250,
         osint_day=3_000, lanes=20, central_sites=1, hubs=0, dr=0.0, training_cores=32),
    dict(key="L", name="Large enterprise", who="~30,000 staff, national branches",
         staff=30_000, public_users=2_000_000, public_dau=0.02, pages_day=30_000, corpus_pages=2_000_000,
         agents=400, series=50_000, tx_day=3_000_000, tickets_day=20_000, words_day=1_000_000,
         incidents_day=300, cameras=2_000, osint_day=10_000, lanes=150, central_sites=2, hubs=3, dr=0.5,
         training_cores=64),
    dict(key="N1", name="Sector-wide national", who="national bank / telecom / ministry, ~10M customers",
         staff=100_000, public_users=10_000_000, public_dau=0.01, pages_day=200_000, corpus_pages=10_000_000,
         agents=1_500, series=500_000, tx_day=20_000_000, tickets_day=150_000, words_day=5_000_000,
         incidents_day=1_000, cameras=5_000, osint_day=30_000, lanes=800, central_sites=2, hubs=6, dr=1.0,
         training_cores=128),
    dict(key="N2", name="Country-wide", who=f"government platform, all {IRAQ_POPULATION / 1e6:.1f}M residents",
         staff=500_000, public_users=IRAQ_POPULATION, public_dau=0.005, pages_day=1_000_000,
         corpus_pages=30_000_000, agents=4_000, series=5_000_000, tx_day=60_000_000, tickets_day=600_000,
         words_day=20_000_000, incidents_day=3_000, cameras=20_000, osint_day=80_000, lanes=2_500,
         central_sites=2, hubs=19, dr=1.0, training_cores=256),
]

# --------------------------------------------------------------------------------------------------------------
# 3. Profiles: "best" = highest-quality production configuration; "lean" = the same solutions with cost levers
# --------------------------------------------------------------------------------------------------------------
BEST = {
    "01": dict(model="g8", turns=3, prompt=2600, cached=700, out=250, cache_hit=0.4, peak=0.10),
    "02": dict(model="g8", dau=0.30, qpd=4, prompt=2400, cached=600, out=350, think_share=0.2, think_tok=600,
               peak=0.15, chunks_per_page=1.5, change=0.02),
    "03": dict(pages_per_doc=3, prompt=2500, out=400, recheck_share=0.2, window_h=20),
    "04": dict(talk_h=5, calls_per_h=15, prompt=1100, out=250, whisper=1.0, window_h=20),
    "05": dict(runs_per_day=24),
    "06": dict(peak_factor=4, flag_rate=0.003, prompt=1200, out=200, peak=0.15),
    "07": dict(peak_factor=3, teacher_share=0.05, prompt=800, out=60),
    "08": dict(tok_per_word=1.6, post_share=0.3, seg_words=40, window_h=20),
    "09": dict(calls=6, prompt=2000, out=300, think=500, g30_share=0.3, peak=0.10),
    "10": dict(hi_share=0.4, hi_fps=2.0, lo_fps=1.0, activity=0.5, alerts_per_cam_day=0.5, agent_calls=5,
               prompt=1500, out=300, think=400, peak=0.10),
    "11": dict(model="g8", prompt=900, out=250, briefs_day=24, brief_prompt=20000, brief_out=2500),
    "12": dict(fps=10.0, activity=0.6, roi=0.6, vehicles_lane_day=8000, violation_rate=0.005),
    "dr_scale": 1.0,
}
LEAN_CHANGES = {
    "01": dict(model="g3", cache_hit=0.6, prompt=1800),
    "02": dict(think_share=0.0, prompt=1800),
    "03": dict(recheck_share=0.0),
    "04": dict(whisper=0.6),          # beam 1
    "08": dict(post_share=0.1),
    "09": dict(g30_share=0.0),
    "10": dict(lo_fps=0.0),           # analyse entrance / high-risk cameras only
    "11": dict(model="g3"),
    "12": dict(fps=5.0),
    "dr_scale": 0.5,                  # warm standby instead of active-active
}


def lean_profile() -> dict:
    p = copy.deepcopy(BEST)
    for k, v in LEAN_CHANGES.items():
        if isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


PROFILES = {"best": BEST, "lean": lean_profile()}

SOLUTIONS = {
    "01": "Arabic/English assistant (public)", "02": "Enterprise document Q&A (RAG)",
    "03": "Document intelligence", "04": "Speech analytics", "05": "Load / demand forecasting",
    "06": "Fraud & anomaly detection", "07": "Ticket classification", "08": "AR/KU/EN translation",
    "09": "IT-operations agent", "10": "CCTV weapon detection", "11": "Situational awareness (OSINT)",
    "12": "Traffic ANPR & section speed",
}


# --------------------------------------------------------------------------------------------------------------
# 4. Demand per solution
# --------------------------------------------------------------------------------------------------------------
class Demand:
    """Cores per pool, LLM load per model (in node-equivalents at target utilisation), storage, network."""

    def __init__(self):
        self.cores = {}        # pool -> cores
        self.llm = {}          # model -> node-equivalents
        self.storage_tb = 0.0
        self.net_mbps = 0.0
        self.ram_extra_gb = 0.0

    def add_cores(self, pool, c):
        self.cores[pool] = self.cores.get(pool, 0.0) + c

    def add_llm(self, model, requests_per_s, prefill, decode, util, hw):
        """requests_per_s x (prefill tokens, decode tokens) per request, at the design point."""
        cap = llm_capacity(model, hw)
        load = requests_per_s * (prefill / cap["prefill"] + decode / cap["decode"]) / util
        self.llm[model] = self.llm.get(model, 0.0) + load

    def llm_cores(self):
        return sum(v * LLM_NODES[m]["cores"] for m, v in self.llm.items())

    def total_cores(self):
        return sum(self.cores.values()) + self.llm_cores()


def per_day(x, window_h=24.0):
    return x / (window_h * 3600.0)


def peak_rate(daily, share):
    return daily * share / 3600.0


def demand(sol: str, s: dict, p: dict, hw: dict = HW) -> Demand:
    d, q = Demand(), p.get(sol, {})
    ui, ub = hw["util_interactive"], hw["util_batch"]
    if sol == "01":
        req = s["public_users"] * s["public_dau"] * q["turns"]
        r = peak_rate(req, q["peak"])
        d.add_llm(q["model"], r * (1 - q["cache_hit"]), q["prompt"] - q["cached"], q["out"], ui, hw)
        d.add_cores("services", 2 + r * 0.05)                               # API, retrieval, answer cache
        d.storage_tb = req * 6e3 * 90 / 1e12                                 # 90 days of conversation logs
    elif sol == "02":
        qd = s["staff"] * q["dau"] * q["qpd"]
        r = peak_rate(qd, q["peak"])
        d.add_llm(q["model"], r, q["prompt"] - q["cached"], q["out"] + q["think_share"] * q["think_tok"], ui, hw)
        chunks = s["corpus_pages"] * q["chunks_per_page"]
        d.add_cores("services", 2 + chunks * q["change"] * (hw["embed_core_s_per_chunk"]
                    + hw["convert_core_s_per_page"] / q["chunks_per_page"]) / (20 * 3600) / ub)
        d.ram_extra_gb = chunks * 768 * 4 * 1.5 / 1e9                        # fp32 vectors + index overhead
        d.storage_tb = (s["corpus_pages"] * 0.3e6 + chunks * 5e3) / 1e12
    elif sol == "03":
        pages = s["pages_day"]
        d.add_cores("services", 2 + pages * hw["ocr_core_s_per_page"] / (q["window_h"] * 3600) / ub)
        docs = per_day(pages / q["pages_per_doc"], q["window_h"])
        d.add_llm("g3", docs, q["prompt"], q["out"], ub, hw)
        if q["recheck_share"]:
            d.add_llm("g8", docs * q["recheck_share"], q["prompt"] + 500, q["out"], ub, hw)
        d.storage_tb = pages * 0.5e6 * 90 / 1e12
    elif sol == "04":
        audio_h = s["agents"] * q["talk_h"]
        d.add_cores("services", 2 + audio_h * hw["whisper_core_h_per_audio_h"] * q["whisper"] / q["window_h"] / ub)
        d.add_llm("g8", per_day(audio_h * q["calls_per_h"], q["window_h"]), q["prompt"], q["out"], ub, hw)
        d.storage_tb = audio_h * 30e6 * 90 / 1e12                            # compressed audio, 90 days
    elif sol == "05":
        d.add_cores("services", 2 + s["series"] * q["runs_per_day"] * hw["ttm_core_s_per_series"] / 86400 / 0.5)
        d.storage_tb = s["series"] * 24 * 365 * 3 * 32 / 1e12               # 3 years of hourly history
    elif sol == "06":
        tps = s["tx_day"] / 86400 * q["peak_factor"]
        d.add_cores("services", 2 + tps * hw["fraud_core_ms_per_tx"] / 1000 / hw["util_tx"])
        d.add_llm("g8", peak_rate(s["tx_day"] * q["flag_rate"], q["peak"]), q["prompt"], q["out"], ui, hw)
        d.storage_tb = s["tx_day"] * 1e3 * 400 / 1e12
    elif sol == "07":
        tps = s["tickets_day"] / 86400 * q["peak_factor"]
        d.add_cores("services", 2 + tps * hw["ticket_core_s"] / 0.5)
        d.add_llm("g3", per_day(s["tickets_day"] * q["teacher_share"]), q["prompt"], q["out"], ub, hw)
        d.storage_tb = s["tickets_day"] * 5e3 * 365 / 1e12
    elif sol == "08":
        tok_s = per_day(s["words_day"] * q["tok_per_word"], q["window_h"])
        d.add_cores("madlad", tok_s / hw["madlad_tok_s_per_16c"] / ub * 16)
        segs = per_day(s["words_day"] * q["post_share"] / q["seg_words"], q["window_h"])
        seg_tok = q["seg_words"] * q["tok_per_word"]
        if q["post_share"]:
            d.add_llm("g8", segs, 2 * seg_tok + 400, seg_tok, ub, hw)
        d.add_cores("services", 2)
        d.storage_tb = s["words_day"] * 2 * 12 * 365 / 1e12                 # translation memory
    elif sol == "09":
        r = peak_rate(s["incidents_day"] * q["calls"], q["peak"])
        d.add_llm("g8", r * (1 - q["g30_share"]), q["prompt"], q["out"] + q["think"], ui, hw)
        if q["g30_share"]:
            d.add_llm("g30", r * q["g30_share"], q["prompt"], q["out"] + q["think"], ui, hw)
        d.add_cores("services", 2)
        d.storage_tb = s["incidents_day"] * 200e3 * 365 / 1e12
    elif sol == "10":
        cams = s["cameras"]
        fps = q["hi_share"] * q["hi_fps"] + (1 - q["hi_share"]) * q["lo_fps"]
        analysed = cams if q["lo_fps"] else cams * q["hi_share"]
        d.add_cores("vision10", (cams * fps * q["activity"] * hw["vision_core_s_per_frame"]
                                 + analysed * hw["decode_core_720p"]) / hw["util_vision"])
        d.add_cores("services", 4)                                          # console, API, notifications
        d.add_llm("g8", peak_rate(analysed * q["alerts_per_cam_day"] * q["agent_calls"], q["peak"]),
                  q["prompt"], q["out"] + q["think"], ui, hw)
        d.storage_tb = analysed * (q["alerts_per_cam_day"] * 3e6 * 30 + 2e9) / 1e12   # events + training data
        d.net_mbps = analysed * 1.5                                         # 720p sub-streams
    elif sol == "11":
        r = per_day(s["osint_day"])
        d.add_llm(q["model"], r, q["prompt"], q["out"], ub, hw)
        d.add_llm(q["model"], per_day(q["briefs_day"]), q["brief_prompt"], q["brief_out"], ub, hw)
        d.add_cores("services", 2 + s["osint_day"] / 20_000)
        d.storage_tb = s["osint_day"] * 30e3 * 365 / 1e12
    elif sol == "12":
        lanes = s["lanes"]
        per_lane = (q["fps"] * q["activity"] * hw["vision_core_s_per_frame"] * q["roi"]
                    + hw["decode_core_1080p"] + 0.05)                        # + plate OCR
        d.add_cores("vision12", lanes * per_lane / hw["util_vision"])
        d.add_cores("services", 2)
        reads = lanes * q["vehicles_lane_day"]
        d.storage_tb = (reads * 40e3 + reads * q["violation_rate"] * 4e6 * 365) / 1e12
        d.net_mbps = lanes * 6.0                                             # 1080p main streams
    return d


# --------------------------------------------------------------------------------------------------------------
# 5. From demand to a deployable estate
# --------------------------------------------------------------------------------------------------------------
POOL_NODE = {"services": 4, "madlad": 16, "vision10": 16, "vision12": 16}    # LPAR size per pool (cores)
POOL_RAM_PER_CORE = {"services": 8, "madlad": 8, "vision10": 4, "vision12": 4}
POOL_NAMES = {"g8": "LLM: Granite 4.2-8B Q8_0", "g3": "LLM: Granite 4.2-3B Q8_0", "g30": "LLM: Granite 4.2-30B Q4_0",
              "madlad": "Translation: MADLAD-400-3B", "vision10": "Vision: weapons (cameras)",
              "vision12": "Vision: ANPR (lanes)", "services": "Services, OCR, ASR, scoring"}
SITE_BOUND = {"vision10", "vision12"}       # run next to the cameras (hubs); not duplicated by DR


NO_SPARE = {"g30"}                          # the 30B node fails over to the 8B pool instead of keeping a spare


def pools_for(sols: dict, scale: dict, p: dict, dr: float) -> list:
    """Sum demand per pool, round to whole LPARs per location, add N+1 and the disaster-recovery copy."""
    pools = {}
    for d in sols.values():
        for m, v in d.llm.items():
            pools[m] = pools.get(m, 0.0) + v * LLM_NODES[m]["cores"]
        for k, v in d.cores.items():
            pools[k] = pools.get(k, 0.0) + v
    rows = []
    for k, cores in pools.items():
        if cores <= 0:
            continue
        node = LLM_NODES[k]["cores"] if k in LLM_NODES else POOL_NODE[k]
        ram_node = LLM_NODES[k]["ram_gb"] if k in LLM_NODES else POOL_RAM_PER_CORE[k] * node
        locations = scale["hubs"] if (k in SITE_BOUND and scale["hubs"]) else 1
        per_loc = math.ceil(cores / locations / node)
        spare = 0 if k in NO_SPARE else max(1, math.ceil(0.1 * per_loc))
        prod_nodes, ha_nodes = per_loc * locations, (per_loc + spare) * locations     # N+1 at every location
        dr_nodes = 0 if k in SITE_BOUND else math.ceil(ha_nodes * dr)
        rows.append(dict(pool=k, name=POOL_NAMES[k], demand=cores, node=node, locations=locations,
                         prod_nodes=prod_nodes, ha_nodes=ha_nodes, dr_nodes=dr_nodes,
                         cores=(ha_nodes + dr_nodes) * node, ram_gb=(ha_nodes + dr_nodes) * ram_node))
    return rows


def size_solution(sol: str, scale: dict, profile: str = "best", hw: dict = HW) -> dict:
    """Footprint of ONE solution deployed alone (production + N+1, no DR, no shared platform)."""
    p = PROFILES[profile]
    d = demand(sol, scale, p, hw)
    rows = pools_for({sol: d}, scale, p, 0.0)
    return dict(demand=d.total_cores(), cores=sum(r["cores"] for r in rows),
                ram_gb=sum(r["ram_gb"] for r in rows) + d.ram_extra_gb, storage_tb=d.storage_tb, rows=rows)


def size(scale: dict, profile: str = "best", hw: dict = HW) -> dict:
    """All twelve solutions on one shared estate (shared LLM pools), with HA, DR, non-prod, training, platform."""
    p = PROFILES[profile]
    sols = {k: demand(k, scale, p, hw) for k in SOLUTIONS}
    dr_f = scale["dr"] * p["dr_scale"]
    rows = pools_for(sols, scale, p, dr_f)
    prod = sum(r["prod_nodes"] * r["node"] for r in rows)
    ha = sum(r["ha_nodes"] * r["node"] for r in rows)
    dr = sum(r["dr_nodes"] * r["node"] for r in rows)
    nonprod = max(16, math.ceil(0.10 * ha))
    training = scale["training_cores"]
    sub = ha + dr + nonprod + training
    platform = 8 + math.ceil(0.05 * sub)              # VIOS, proxy/SSO, monitoring, databases, backup
    total = sub + platform
    extra_ram = sum(d.ram_extra_gb for d in sols.values())
    ram_gb = (sum(r["ram_gb"] for r in rows) + extra_ram * (1 + dr_f)
              + (nonprod + training) * 8 + platform * 4)
    storage = sum(d.storage_tb for d in sols.values()) + 0.2 * (scale["central_sites"] + scale["hubs"])  # + models
    # servers: central sites hold everything except the camera pools when there are regional hubs
    vision = sum(r["cores"] for r in rows if r["pool"] in SITE_BOUND)
    central = total - (vision if scale["hubs"] else 0)
    per_site = math.ceil(central / scale["central_sites"])
    sc = hw["server_cores"]
    central_servers = scale["central_sites"] * max(2, math.ceil(per_site / sc))
    hub_servers = scale["hubs"] * max(1, math.ceil(vision / scale["hubs"] / sc)) if scale["hubs"] else 0
    return dict(scale=scale, profile=profile, solutions=sols, pools=rows, prod=prod, ha=ha, dr=dr,
                nonprod=nonprod, training=training, platform=platform, total=total, ram_tb=ram_gb / 1000,
                storage_tb=storage, net_gbps=sum(d.net_mbps for d in sols.values()) / 1000,
                central_servers=central_servers, hub_servers=hub_servers, per_site_cores=per_site,
                vision_cores=vision, hub_cores=math.ceil(vision / scale["hubs"]) if scale["hubs"] else 0,
                big_servers=scale["central_sites"] * math.ceil(per_site / hw["big_server_cores"]),
                ingest_core_h=scale["corpus_pages"] * (hw["convert_core_s_per_page"]
                                                       + BEST["02"]["chunks_per_page"] * hw["embed_core_s_per_chunk"]) / 3600)


def sensitivity(factor: str, values, profile="best"):
    out = {}
    for v in values:
        hw = dict(HW, **{factor: v})
        out[v] = [size(s, profile, hw)["total"] for s in SCALES]
    return out


if __name__ == "__main__":
    for m in LLM_NODES:
        c = llm_capacity(m)
        print(f"{m:4} prefill {c['prefill']:6.0f} tok/s  decode {c['decode']:6.0f} tok/s  "
              f"per stream {c['stream']:5.1f} tok/s  weights {c['weights_gb']:.1f} GB")
    for prof in PROFILES:
        print(f"\n== profile {prof}")
        for s in SCALES:
            r = size(s, prof)
            per = "  ".join(f"{k}:{d.total_cores():.0f}" for k, d in r["solutions"].items())
            alone = "  ".join(f"{k}:{size_solution(k, s, prof)['cores']}" for k in SOLUTIONS)
            print(f"{s['key']:3} prod {r['prod']:6} ha {r['ha']:6} dr {r['dr']:6} total {r['total']:6} "
                  f"RAM {r['ram_tb']:6.1f} TB  storage {r['storage_tb']:7.1f} TB  net {r['net_gbps']:5.1f} Gbps  "
                  f"servers {r['central_servers']}+{r['hub_servers']}\n    shared {per}\n    alone  {alone}")
