"""Shared computer-vision building blocks for the CCTV solutions (10 weapon detection, 12 traffic).

video     : resilient RTSP/file readers (PyAV), newest-frame-only, reconnect with backoff
motion    : cheap motion gate so the detector only runs when something moves
detector  : RT-DETR(v2) ONNX Runtime wrapper (CPU, Power10 MMA via MLAS)
tracker   : IoU tracker with hit/miss counting (alert only on persistent detections)
evidence  : SHA-256 hashing, Ed25519 signing, hash chains for tamper-evident records
store     : SQLite (WAL) helper with an append-only audit table
auth      : bearer tokens -> roles for the operator APIs
"""
