# Compliance: compute choices for Iraqi enterprises

> **Not legal advice.** Export-control rules changed several times between 2023 and 2026. Confirm the
> classification of any accelerator with the vendor (its ECCN) and, when in doubt, with export-control
> counsel before you buy or rent it. Last reviewed: **2026-11-15**.

## The rule this repo follows

Every solution must do one of these two things:

1. **CPU-only on-premises (default).** Inference *and* fine-tuning run on IBM Power10/Power11 CPUs with
   their built-in MMA matrix units. No accelerator, no export-controlled chip, and the data never leaves the
   enterprise's own data center in Iraq.
2. **GPU only where the GPU is not regulated for Iraq.** Used only for optional fine-tuning jobs that are too
   slow on CPU. The resulting model always returns to Power for inference.

## Why GPUs are a problem

US Export Administration Regulations (EAR) control advanced AI chips under ECCN **3A090** (and computers
containing them under **4A090**). Licence requirements for these items apply to destinations in Country
Groups **D:1, D:4 and D:5**. **Iraq is in Country Group D:1** (and D:3, D:4), so shipping such GPUs to Iraq, or
to an Iraqi end user, needs a US licence. Iraq also has country-specific controls in EAR part 746.

3A090 is defined by **Total Processing Performance (TPP)** and **performance density (PD = TPP / die area in mm²)**:

* TPP = (dense, non-sparse) TOPS × bit length of the operation, maximised over data types.
* **3A090.a**: TPP ≥ 4800, **or** TPP ≥ 1600 with PD ≥ 5.92.
* **3A090.b**: TPP ≥ 2400 and < 4800 with PD ≥ 1.6 and < 5.92, **or** TPP ≥ 1600 with PD ≥ 3.4 and < 5.92.

### Indicative screening of common GPUs

Computed from public datasheets. **Indicative only**: the vendor's official ECCN decides.

| GPU | Best dense rate used | TPP | Die (mm²) | PD | Indicative result |
|---|---|---:|---:|---:|---|
| NVIDIA T4 | INT8 130 TOPS | 1,040 | 545 | 1.9 | Below 3A090 thresholds |
| NVIDIA V100 | FP16 tensor 125 TFLOPS | 2,000 | 815 | 2.5 | Below 3A090 thresholds |
| NVIDIA A10 | INT8 250 TOPS | 2,000 | 628 | 3.2 | Below 3A090 thresholds |
| NVIDIA L4 | FP8 ~242 TFLOPS | ~1,940 | 294 | 6.6 | **3A090.a**: controlled (PD) |
| NVIDIA A100 | FP16 312 TFLOPS | 4,992 | 826 | 6.0 | **3A090.a**: controlled |
| NVIDIA L40S, H100, H200, B200 | FP8 | > 4,800 | | | **Controlled** |
| AMD MI300X, Intel Gaudi 3 | FP8 | > 4,800 | | | **Controlled** |
| Consumer RTX 4090 / 5090 | FP8 | > 4,800 | | | **Controlled** |

Practical reading:

* Older/smaller GPUs (T4, V100, A10 class) appear to fall **below** the 3A090 thresholds. They are still slow
  for modern models, but enough to LoRA-tune the 3B models in this repo or a 278M encoder.
* Anything current and fast is controlled. Do not plan around it for Iraq.
* Using a GPU in a foreign cloud is not a physical export, but providers screen customers (sanctions,
  know-your-customer rules for infrastructure-as-a-service), and **your data leaves Iraq**. Use cloud GPUs only
  with synthetic or non-sensitive data, and only if your regulator allows it.

### IBM Cloud specifics

* IBM Cloud VPC GPU profiles (`ibmcloud is instance-profiles | grep gx`) include L4, L40S, H100/H200, MI300X and
  Gaudi 3 classes (all controlled by the table above). V100 (`gx2`) profiles, where still offered, are the
  only class that screens below the thresholds.
* **IBM Power Virtual Server and on-prem Power servers have no GPU**: the AI work runs on the CPU's MMA units.
  That is the core of this repo.
* **IBM Spyre** (the AI accelerator card for Power11 and IBM Z): check its ECCN with IBM before ordering it for
  Iraq. No solution here depends on it.

## Data residency and sector rules

* No IBM Cloud region is in Iraq. Banking data (Central Bank of Iraq supervision), government data and
  telecom subscriber data are commonly expected to stay in-country. The **on-prem** path keeps it there.
* Every solution runs **fully offline** after setup (`HF_HUB_OFFLINE=1` in the services), so it works on
  air-gapped networks.
* Personal data: Solutions 02, 04 and 09 can process personal data (documents, call recordings, logs). Apply
  retention limits and access control; Solution 04's step-by-step includes a consent/retention step.

## Model licences

Only models with permissive licences are used as defaults (Apache-2.0 or MIT). See [`MODELS.md`](MODELS.md).
Models with research-only or non-commercial licences (for example NLLB-200, Qwen2.5-3B, Granite Speech
`-nc` variants) are excluded.
