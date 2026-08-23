# FORENSIC AUDIT REPORT — QHDALabs Wildfire Risk PL v5

**Subject:** node `jawor`, analysis date 2026-08-23
**Observed result:** NDWI stress 1.000 · QTE 0.396 · FWI 0.405 · Bridge fired · Final 0.7528 · CRITICAL
**Scope:** `/v5` only. No conclusions carried over from v1–v4.
**Run:** `run_id ff037e0a-d9f8-4799-a8fd-099b9b6093e3`, `git_sha f6a9ab1e1fdba1961f3c1d61548bb6a74ad5bf14`
**Method:** code is the source of truth. Every claim is tagged FACT (verified in code/data), INFERENCE (logical consequence), or UNKNOWN / NOT PROVEN FROM CODE.

---

> ## Remediation status — v5.1
>
> This report describes **v5.0 as audited on 2026-08-23**. It is retained
> unchanged as the baseline record. **v5.1** is a controlled correction that
> fixes the six CRITICAL/HIGH findings without touching the architecture.
>
> | Finding | Severity | v5.1 |
> | --- | --- | --- |
> | F-01 NDWI rank published as measurement | CRITICAL | **FIXED** — fusion reads `ndwi_stress_latest`; rank kept as `ndwi_stress_rank`; bounds published |
> | F-02 Stale acquisition reported as SUCCESS | CRITICAL | **FIXED** — `MAX_ACQUISITION_AGE_DAYS=14`, `stale_acquisition` flag, `DEGRADED`, date on every score and alert |
> | F-03 Bridge label misstates the mechanism | CRITICAL | **FIXED (label)** — relabelled; the sign-test mechanism itself is unchanged by design |
> | F-04 Backend-dependent tier | HIGH | **FIXED** — analytic `bridge_rate` in both engines; `bridge_near_threshold` flag |
> | F-05 Dead soil channel | HIGH | **FIXED** — `soil_moisture_0_to_7cm`; missing data flagged and renormalised |
> | F-06 FWI blind to antecedent rain | HIGH | **FIXED** — 14-day exponentially decayed rainfall |
> | F-07 … F-14 | MEDIUM / LOW / INFO | **OPEN** — see §9 |
>
> Effect on the audited case: Jawor `0.7528 CRITICAL` → `0.5261 MODERATE`,
> sentinel step `SUCCESS` → `DEGRADED`, alert count `1` → `0`.
>
> F-03 is marked *fixed (label)* deliberately: v5.1 corrects the claim, not the
> mechanism. Making the bridge genuinely sequential is a model change, not a
> correction, and belongs to a calibration release.

---

## Executive Summary

1. **The score is exactly reproducible.** Every component — base 0.6474, bridge +0.08, network +0.0254 — replays to within 5×10⁻⁴ from `risk_scores.json`. No arithmetic error, no corruption.
2. **"NDWI stress 1.000" is a rank, not a measurement.** The absolute stress mapping yields 0.5312 for Jawor. The 1.000 comes from min-max normalisation across the 34-node network — by construction exactly one node scores 1.000 on every run, however wet the region is.
3. **The satellite channel is 18 days stale and unchanged across four runs.** Latest acquisition 2026-08-05, fetched 2026-08-15, reused on 08-16, 08-19 and 08-23. The sentinel step made *zero* API requests and still reported `SUCCESS` at `coverage_percent: 100.0`.
4. **The quantum bridge is a sign test.** The Qiskit path reduces analytically to `bridge_rate = sin²(θ_past/2)`, so `bridge_fired ⟺ ndwi_values[-2] < 0`. Verified on all 34 nodes: 34/34 agreement, zero mismatches.
5. **The bridge decision is wind-independent.** Wind never reaches the ancilla qubit. The label "Sequential dry→wind pattern confirmed" is not supported by the implementation.
6. **Two knife-edge margins carry the alert.** Jawor clears CRITICAL by +0.0028, and its bridge clears its own threshold by +0.0024 on a past-moisture value of −0.003. Remove the bridge bonus and the node is HIGH at 0.6428.
7. **The backend changes the answer.** At Jawor's operating point the NumPy fallback engine fires the bridge for 3 of 8 seeds. The tier is a function of which simulator ran.
8. **Jawor is the wettest node in the network.** 91.6 mm of rain in the seven days before the run — the highest of all 34 nodes — at 19 °C with `drought_days = 1`.
9. **The soil-moisture channel is dead network-wide.** All 34 nodes carry `soil_min: null`; `_fwi_from_weather` silently substitutes a hardcoded 0.20, with no quality flag raised.
10. **Ignition never touches the score.** `ignition_score`, FEI and QIES are all computed *from* the final score. The alert tier is a susceptibility ranking published as "Unified Wildfire Risk".

## Overall Verdict

**SIGNIFICANT LOGIC ISSUE FOUND**

The arithmetic is sound — the published 0.7528 replays exactly from the published signals. The defects are semantic: three of the four fusion inputs do not measure the quantity their name and documentation claim, and the satellite channel carrying 45 % of the weight has been frozen on one 2026-08-05 acquisition since 2026-08-15.

**DATA / CACHE ISSUE FOUND** applies as a co-primary verdict and is documented in full in §3.

---

## 1. Data Flow

Five steps, orchestrated by `run_all.py` as separate subprocesses with a dependency gate (`STEP_DEPENDENCIES`, `run_all.py:47`). Each writes an atomic manifest to `topology/pipeline_steps/<step>.json`.

```
  EXTERNAL                    STEP / FUNCTION                    ARTIFACT                 CACHE

  Open-Meteo archive ──▶ [1] topology                       ──▶ nodes.json            .cache_topology/
   (hourly, 14 d)           fetch_weather_history()             graph.json             weather_*.pkl
                            enrich_node()                       network_map.html       TTL 6 h  (pickle)
                            build_adjacency_graph(60 km)
                                     │
                                     │  nodes[].latest{temp_max,rh_min,wind_max,
                                     │                 rain_total,soil_min=NULL,vpd_mean}
                                     │  nodes[].drought_days
                                     ▼
  Copernicus CDSE ─────▶ [2] sentinel                       ──▶ ndwi_sentinel.json   .cache_topology/
   Statistical API          fetch_ndwi_timeseries()             nodes_enriched.json    sentinel/*.json
   S2-L2A, P10D bins        └─ B03/B08 → ndwi_surface_water     network_map_v2.html    TTL 10 d  (JSON)
   SCL 3,8,9,10,11 masked   └─ B8A/B11 → vegetation_moisture                           fingerprinted
                            ndwi_stress_latest =(1-m)/2                                ── 0 API CALLS
                            normalise_stress_across_network()                             THIS RUN
                                     │      └─ MIN-MAX over 34 nodes  ◀── population-dependent
                                     ▼
                       [3] qte  ── qhdalabs_wildfire_qte_v1.py ──▶ qte_results.json    (no cache)
                            encode_node()  → 5 Ry angles
                            _run_qiskit_qte()  ── analytic, 0 shots
                            └─ bridge_rate = sin²(θ_past/2)   ◀── reduces to sign(ndwi[-2])
                            compute_qte_score(zz, bridge)
                                     │
                       [4] ignition ─────────────────────────▶ ignition_scores.json  ignition_cache/
                            OSM / BDOT10k / LPIS / FIRMS         (6 weighted sublayers) derived/ + FIRMS
                                     │                                                  fingerprinted
                                     ▼
                       [5] fusion ── compute_risk_score()   ──▶ risk_scores.json       (no cache)
                            base = .45·NDWI + .30·QTE                alerts.json
                                 + .15·FWI  + .10·TREND              final_map.html
                            × eco_mult  + 0.08·bridge  + 0.05·net
                            final = clip(·, 0, 1) ──▶ _tier()
                                     │
                                     ├──▶ FEI  = f(final, ignition)   ◀── ignition enters HERE ONLY
                                     └──▶ QIES = f(final, ignition)       (never feeds back)
```

### Stage table

| Stage | Function | Output | Cache / TTL | Fresh this run? | Quality flags kept |
|---|---|---|---|---|---|
| Weather | `fetch_weather_history` `topology_v1.py:448` | `weather_history[]` | pickle, 6 h | Yes — fetched 2026-08-23 15:03 | none; NaN→null silently |
| Topology | `enrich_node` `topology_v1.py:629` | `nodes.json`, `graph.json` | — | Yes | none |
| **Sentinel** | `fetch_ndwi_timeseries` `sentinel_v1.py:479` | `ndwi_sentinel.json` | JSON, 10 d | **No — 34/34 cache hits, 0 API** | `cloud_masked`, `*_uncalibrated`, `b11_not_gao` |
| Normalise | `normalise_stress_across_network` `sentinel_v1.py:672` | `ndwi_stress_normalised` | — | Recomputed | **none — mapping not recorded** |
| QTE | `_run_qiskit_qte` `qte_v1.py:336` | `qte_results.json` | — | Recomputed (deterministic) | `calibration_status`, `random_seed`, `backend` |
| Ignition | `run_ignition_pipeline` | `ignition_scores.json` | derived GIS + FIRMS | 3 cache hits, FIRMS cached | `coverage_percent`, `valid_for_fusion`, `missing_layers` |
| Fusion | `compute_risk_score` `fusion_v1.py:256` | `risk_scores.json`, `alerts.json` | — | Recomputed | `calibration_flags`, `upstream_statuses`, `operational_validity=false` |

**Flag propagation gap.** The sentinel step correctly attaches `vegetation_moisture_index_uncalibrated` and `cloud_masked` to each result. Neither survives into `risk_scores.json` — fusion emits a fixed `calibration_flags: ["satellite_stress_mapping_uncalibrated"]` regardless of what upstream reported, and Jawor's published `data_quality_flags` is empty despite its 2-observation series.

---

## 2. Sentinel / NDWI Audit

### 2.1 Which indices are computed

Two, both from a single evalscript (`EVALSCRIPT_INDICES`, `sentinel_v1.py:432`), returned as bands `B0` and `B1`:

| Name | Formula | Bands | Res. | Reaches the score? |
|---|---|---|---|---|
| `ndwi_surface_water` | `(B03 − B08) / (B03 + B08)` | Green, NIR | 10 m | **No** — stored only |
| `vegetation_moisture_index` | `(B8A − B11) / (B8A + B11)` | NIR, SWIR | 20 m | **Yes** — the sole satellite input |

**FACT.** Only `vegetation_moisture_index` propagates. At `sentinel_v1.py:625` the field `ndwi_values` is assigned `moisture_values`, and every downstream consumer — `ndwi_latest`, `ndwi_mean_30d`, `ndwi_trend_14d`, `ndwi_stress_latest` — derives from that array. The Green-NIR series is written to `indices.ndwi_surface_water.raw_series` and never read again.

### 2.2 Is the "NDWI" naming methodologically correct?

**Partly, and the code is more honest than the field names.** The module header and `index_definitions()` state plainly that Green-NIR "is not the Gao (1996) vegetation liquid-water index" and that "Sentinel-2 B11 at 1.61 µm is an approximation, not Gao's 1.24 µm channel". The README repeats this correctly. That is good practice.

The problem is that the *runtime field names* contradict the documentation. The variable carrying B8A/B11 through the pipeline is `ndwi_values`, its summary is `ndwi_latest`, its stress is `ndwi_stress_latest`, its weight constant is `W_NDWI`, and the alert text says "NDWI stress". A reader of `alerts.json` sees "NDWI" and has no way to learn the number is NIR-SWIR canopy moisture. *(INFERENCE: a legacy artefact that the v5 documentation pass corrected in prose but not in code.)*

### 2.3 The four distinct quantities

| Quantity | Definition | Range | Jawor | Meaning |
|---|---|---|---|---|
| raw index | `(B8A−B11)/(B8A+B11)`, bbox mean per P10D bin | [−1, 1] | −0.0624 | Physical canopy moisture proxy |
| `ndwi_stress_latest` | `clip((1 − m)/2, 0, 1)` on the *last* bin | [0, 1] | 0.5312 | Affine rescale — absolute, uncalibrated |
| **`ndwi_stress_normalised`** | `(v − min)/(max − min)` over the 34-node set | [0, 1] | **1.0000** | **Rank within this run's population** |
| `ndwi_trend_14d` | polyfit slope vs `arange(n)` — per bin index | unbounded | −0.0594 | Not per-day, not 14-day (see 2.7) |

**FACT.** Fusion consumes `ndwi_stress_normalised` — the rank — at `fusion_v1.py:271`:

```python
ndwi_stress = float(
    node.get("ndwi_stress_normalised") or node.get("ndwi_stress_latest", 0.0)
)
```

### 2.4 The raw → stress function

```python
# v5/qhdalabs_wildfire_sentinel_v1.py:600
ndwi_stress_latest = float(np.clip((1.0 - moisture_values[-1]) / 2.0, 0.0, 1.0))
```

Threshold-free, monotone decreasing, explicitly tagged `calibration_status: "uncalibrated"`, `operational: false`, `thresholds: {}`. Note it uses `moisture_values[-1]` only — the whole 30-day series informs the mean, min and trend, but the stress carrying 45 % of the weight is a single bin.

### 2.5 Clipping and saturation — did Jawor saturate?

**No, not at the stress mapping. Yes, at the normaliser.** This distinction is the single most important finding in the satellite layer.

- The affine map reaches 1.0 only at `m = −1.0`. Jawor's −0.0624 gives 0.5312 — nowhere near clipping. No realistic canopy value saturates it. **(FACT)**
- The 1.000 is produced entirely by `normalise_stress_across_network`. Jawor holds the network maximum, so `(v − lo)/(hi − lo) = 1.0` identically. **(FACT)**
- Many raw values therefore *cannot* map to 1.0 — but exactly one node maps to 1.0 on every run, whatever the absolute conditions. Saturation is guaranteed, not earned. **(FACT)**

| Node | raw moisture | stress_latest | normalised | Δ vs Jawor (raw) | Δ vs Jawor (norm) |
|---|---|---|---|---|---|
| **jawor** | −0.0624 | 0.5312 | **1.0000** | — | — |
| rychtal | 0.0350 | 0.4825 | 0.7645 | 0.0487 | 0.2355 |
| zlotoryja | 0.0432 | 0.4784 | 0.7443 | 0.0528 | 0.2557 |
| szklarska | 0.3506 | 0.3247 | 0.0000 | 0.2065 | 1.0000 |

The whole network occupies a raw stress band of width 0.2065. Min-max stretches it to [0, 1] — amplification of **4.84×**. A 0.0487 raw gap between Jawor and the runner-up becomes a 0.2355 gap in the number multiplied by 0.45.

### 2.6 Is the normalisation global, local, or population-dependent?

**Population-dependent, recomputed from scratch on every run, over whichever nodes returned data. (FACT)** No fixed reference, no historical baseline, no climatological anchor. Verified consequences:

- Drop Jawor and rychtal jumps 0.7645 → 1.0000, amplification rising 4.84× → 6.34×.
- A node's score changes when *other* nodes change, with no change in its own forest.
- A uniformly wet region and a uniformly parched region produce the same top-node score of 1.000.
- The single guard is `if span < 0.01: return raw values` — a 1 %-of-scale escape hatch this run's span of 0.2065 never approaches.

**The mapping is not persisted.** Neither `ndwi_sentinel.json` nor `risk_scores.json` records the `lo`/`hi` used, so a published score cannot be inverted back to a physical moisture value.

### 2.7 Does −0.0624 really mean extreme stress?

**What v5 computes.** −0.0624 is the cloud-free bbox mean of `(B8A−B11)/(B8A+B11)` over a ~5 km box, for the P10D bin beginning 2026-08-05, at 100 % valid-pixel fraction. It is the lowest such value in the network. Within the code's own uncalibrated mapping it is the network's driest reading, and that is the entire content of the claim. **(FACT)**

**Whether it means extreme stress: NOT PROVEN FROM CODE.** No threshold table, no reference distribution, no seasonal baseline, no validation set anywhere in `/v5`. `SATELLITE_STRESS_CONFIG.thresholds` is an empty dict and `calibration_status` is `"uncalibrated"` at every layer. The code does not claim otherwise — it declines to.

Two structural cautions provable from the code alone:

- Jawor's series has **n = 2** observations where 30 of 34 nodes have 3. Its 2026-07-26 bin was dropped by the cloud filter. A single bin determines a 45 %-weight input with no consistency check available. **(FACT)**
- The bbox is a fixed 0.045° square around the district centroid with **no land-cover mask**. Water, bare soil, harvested fields and settlement inside the box all depress B8A−B11 exactly as canopy drying does. Nothing in `/v5` distinguishes them. **(FACT)** — a plausible alternative explanation for a persistent single-node outlier; **UNKNOWN** whether it applies to Jawor without inspecting the imagery.

---

## 3. Cache Freshness Audit

| Question | Answer from code |
|---|---|
| What does "fresh" mean? | Solely `datetime.fromisoformat(entry["expires_at"]) > now`, plus schema and query-fingerprint match. `sentinel_v1.py:224` |
| Does it mean fresh acquisition? | **No.** The classifier never opens `result["dates"]` or `result["fetch_date"]`. |
| Only that TTL has not lapsed? | **Yes — exactly that, and nothing more.** |
| TTL in v5? | 10 days. `DEFAULT_CACHE_TTL_DAYS = 10`, overridable by `--cache-ttl-days` or `SENTINEL_CACHE_TTL_DAYS`. |
| Can one observation persist for days? | **Yes — up to 10 days of runs, and longer if the last cloud-free bin predates the fetch.** |
| Is acquisition time preserved? | **Yes**, in `result["dates"]` and `result["fetch_date"]`, retained through `ndwi_sentinel.json` and `nodes_enriched.json`. |
| Does the risk report show it? | **No.** `_save_results` (`fusion_v1.py:488`) copies `ndwi_latest` but no date field. `risk_scores.json`, `alerts.json` and `final_map.html` contain no acquisition timestamp. |
| Can cache hold a CRITICAL alert open? | **Yes, and it demonstrably did.** See below. |

### 3.1 The three-day scenario, answered from the repository

Four committed runs of `ndwi_sentinel.json` are in git history, and the Jawor record is byte-identical across all of them:

| Run `generated_at` | Commit | `fetch_date` | Acquisition bins | `ndwi_latest` | Alert tier |
|---|---|---|---|---|---|
| 2026-08-15T21:47 | `6cac72c` | 2026-08-15T20:29:35.746039 | 07-16, 08-05 | −0.0624 | — |
| 2026-08-16T12:45 | `c2e47fe` | 2026-08-15T20:29:35.746039 | 07-16, 08-05 | −0.0624 | — |
| 2026-08-19T09:38 | `f6a9ab1` | 2026-08-15T20:29:35.746039 | 07-16, 08-05 | −0.0624 | HIGH 0.6894 |
| **2026-08-23T13:04** | working tree | 2026-08-15T20:29:35.746039 | 07-16, 08-05 | −0.0624 | **CRITICAL 0.7528** |

**FACT.** Four pipeline runs spanning eight days rest on **one** Sentinel snapshot, fetched once on 2026-08-15, whose newest acquisition is 2026-08-05 — 18 days old on the audit date. These are **not** independent observations. Each run stamps a new `generated_at`, so the artifact looks freshly produced.

**FACT.** Across those runs `ndwi_stress` stayed pinned at 1.00 while the tier moved HIGH → CRITICAL. The escalation came entirely from the weather-driven terms (`drought_days` 0→1 and FWI), not from any new satellite information.

### 3.2 The step still reported success

```json
// v5/topology/pipeline_steps/sentinel.json — this run
"status":               "SUCCESS",
"duration_seconds":     1.607,
"coverage_percent":     100.0,
"api_requests":         0,
"cache_hits":           34,
"cache_classification": {"fresh": 34, "stale": 0, "missing": 0, "invalid": 0, "legacy": 0}
```

A 1.6-second step making zero HTTP calls reports 100 % satellite coverage. `coverage_percent` is `ok / len(nodes) * 100` where `ok` counts nodes holding *any* result, cached or live (`sentinel_v1.py:1105`). It measures cache completeness and is published under a name that reads as data currency.

**Credit where due:** the cache design is otherwise strong. The fingerprint covers node position, period, radius, collection, aggregation interval, resolution, cloud limit, index definitions and an evalscript SHA-256, so a config change correctly invalidates. Legacy pickles require an explicit `--legacy-sentinel-cache` flag and are force-flagged `legacy_provisional`. `stale-if-error` and `stale-if-offline` paths tag their results. The gap is narrow and specific: **nothing anywhere compares acquisition age to wall-clock time.**

---

## 4. QTE / Bridge Audit

### 4.1 Encoding

Five classical scalars become five `Ry` rotation angles in [0, π] (`encode_node`, `qte_v1.py:486`). Each qubit is one feature; no superposition of alternatives, no data re-uploading.

| Qubit | Feature | Encoding | Jawor input | θ (rad) |
|---|---|---|---|---|
| **q0 — ancilla** | moisture, previous bin | `clip((1−m)/2)·π` | −0.003 | 1.5755 |
| q1 | moisture, latest bin | `clip((1−m)/2)·π` | −0.0624 | 1.6688 |
| q2 | wind_max | `clip((v−4)/6)·π` | 20.3 m/s | 3.1416 (sat.) |
| q3 | temp_max | `clip((v−15)/15)·π` | 19.0 °C | 0.8378 |
| q4 | drought_days | `clip(v/30)·π` | 1 | 0.1047 |

### 4.2 Gates, measurement, and what actually executes

Declared circuit: `Ry`×5 → `CZ(0,1)` → `CZ(1,2)` → measure q0 → conditional `CZ(2,3)` → `CZ(3,4)` → read ⟨Z⊗Z⟩ on the four adjacent pairs. Two engines implement it.

**The Qiskit path — the one that ran** (`backends: ["qiskit_statevector"]`) — does *not* execute the dynamic circuit it builds. Lines 351–375 assemble `qc` with a real `measure` and `if_test`; that object is then abandoned, never transpiled and never run. The comment concedes it exists "only to validate the dynamic circuit compiles correctly", but no compile or validation call is made on it. **It is dead code. (FACT)**

What actually computes the result is a two-branch analytic expansion at lines 383–445: build `sv0` without the bridge CZ, `sv1` with it, then return `P(0)·⟨ZZ⟩₀ + P(1)·⟨ZZ⟩₁`. There are **no shots**. `QTE_N_SHOTS = 256` in `config.py` is unused on this path.

### 4.3 The bridge reduces to a sign test

Because `CZ` is diagonal it cannot change any measurement probability, so `prob_ancilla_1` collapses to the bare rotation:

```
bridge_rate  = sin²(θ_past / 2)
bridge_fired = bridge_rate > 0.5
             ⟺ θ_past > π/2
             ⟺ clip((1 − m_past)/2) > 0.5
             ⟺ m_past < 0
             ⟺ ndwi_values[-2] < 0
```

Verified numerically to < 1e-12, and **against every node in the run: 34/34 agreement, 0 mismatches.** The entanglement, the mid-circuit measurement and the conditional bridge contribute nothing to the firing decision.

**The description is not supported.** "Sequential dry→wind pattern confirmed 🔥" (`fusion_v1.py:242`) claims a two-stage temporal pattern. In the implementation:

- Wind never reaches the ancilla. Sweeping wind_max across 0, 5, 20.3 and 40 m/s leaves `bridge_rate` unchanged to within 1e-12. **(FACT)**
- The *current* moisture q1 does not affect it either — only `ndwi_values[-2]`.
- The bridge gates whether `CZ(2,3)` is applied, which does shift `zz_23` — but that is downstream of a decision already made without wind.
- The two bins are 20 days apart for Jawor, so "last week" in the docstring (`qte_v1.py:25`) is wrong by a factor of ~3.

A defensible description: *"the previous cloud-free composite had negative NIR-SWIR moisture."*

### 4.4 Deterministic or probabilistic?

| Backend | Method | bridge_rate | Deterministic? | Jawor outcome |
|---|---|---|---|---|
| **qiskit_statevector** | analytic, 2 branches | 0.502356 | Yes, bit-exact | fired (margin +0.0024) |
| numpy_statevector | 256 sampled shots | 0.453 – 0.543 | Per-seed only | **fires for 3 of 8 seeds** |

Measured at Jawor's exact operating point: seeds 20260601, 2 and 5 fire; seeds 1, 3 and 4 do not. Because `bridge_fired` is worth 0.10 inside `compute_qte_score` *and* 0.08 in fusion, the two backends differ by ~0.11 on the final score — **39× the 0.0028 margin. Jawor's tier is a function of which simulator was installed. (FACT)**

Shot artefacts are real on the NumPy path: with true p = 0.5024, 256 shots give σ ≈ 0.031, so `bridge_fired` is near a coin flip. The pipeline seeds per node (`random_seed + index`, `qte_v1.py:619`), so results are reproducible for a fixed node ordering — but *reproducible* is not *stable*: inserting a node upstream shifts every later node's seed.

### 4.5 Is there a classical baseline?

**None exists in `/v5`. (FACT)** No classical comparator, no ablation, no A/B harness. The docstring claim that the bridge "captures what RF cannot see" (`fusion_v1.py:19`) is untested here.

**A baseline is trivially available, because §4.3 already derived it:**

```python
bridge_fired = ndwi_values[-2] < 0
zz_ab        = cos(theta_a) * cos(theta_b)      # for the unbridged pairs
qte_score    = 0.30*abs(zz_12) + 0.25*abs(zz_34) + 0.20*max(0, -zz_01) \
             + 0.15*abs(zz_23) + 0.10*bridge_fired
```

Every ⟨Z⊗Z⟩ on a product state of `Ry` rotations is a product of cosines; the diagonal `CZ` gates leave all Z-basis expectations untouched. *(INFERENCE: the QTE is a closed-form polynomial in five cosines, computable in microseconds without Qiskit.)* This is a statement about what **this** code does, not a judgement of quantum methods in general — a circuit with non-diagonal entangling gates, or measurement in a rotated basis, would not collapse this way.

---

## 5. Jawor Forensic Replay

Reconstructed independently from cache files and node data, then compared against `risk_scores.json`.

| # | Step | Computation | Result | Published | ✓ |
|---|---|---|---|---|---|
| 1 | Raw satellite | B8A/B11 bin 2026-08-05, 100 % valid px | −0.0624 | −0.0624 | ✓ |
| 2 | Stress mapping | `(1 − (−0.0624)) / 2` | 0.5312 | 0.5312 | ✓ |
| 3 | **Min-max normalise** | `(0.5312 − 0.3247) / 0.2065` | **1.0000** | 1.0000 | ✓ |
| 4 | Trend slope | `polyfit([0,1], [−0.003, −0.0624])` | −0.0594 | −0.0594 | ✓ |
| 5 | Trend signal | `min(1, max(0, 0.0594 × 3))` | 0.1782 | — | — |
| 6 | Bridge gate | `sin²(1.5755/2) = 0.502356 > 0.5` | fired | fired | ✓ |
| 7 | ZZ terms | zz01 0.000461 · zz12 0.097861 · zz23 −0.669131 · zz34 0.665465 | — | — | ✓ |
| 8 | QTE score | `.30(.0979)+.25(.6655)+.20(0)+.15(.6691)+.10` | 0.3961 | 0.3961 | ✓ |
| 9 | FWI proxy | `.22(.186)+.20(.333)+.14(.36)+.14(1.0)+.17(.333)+.08(.022)+.05(.975)` | 0.4052 | 0.4052 | ✓ |
| 10 | Base fusion | `.45(1.0)+.30(.3961)+.15(.4052)+.10(.1782)` | 0.6474 | 0.6474 | ✓ |
| 11 | Ecosystem × | eco = "mixed" → 1.00 | 0.6474 | 1.00 | ✓ |
| 12 | Bridge bonus | +0.08 | 0.7274 | 0.08 | ✓ |
| 13 | Network prop. | `+0.05 × mean(15 neighbours) = 0.05 × 0.5071` | 0.7528 | 0.0254 | ✓ |
| 14 | **Final / tier** | `clip(0.75278)` → > 0.75 | **0.7528 CRITICAL** | 0.7528 CRITICAL | ✓ |

**The score reproduces exactly. 0.7528 is what this implementation must produce from these inputs.** Nothing is missing, nothing unexplained, no hidden state.

### 5.1 Counterfactuals

| Scenario | Score | Tier | Δ |
|---|---|---|---|
| **As published** | 0.7528 | CRITICAL | — |
| Past moisture −0.003 → +0.001 (bridge silent) | 0.6428 | HIGH | −0.1100 |
| Absolute stress 0.5312 instead of the rank | 0.5418 | MODERATE | −0.2110 |
| Both corrections together | 0.4318 | LOW | −0.3210 |

The two design decisions audited in §2.5 and §4.3 are jointly worth **0.32 of score and three full tiers** on this node.

### 5.2 Ground conditions at Jawor on the run date

**Jawor received 91.6 mm of rain in the seven days before the run — the highest of all 34 nodes** (42.1 mm on 08-21, 20.8 mm on 08-22), at `temp_max 19.0 °C` and `drought_days 1`. Its satellite reading predates that rainfall by 16 days.

Two independent inputs already encoded the wetness — `rain_total` and `drought_days` — and the fusion weighted them at 0.08 × (1/45) ≈ 0.0018 and 0.05 × 0.975 ≈ 0.049 inside a 0.15-weight FWI term. The 45 %-weight channel could not see it at all.

*(INFERENCE: the single most alarming node in this run is, on the two directly-measured wetness variables available to the pipeline, the wettest node in the network. This is a consistency failure between channels, not a numerical error.)*

---

## 6. Fusion Audit

```python
# v5/qhdalabs_wildfire_fusion_v1.py:298–304
base     = W_NDWI*ndwi_stress + W_QTE*qte_score + W_FWI*fwi + W_TREND*trend_s
modified = base * eco_mult
modified += BRIDGE_BONUS if bridge_fired else 0.0
modified += NETWORK_COEFF * net_stress
final    = float(np.clip(modified, 0.0, 1.0))
```

| # | Question | Finding |
|---|---|---|
| 1 | Do the weights sum correctly? | **Yes.** 0.45 + 0.30 + 0.15 + 0.10 = 1.00 exactly. `config.py` and `fusion_v1.py` agree, though the constants are duplicated with no shared import — a drift hazard (F-12). |
| 2 | Does the order of operations make sense? | **Partly.** Blend-then-multiply is defensible, but the multiplier lands on all four terms (see 3) and the two additive bonuses land *outside* it, so a spruce node's bridge bonus is worth 1.63× a pine node's in relative terms. Undocumented. |
| 3 | Does the ecosystem multiplier over-amplify? | **Yes, unintentionally.** `base * eco_mult` scales FWI, trend and QTE as well as the canopy term. Fuel flammability is a property of the stand, not of today's wind or a quantum correlation — yet a pine node's wind contribution is inflated 30 %. Verified: identical inputs give pine = mixed × 1.30 on the *whole* score. |
| 4 | Can the bridge alone force CRITICAL? | **Yes — and on this run it did.** 0.7528 − 0.08 = 0.6728, i.e. HIGH. `BRIDGE_BONUS` = 0.08 exceeds half the 0.15 tier width, so a single boolean derived from the sign of one cached float can move any node a full tier. |
| 5 | Feedback loop or spatial amplification? | **No feedback loop** — `net_stress` reads `ndwi_stress_normalised`, never a fusion output; one pass, idempotent, hard-bounded at 0.05. **But amplification is real:** neighbours are min-max ranks, so the term inherits the 4.84× stretch, and with `NEIGHBOR_KM = 60` Jawor has 15 neighbours — 45 % of the network averaged into one "local" signal. |
| 6 | Does clipping hide differences? | **Not on this run** (`clipped_to_one_count: 0`). Structurally the headroom is 1.00 × 1.30 + 0.08 + 0.05 = **1.43**, so 43 % of the top range can collapse to 1.0. The real hidden saturation is upstream: `ndwi_stress = 1.000` is pinned every run and `saturation_diagnostics` does not measure it. |
| 7 | Are the thresholds calibrated? | **Arbitrary.** 0.45 / 0.60 / 0.75 are evenly spaced at 0.15 with no derivation, reference incident set, or validation anywhere in `/v5`. The code says so: `calibration_status: "uncalibrated"`, `operational_validity: false`. The weights 0.45/0.30/0.15/0.10 are likewise unsourced. |

### 6.1 The falsy-zero defect

```python
# v5/qhdalabs_wildfire_fusion_v1.py:271 — and again at :288 for neighbours
ndwi_stress = float(
    node.get("ndwi_stress_normalised") or node.get("ndwi_stress_latest", 0.0)
)
```

Min-max guarantees the least-stressed node normalises to exactly `0.0`, which is falsy, so the `or` chain discards it and substitutes the *raw absolute* value. Confirmed: szklarska has `ndwi_stress_normalised = 0.0` but is scored with **0.3247**.

Worse, the two code paths disagree. In its own score szklarska counts as 0.3247; as a neighbour at line 288 the same `or 0.0` resolves to 0.0. **The same node has two different stress values in the same run.** Fires on exactly one node per run — always the wettest, the one whose score matters least, which is why it has gone unnoticed.

### 6.2 Tier margins across the network

Sixteen of 34 nodes sit within one bridge-bonus (0.08) of a tier boundary. piszowice (0.5897) and zgorzelec (0.5860) are within 0.014 of HIGH. The banding is far finer than the demonstrated instability of its inputs.

---

## 7. Ignition Layer Audit

### 7.1 Does ignition_score affect the final score?

**No. Unambiguously not. (FACT)** In `compute_risk_score`, `final` is computed at line 305; `ignition` is not fetched from the map until line 307. The dependency runs strictly the other way:

```python
# v5/qhdalabs_wildfire_fusion_v1.py:305–320
final = float(np.clip(modified, 0.0, 1.0))          # ← score is already fixed

ignition = (ignition_map or {}).get(nid)
fei  = compute_fei(final * 100.0, ignition_score)    # ← final is the INPUT
qies = compute_qies(final * 100.0, ignition_score)
```

Verified two ways: recomputing `base_score` from the four published signals matches all 34 nodes to 5×10⁻⁴ with no ignition term; and `fusion_weights` contains no ignition entry. Passing `ignition_map=None` leaves the score unchanged, setting only a `DEGRADED` status and an `ignition_invalid_or_unavailable` flag.

### 7.2 Then why is it computed, and where is it used?

A genuine, well-built layer: six weighted sublayers (roads 0.25, railways 0.20, powerlines 0.15, tourism 0.15, agriculture 0.15, historical FIRMS KDE 0.10, asserted to sum to 1.0 at import), sourced from OSM, BDOT10k, LPIS/CLC and NASA FIRMS, with fingerprinted caches, a `MIN_COVERAGE_FOR_FUSION = 0.70` gate and last-known-good protection. Its outputs feed exactly three places: `ignition_scores.json`; the diagnostic `fei` and `qies` fields; and the `DEGRADED`/`SUCCESS` pipeline status. **None changes the alert tier.**

For Jawor: `ignition_score 33.5` (low), coverage 100 %, `FEI 50.22`, `QIES 19.12` — both verified reproducible. Note the direction: FEI = √(75.28 × 33.5) = 50.22 pulls the picture down substantially, and QIES at 19.12 flags a strongly imbalanced pair. **The two indices that do incorporate ignition both read far less alarming than the tier that gets published.**

### 7.3 Is "Unified Wildfire Risk" misleading?

**Yes. (INFERENCE)** The banner at `fusion_v1.py:441` and the header claim to integrate "all three signal layers into one unified wildfire risk score" (line 9). The score integrates satellite, QTE, weather and trend — all readiness channels. Ignition, the layer whose own module docstring correctly calls it orthogonal, is excluded from the number that drives alerts.

| Concept | Present in v5? | Where | Assessment |
|---|---|---|---|
| **A · Susceptibility / fire readiness** | **Yes** | `final_score` | What the alert tier actually measures — and `compute_fei`'s own signature names it `readiness`, which is the honest term. |
| **B · Ignition probability** | **Yes, cleanly** | `ignition_score` | Well-separated, honestly documented, properly gated on coverage. The strongest-designed layer in v5. |
| **C · Expected wildfire risk** | **Approximated, unused** | `fei` | √(R·I) is the right shape for AND-logic, but the extremity bonuses are unsourced and nothing consumes FEI for alerting. |

The architecture gets the separation right; the naming and the alert wiring do not. Renaming `final_score` to `readiness_score` — matching the parameter name the code already uses internally — would resolve most of this at near-zero cost.

---

## 8. Tests and Reproducibility

**Existing suite: 49 tests, all passing** (21 s). Coverage is strong on I/O robustness — FIRMS resumption, checksum mismatch, atomic writes, secret masking, cache fingerprinting, interpreter re-exec, EFFIS contracts. It is thin on scoring semantics: no test asserts what a score *means*, and two tests actively mask findings in this report.

| Existing test | Gap |
|---|---|
| `test_numpy_qte_is_deterministic_for_seed` | Asserts same-seed reproducibility only. Says nothing about cross-seed stability of `bridge_fired`, which is where the instability lives. |
| `test_qiskit_and_numpy_qte_are_compatible` | Compares `bridge_rate` at `abs=0.08`. Near p = 0.5 that tolerance is wide enough to hide a boolean flip that moves a node a full tier. |

### 8.1 Tests added

One new file, **`v5/tests/test_forensic_audit.py` — 40 tests, all passing, ruff-clean**. No production logic touched. Combined suite: **89 passed**.

Tests that pin a defect are named and docstringed `DEFECT:`, so a future fix fails loudly rather than silently changing alert semantics.

| # | Area | Tests | Notable assertion |
|---|---|---|---|
| 1 | NDWI saturation | 5 | Min-max always saturates exactly one node; amplification > 4×; removing the max promotes the runner-up to 1.000 |
| 2 | Cache vs acquisition freshness | 3 | An 8-day-old entry holding an 18-day-old acquisition classifies `fresh`; alerts carry no date field |
| 3 | Jawor forensic replay | 2 | Full replay to 5×10⁻⁴; CRITICAL margin is smaller than the bridge bonus |
| 4 | QTE determinism | 5 | `bridge_rate = sin²(θ/2)` to 1e-12; wind-independence to 1e-12; NumPy backend disagrees across 8 seeds |
| 5 | Bridge threshold edges | 4 | The boundary sits exactly at `m_past = 0`, and floating point puts 0.0 itself on the firing side |
| 6 | Network propagation | 2 | Idempotent, no feedback path, contribution bounded by `NETWORK_COEFF` |
| 7 | Missing ignition | 2 | `ignition_map=None` leaves the score identical; all 34 base scores reconstruct without an ignition term |
| 8 | Stale Sentinel data | 2 | 100 % coverage with 0 API requests; trend slope is per-bin, so cloud gaps inflate it |
| 9 | Extreme values | 5 | Unclipped headroom exceeds 1.0; eco multiplier scales weather too; soil defaults silently; a 92 mm week is invisible to FWI |
| 10 | Threshold boundaries | 10 | 0.4500 / 0.6000 / 0.7500 land in the *lower* tier — comparisons are strictly exclusive |

### 8.2 Reproducibility verdict

Strong on provenance: every run records `run_id`, `git_sha`, interpreter path, Python version, per-step manifests, data versions, cache-hit counts and API counts. Atomic writes throughout. The weakness is the two-backend QTE: `backends` and `backend_versions` are recorded, but the two engines can disagree on a tier and nothing warns when a node sits near p = 0.5.

---

## 9. Findings Ranked by Severity

### CRITICAL

#### F-01 — "NDWI stress 1.000" is a within-run rank published as an absolute measurement

- **Problem** — Fusion consumes `ndwi_stress_normalised`, a min-max rescale over whichever nodes returned data. Exactly one node scores 1.000 every run regardless of absolute conditions, and it contributes the full 0.45 weight.
- **Evidence** — `sentinel_v1.py:672` `normalise_stress_across_network`; `fusion_v1.py:271`. Raw span 0.3247–0.5312 stretched to [0,1] = 4.84× amplification. Jawor's absolute stress is 0.5312, not 1.0.
- **Impact** — Worth **0.211 of score and two tiers** on Jawor. The system cannot report "no node is stressed" — it always nominates a worst node. Cross-run and cross-region comparison is meaningless. The `lo`/`hi` pair is not persisted, so scores cannot be inverted to physical values.
- **Fix** — Feed `ndwi_stress_latest` (absolute) to fusion and publish the rank alongside as a separate diagnostic. If normalisation is wanted, anchor it to a fixed historical reference distribution, and always persist the mapping bounds in `ndwi_sentinel.json`.

#### F-02 — A single 18-day-old acquisition has driven four runs, reported as SUCCESS at 100 % coverage

- **Problem** — "Fresh" means only that the 10-day TTL has not lapsed. Nothing compares acquisition date to wall clock. A step with 0 API requests publishes `coverage_percent: 100.0` and `status: SUCCESS`.
- **Evidence** — `_load_cache_entry` `sentinel_v1.py:224`. Git history: runs on 08-15, 08-16, 08-19 and 08-23 all carry `fetch_date 2026-08-15T20:29:35.746039` and acquisition `2026-08-05`. Sentinel step: 1.6 s, 0 requests, 34 cache hits.
- **Impact** — The 45 %-weight channel was frozen while the tier escalated HIGH → CRITICAL on weather alone. A CRITICAL alert can persist for the full TTL with no new satellite data, and no consumer can detect it.
- **Fix** — Add `max_acquisition_age_days` alongside the TTL and classify on `max(dates)`, not `expires_at`. Emit a `stale_acquisition` quality flag and downgrade to `DEGRADED` past the threshold. Split the manifest into `coverage_percent` and `fresh_acquisition_percent`.

#### F-03 — The quantum bridge is a sign test on one cached float, and the label misstates it

- **Problem** — `bridge_rate = sin²(θ_past/2)` exactly, so `bridge_fired ⟺ ndwi_values[-2] < 0`. Wind never enters the decision, yet the alert reads "Sequential dry→wind pattern confirmed 🔥" and the header claims it "captures what RF cannot see".
- **Evidence** — `_run_qiskit_qte` `qte_v1.py:336`; verified to 1e-12 and confirmed on 34/34 nodes with zero mismatches. Wind swept 0→40 m/s leaves `bridge_rate` unchanged. The dynamic circuit built at lines 351–375 is never executed.
- **Impact** — Worth 0.18 of score (0.10 in `qte_score` + 0.08 in fusion). On Jawor it fires on `m_past = −0.003`, i.e. 0.003 from a sign flip, and alone carries the node from HIGH to CRITICAL.
- **Fix** — Relabel to what it tests. If a genuine dry→wind sequence is wanted, make the gate depend on both a lagged dryness term and a subsequent wind term. Delete the unexecuted circuit or actually run it.

### HIGH

#### F-04 — The alert tier depends on which simulator backend is installed

- **Problem** — Qiskit computes `bridge_rate` analytically; NumPy samples 256 shots. At Jawor's p = 0.5024 the NumPy path fires for 3 of 8 seeds (σ ≈ 0.031).
- **Evidence** — Measured at Jawor's exact angles: seeds 20260601/2/5 fire, seeds 1/3/4 do not. `test_qiskit_and_numpy_qte_are_compatible` compares at `abs=0.08` and cannot catch it.
- **Impact** — Backend choice shifts the final score by ~0.11 — 39× Jawor's 0.0028 margin. Two installations of the same commit legitimately disagree on CRITICAL.
- **Fix** — Make the NumPy path analytic too (`bridge_rate = sin²(θ/2)`) so the engines are identical by construction; keep sampling only behind an explicit flag. Emit a `bridge_near_threshold` flag whenever `|rate − 0.5| < 0.05`.

#### F-05 — Soil moisture is null on all 34 nodes and silently becomes a hardcoded 0.20

- **Problem** — Open-Meteo returns nothing for `soil_moisture_0_to_1cm`; NaN is sanitised to null and `_fwi_from_weather` substitutes 0.20 via an `or` default. No flag, no warning, no manifest entry.
- **Evidence** — `topology_v1.py:475` requests the variable; `soil_min` and `soil_mean` are null for every node on every day in `nodes_enriched.json`. `fusion_v1.py:127`: `soil = float(latest.get("soil_min") or 0.20)`.
- **Impact** — The 0.20 weight — the second-largest FWI term — is a constant 0.0667 for every node, contributing nothing discriminative while appearing to. A drought signal channel is dead and the pipeline reports SUCCESS.
- **Fix** — Use the variable the archive API actually serves (ERA5 exposes `soil_moisture_0_to_7cm`). Until then, raise a quality flag and drop the term with renormalised weights rather than substituting a constant.

#### F-06 — FWI sees only the latest day's rain, so a 92 mm week is invisible

- **Problem** — `rain_pen = max(0, 1 − rain/4)` reads `latest.rain_total` alone. Fourteen days of history are fetched and stored, then ignored.
- **Evidence** — `fusion_v1.py:132`. Jawor: 0.1 mm today → `rain_pen = 0.975`, despite 91.6 mm in the preceding seven days. `drought_days` registers the wetness but is weighted 0.08 × (1/45) ≈ 0.0018.
- **Impact** — The wettest node in the network is the only CRITICAL one. No antecedent-precipitation memory exists anywhere in the fusion.
- **Fix** — Replace the single-day term with a decayed antecedent precipitation index over the 14-day series already in `weather_history`.

### MEDIUM

#### F-07 — The wettest node gets two contradictory stress values in the same run

- **Problem** — Min-max guarantees a 0.0 for the least-stressed node; 0.0 is falsy, so `x or y` substitutes the raw absolute value at `fusion_v1.py:271` — but the neighbour path at line 288 resolves to 0.0.
- **Evidence** — szklarska: `ndwi_stress_normalised = 0.0`, scored with 0.3247, counted as a neighbour at 0.0.
- **Impact** — Inflates the wettest node's score by the full raw value every run and makes network propagation inconsistent with self-scoring. Low blast radius today only because it hits the node that matters least.
- **Fix** — Replace both `or` chains with explicit `is None` checks. The same pattern also affects `ndwi_trend_14d` and `soil_min`.

#### F-08 — ndwi_trend_14d is neither a 14-day window nor a per-day rate

- **Problem** — The slope is fitted against `np.arange(len(values))` — bin index, not date — over a 30-day window of P10D composites. Cloud gaps are invisible to the fit.
- **Evidence** — `sentinel_v1.py:592`. Jawor's two bins are 20 days apart but treated as one step: slope −0.0594, per-day −0.00297. rychtal's bins are 10 days apart: slope −0.0455, per-day −0.00455.
- **Impact** — Cloud gaps inflate trend magnitude. Jawor gets the network's largest trend signal (0.1782) while actually drying *more slowly* per day than nodes ranked below it — the channel inverts its own ranking.
- **Fix** — Fit against real dates in days and rename to `moisture_trend_per_day`. Refuse a trend for n < 3 rather than returning a two-point difference.

#### F-09 — The ecosystem multiplier scales weather, trend and QTE, not just fuel

- **Problem** — `modified = base * eco_mult` applies the fuel factor to all four terms. Additive bonuses sit outside it, so their relative worth varies inversely with the multiplier.
- **Evidence** — `fusion_v1.py:301`; verified — identical inputs give pine = mixed × 1.30 on the whole score. Headroom becomes 1.00 × 1.30 + 0.13 = 1.43.
- **Impact** — A pine stand's wind reading is inflated 30 % for reasons of species composition. 43 % potential clipping range.
- **Fix** — Apply the multiplier to the fuel-state terms only, or fold it in as a weighted term rather than a global scalar.

#### F-10 — "Unified Wildfire Risk" excludes the ignition layer it implies

- **Problem** — The alert tier derives from `final_score`, which is a readiness index. Ignition, FEI and QIES are computed downstream and never affect it.
- **Evidence** — `fusion_v1.py:305–320`; `fusion_weights` has no ignition entry; all 34 base scores reconstruct without one. Jawor: ignition 33.5, FEI 50.22, QIES 19.12 — all substantially less alarming than the published tier.
- **Impact** — Consumers reasonably read CRITICAL as "fire likely here", when it means "this stand would burn well if something ignited it". Two better-informed indices are computed and then not used for alerting.
- **Fix** — Rename `final_score` to `readiness_score` — the term `compute_fei` already uses internally — and either alert on FEI or state plainly that alerts are readiness-only.

### LOW

#### F-11 — Upstream quality flags do not reach the published score

- **Problem** — Sentinel attaches `vegetation_moisture_index_uncalibrated`, `cloud_masked`, `sentinel_b11_is_not_gao_1_24um` and records `n_observations`. Fusion emits a fixed `calibration_flags` list and ignores all of it.
- **Evidence** — `fusion_v1.py:329`: `calibration_flags = ["satellite_stress_mapping_uncalibrated"]`, hardcoded. Jawor's `data_quality_flags` is empty despite n = 2.
- **Impact** — A node scored from two cloud-limited observations is indistinguishable from one scored from a full series.
- **Fix** — Propagate upstream flags and add `low_observation_count` below a threshold.

#### F-12 — Fusion constants are duplicated between config.py and fusion_v1.py

- **Problem** — Weights, bonuses, thresholds and eco multipliers are defined in both files. `fusion_v1.py` does not import `config`, so the config copy is inert.
- **Evidence** — `config.py:41–61` vs `fusion_v1.py:95–112`. Values currently agree.
- **Impact** — Editing `config.py` — the file whose name invites it — changes nothing. Silent drift risk.
- **Fix** — Import from `config`, or delete the dead block.

### INFORMATIONAL

#### F-13 — A 60 km neighbour radius averages 45 % of the network into a "local" signal

- **Problem** — `NEIGHBOR_KM = 60` gives a mean degree of 9.5 and a maximum of 15 across 34 nodes. Jawor's "neighbour stress" is the mean of 15 of the other 33.
- **Evidence** — `graph.json`, `neighbor_threshold_km: 60.0`.
- **Impact** — Bounded at 0.05 so the effect is small, and there is no feedback loop. But the term measures a regional average rather than local spatial correlation, and it inherits the min-max stretch from F-01.
- **Fix** — Distance-weight the neighbour mean, or reduce the radius so the term is genuinely local.

#### F-14 — saturation_diagnostics measures the one saturation that is not occurring

- **Problem** — The block tracks `unclipped_score > 1.0` and reports 0. Meanwhile `ndwi_stress` is pinned at exactly 1.000 for one node every run, which is unmeasured.
- **Evidence** — `fusion_v1.py:491`; `saturation_diagnostics.clipped_to_one_count: 0`.
- **Impact** — The diagnostic reads reassuring while the real saturation sits upstream of it.
- **Fix** — Add per-signal saturation counts and record the normalisation `lo`/`hi` in the same block.

---

## 10. Final Interpretation of the Jawor Alert

### FACT

- The score 0.7528 is arithmetically correct for this implementation. Every component replays to within 5×10⁻⁴. No bug in the computation, no cache corruption, no data-integrity failure.
- The tier is correct given the score: 0.7528 > 0.75.
- The satellite input is a single 2026-08-05 acquisition, fetched 2026-08-15, unchanged across four runs — 18 days old on the audit date.
- "NDWI stress 1.000" means Jawor held the network maximum. Its absolute uncalibrated stress is 0.5312.
- The bridge fired because `ndwi_values[-2] = −0.003 < 0`. Not because of wind, and not because of any sequence.
- Removing the bridge bonus gives 0.6428 → HIGH. Using absolute rather than ranked stress gives 0.5418 → MODERATE. Both gives 0.4318 → LOW.
- Jawor recorded 91.6 mm of rain in the seven days before the run — the highest of all 34 nodes — at 19 °C with `drought_days = 1`.
- The pipeline labels its own output `operational_validity: false`, `calibration_status: "uncalibrated"`, `recommendation_status: "research_only_uncalibrated"`, `drone_recommended: false`.

### INFERENCE

- The alert is **a correct execution of the implementation and simultaneously not evidence of elevated wildfire risk at Jawor on 2026-08-23**. Both halves hold: the code did exactly what it says, and what it says does not measure current fire danger.
- The escalation HIGH → CRITICAL between 08-19 and 08-23 was driven entirely by weather terms, with the 45 %-weight satellite channel frozen throughout.
- Jawor's persistent top rank across four runs reflects one cached snapshot, not a persistent physical condition. A stable low B8A−B11 at one site is at least as consistent with a fixed non-forest fraction inside the 5 km bbox as with drought.
- Two knife-edge margins — +0.0028 on the tier and +0.0024 on the bridge — mean the alert is not robust to any perturbation of its inputs, including the choice of simulator backend.
- The label "Unified Wildfire Risk" overstates scope: the score is a readiness index and excludes the ignition layer computed alongside it, whose own indices (FEI 50.22, QIES 19.12) read considerably less alarming.

### UNKNOWN

- Whether Jawor's canopy was genuinely water-stressed on 2026-08-05. **NOT PROVEN FROM CODE** — no thresholds, no baseline, no validation set exists in `/v5`.
- What the 5 km bbox actually contains. No land-cover mask is applied, so the fraction of non-forest surface is undetermined without inspecting the imagery.
- Whether the weights 0.45/0.30/0.15/0.10 and the thresholds 0.45/0.60/0.75 have any predictive validity. No derivation appears anywhere in the repository.
- Whether the missing 2026-07-26 bin was cloud-masked or genuinely absent, and how a third observation would have changed the trend.
- The true current moisture state. No acquisition after 2026-08-05 has been fetched.

### The four closing questions

**1 · Does the implementation do what it declares?**
Arithmetically yes; semantically no. The pipeline structure, manifests, atomic writes, caching and status contracts are well-built and behave as documented. But three of four fusion inputs carry names that misdescribe them: "NDWI stress" is a rank, "sequential dry→wind" is a sign test, "14-day trend" is a per-bin difference. The README is honest about index naming and calibration; the runtime field names and alert text are not.

**2 · Is the Jawor result correct?**
Correct relative to the implementation, exactly. Verified to 5×10⁻⁴ at every stage. If the goal was to test whether the code computes what it specifies, it passes.

**3 · Are there hidden artefacts?**
Yes, four. (a) Guaranteed rank saturation — one node always scores 1.000. (b) TTL freshness masking an 18-day-old acquisition behind a 100 %-coverage SUCCESS. (c) A quantum layer that reduces to a closed-form sign test, with a dynamic circuit that never executes. (d) A dead soil channel silently replaced by a constant. None is visible in the published outputs.

**4 · What is missing for predictive value?**
A labelled fire-occurrence set for Dolny Śląsk to fit weights and thresholds against; a fixed reference distribution to replace population-relative normalisation; acquisition-age gating so stale data degrades rather than passes; a land-cover mask on the bbox; antecedent precipitation in the FWI; a working soil channel; and a classical baseline to test whether the QTE adds anything over the closed form derived in §4.5.

---

### Closing note on scope

This repository declares itself research-grade and uncalibrated, and it does so in code — `operational_validity: false` on every score, `drone_recommended: false` on every alert, honest methodological caveats in `index_definitions()` and the README. Judged as a research pipeline it is carefully engineered, with unusually good provenance and cache discipline for a project at this stage.

The findings above are not a verdict on that ambition. They are the gap between what the identifiers and alert strings promise and what the arithmetic delivers — and that gap is worth closing precisely *because* the surrounding infrastructure is solid enough to carry a calibrated model once one exists.
