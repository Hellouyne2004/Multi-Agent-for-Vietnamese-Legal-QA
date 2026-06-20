# Retrieval Evidence Audit

- Audit date: 2026-06-19
- Inputs: `legal_qa_eval_100.jsonl`, `retrieval_predictions.jsonl`, `retrieval_100.json`, `failure_analysis.md`, evaluator and retriever code.
- No retrieval rerun and no LLM/API call were used.
- Verdict: **FAIL**. Doc Hit@5 and MRR pass, but both evidence coverage gates fail.

## Important Measurement Caveats

1. Each prediction contains five actually ranked chunks. Positions 6-15 are neighbor/parent context expansion with score `0`, so K=10 is not a true ranked Top-K experiment.
2. The evaluator appends metadata values to content in `retrieved_context_text()`. This does not change any hit in this snapshot, but content-only scoring should still be used going forward.
3. Reported MRR is conditional on hits because misses are dropped by `float_mean`. Standard MRR must assign zero to the two misses: `88.5/91 = 97.25%`, not `99.44%`.
4. Exact substring normalization removes Vietnamese accents but does not normalize punctuation, OCR corruption, `08/8`, units, paraphrases or row-level table relationships.

## A. Quality Gates

| Gate | Numerator/denominator | Value | Threshold | Status |
| --- | ---: | ---: | ---: | --- |
| Doc Hit@5 | 89/91 eligible doc-labelled cases | 97.80% | >=95% | PASS |
| Context Fact Coverage@5 | 110/186 facts, 90 cases | 59.14% | >=80% | **FAIL** |
| Full Context Fact Case Rate@5 | 47/90 evidence cases | 52.22% | >=70% | **FAIL** |
| Forbidden Fact In Context@5 | 4/90 evidence cases | 4.44% | <=5% | PASS, boundary risk |
| Standard MRR@5 | 88.5/91 doc-labelled cases | 97.25% | >=80% | PASS |
| Runtime error rate | 0/100 predictions | 0% | <=5% | PASS |
| Empty result | 2/100; only 1 is an eligible retrieval failure | 2% | no configured gate | MEASURED |
| Average retrieval latency | 53,739 ms / 100 | 537.39 ms | no configured gate | MEASURED |
| P95 retrieval latency | 100 latency samples | 526.55 ms | no configured gate | MEASURED |

The first case took 9,298 ms. Median is 430.5 ms and warm average excluding that case is about 448.9 ms. The average is therefore cold-start sensitive.

Diagnostic metadata metrics: Article Hit@5 `51/75 = 68.00%`; Clause Hit@5 `1/3 = 33.33%`; Point Hit is `N/A_NO_LABEL (0 eligible)`; Level Hit@5 `80/91 = 87.91%`. Five expected articles do not exist as article metadata in the corpus, so Article Hit's corpus-conditioned ceiling is `70/75`.

### Top-K Sensitivity

| Context | Doc hit | Fact coverage | Full cases | Forbidden cases | Standard MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ranked K=1 | 88/91 | 57/186 | 26/90 | 2/90 | 88/91 |
| Ranked K=3 | 89/91 | 103/186 | 42/90 | 4/90 | 88.5/91 |
| Ranked K=5 | 89/91 | 110/186 | 47/90 | 4/90 | 88.5/91 |
| Expanded context first 10 | 89/91 | 120/186 | 51/90 | 4/90 | 88.5/91 |

K=3 to K=5 adds only 7 facts and 5 full cases. Expansion positions 6-15 add 13 facts across six cases, but this is structural expansion rather than ranked retrieval.

## B. Slices, Weakest to Strongest

| Dimension | Slice | Doc hit | Fact coverage | Full cases | Forbidden cases |
| --- | --- | ---: | ---: | ---: | ---: |
| type | hallucination_trap | 2/2 | 0/2 | 0/1 | 0/1 |
| type | procedure | 10/10 | 6/17 | 2/10 | 0/10 |
| difficulty | easy | 16/16 | 11/25 | 8/16 | 0/16 |
| document/category | `45_2019_qh14` / labor | 56/57 | 59/127 | 20/57 | 0/57 |
| type | definition | 14/14 | 12/26 | 7/14 | 0/14 |
| level | article | 65/67 | 72/139 | 28/67 | 0/67 |
| table split | non-table | 77/79 | 86/162 | 36/79 | 0/79 |
| level | clause | 10/10 | 11/20 | 6/10 | 0/10 |
| type | direct_article | 52/54 | 68/117 | 27/54 | 0/54 |
| document/category | `luat_116_2025_qh15_pdf` / cybersecurity | 9/10 | 11/17 | 6/10 | 0/10 |
| difficulty | hard | 17/17 | 21/42 | 5/16 | 0/16 |
| difficulty | medium | 56/58 | 78/119 | 34/58 | 4/58 |
| document/category | `luat_109_2025_qh15_pdf` / tax | 12/12 | 16/18 | 10/12 | 0/12 |
| level | document | 2/2 | 3/3 | 2/2 | 0/2 |
| document/category | appendix / land_price | 12/12 | 24/24 | 11/11 | 4/11 |
| level / table split | table | 12/12 | 24/24 | 11/11 | 4/11 |
| requires_web | false | 89/91 | 110/186 | 47/90 | 4/90 |

`requires_web=true` has no cases. Router, general_chat, unsafe_request, out_of_scope and unsafe slices have no retrieval-evidence denominator and remain `N/A_NOT_APPLICABLE`.

## C. Top 15 Failure Cases

| Case | Expected evidence | Retrieved evidence | Missing/problem | Likely root cause |
| --- | --- | --- | --- | --- |
| `labor_wage_scale_027` | Article 93; thang/bang luong, dinh muc | Five appendix table chunks | `c00115` absent; wrong doc | `bang` falsely triggers table boost; Article 93 metadata also missing |
| `cyber_transition_previous_law_078` | Article 45; Law 86/2015 | Empty result | Evidence exists in `c00175/c00176` | Mentioned old law is incorrectly used as hard target-law filter |
| `labor_strike_definition_015` | Three Article 198 facts | Articles 117, 4, 200, 220, 209 | `c00240` only at expanded rank 7 | Fusion/top-5 cutoff |
| `labor_contract_content_018` | Three Article 21 facts | Articles 22, 13, 33, 28, 14 | `c00028` only at expanded rank 6 | Fusion/top-5 cutoff |
| `labor_retirement_age_040` | Three Article 169 facts | Articles 1, 113, 114, 52, 18 | `c00195` never returned | Chunk is mislabeled Article 168; ranking miss |
| `labor_employee_rights_052` | Three Article 5 facts | Articles 117, 11, 4, 1, 58 | Correct facts at expanded ranks 9/14 | Fusion/top-5 cutoff |
| `labor_employer_obligations_053` | Three Article 6 facts | Articles 11, 52, 163, 41, 177 | `c00009/c00010` absent | Similar-title semantic competition |
| `labor_relationship_principles_054` | Three Article 7 facts | Articles 179, 1, 164, 52, 11 | `c00011` absent | Embedding/sparse/fusion unresolved |
| `labor_worker_definition_056` | Three Article 3 facts | Articles 1, 4, 117, 11, 178 | `c00003` absent | Generic legal vocabulary dilutes ranking |
| `labor_forced_labor_definition_057` | Three Article 3 facts | Articles 11, 117, 4, 1, 121 | `c00006` absent | Generic legal vocabulary dilutes ranking |
| `labor_contract_types_008` | Two Article 20 facts | Articles 117, 22, 50, 18, 4 | `c00027` absent | Embedding/sparse/fusion unresolved |
| `tax_effective_date_062` | Article 29; 01/07/2026 | Articles 3, 28, 4, 7, 6 | `c00064` at expanded rank 14 | Fusion/top-5 cutoff |
| `cyber_transition_074` | Two Article 45 facts | Articles 42, 44, 3, 42, 35 | `c00176` at expanded rank 9 | Fusion/top-5 cutoff |
| `labor_overtime_pay_028` | Three percentage facts | Correct Article 98 chunks at ranks 1-4 | 0/3 exact substring | Benchmark punctuation/paraphrase, not retrieval |
| `land_price_cao_ba_quat_084` | Cao Ba Quat, 294.100 | Multi-row table chunks | Forbidden 491.700 is also present | Table chunk/metric does not preserve row association |

## D. Root-Cause Totals

Counts can overlap because one case may have both a corpus and ranking defect.

| Root cause | Evidence-backed total | Status |
| --- | ---: | --- |
| Query parsing/preferences | 2 cases, at least 3 facts | Observed: wage table false positive and cited-law hard filter |
| Hard metadata filter | 1 case | Observed: `cyber_transition_previous_law_078` |
| Fusion/final cutoff | 6 cases, 13 facts at expansion ranks 6-14 | Observed |
| Complete ranking miss | 15 cases, 28 source-matchable facts absent from all stored context | Observed, modality unknown |
| Dense embedding vs sparse search | Cannot split the 28 facts | `N/A_NO_TRACE`: modality candidate ranks were not stored |
| Reranking | 0 evaluated cases | `N/A_NOT_APPLICABLE`: reranker disabled |
| Chunking/article metadata | 5 gold article cases have no matching article metadata | Observed |
| OCR-primary exact-match loss | 9/35 source-unmatchable facts across 8 cases | Manual audit |
| Benchmark/evaluator wording | 26/35 source-unmatchable facts | Paraphrase, punctuation, composition or negative inference |
| Table row contamination | 4/11 table evidence cases | Forbidden value comes from another row in a multi-row chunk |

Of the 43 cases reported as `missing_context_fact`, only 16 are retrieval-only, 5 are mixed retrieval plus source/label problems, and 22 are source/label-only under exact matching. At fact level, 151/186 facts are exact-matchable in their expected document; retrieval covers 110/151 = 72.85% of that measurable ceiling.

Case inventories:

- Wrong doc: `labor_wage_scale_027`, `cyber_transition_previous_law_078`.
- Wrong clause: `labor_working_time_001`, `labor_annual_leave_005`.
- Forbidden context: `land_price_cao_ba_quat_084`, `land_price_chu_manh_trinh_085`, `land_price_lam_son_086`, `land_price_me_linh_087`.
- Empty: `cyber_transition_previous_law_078` is a failure; `unsupported_fake_article_098` is expected behavior for an unsupported citation trap.
- Wrong article (24): `labor_contract_types_008`, `labor_employee_termination_009`, `labor_strike_definition_015`, `labor_collective_bargaining_016`, `labor_contract_content_018`, `labor_assignment_020`, `labor_part_time_022`, `labor_illegal_termination_024`, `labor_wage_scale_027`, `labor_salary_stop_work_029`, `labor_unpaid_leave_033`, `labor_retirement_age_040`, `labor_notice_short_fixed_046`, `labor_return_documents_048`, `labor_employee_rights_052`, `labor_employer_obligations_053`, `labor_relationship_principles_054`, `labor_worker_definition_056`, `labor_forced_labor_definition_057`, `tax_effective_date_062`, `tax_taxpayer_064`, `tax_non_resident_072`, `cyber_transition_074`, `cyber_transition_previous_law_078`.
- Missing expected fact (43): `labor_overtime_002`, `labor_probation_result_007`, `labor_contract_types_008`, `labor_employee_termination_009`, `labor_employer_termination_010`, `labor_salary_payment_011`, `labor_minor_worker_013`, `labor_social_insurance_014`, `labor_strike_definition_015`, `labor_collective_bargaining_016`, `labor_contract_content_018`, `labor_prohibited_acts_019`, `labor_assignment_020`, `labor_severance_allowance_025`, `labor_wage_scale_027`, `labor_overtime_pay_028`, `labor_salary_stop_work_029`, `labor_rest_break_031`, `labor_weekly_rest_032`, `labor_unpaid_leave_033`, `labor_discipline_principles_035`, `labor_dismissal_037`, `labor_minor_rules_039`, `labor_retirement_age_040`, `labor_marriage_leave_041`, `labor_child_marriage_leave_042`, `labor_family_death_leave_043`, `labor_notice_short_fixed_046`, `labor_termination_payment_047`, `labor_return_documents_048`, `labor_discipline_time_limit_049`, `labor_suspension_work_050`, `labor_employee_rights_052`, `labor_employer_obligations_053`, `labor_relationship_principles_054`, `labor_worker_definition_056`, `labor_forced_labor_definition_057`, `tax_effective_date_062`, `tax_non_resident_072`, `cyber_transition_074`, `cyber_transition_previous_law_078`, `cyber_legal_basis_081`, `cyber_no_weather_082`.

## Score and First-Fact-Rank Analysis

- Selected Top-5 scores (`n=490`): min `0.20`, median `0.50`, mean `0.5769`, P75 `0.625`, max `1.85`.
- Rank means: rank 1 `0.8191`, rank 2 `0.6615`, rank 3 `0.5371`, rank 4 `0.4644`, rank 5 `0.4023`.
- Table-intent scores are not comparable with normal scores: table mean `1.3653` versus normal mean `0.4563` because metadata boosts add up to `0.85`.
- First exact fact rank among 151 source-matchable facts: rank 1 `57`, rank 2 `37`, rank 3 `9`, rank 4 `4`, rank 5 `3`, expansion ranks 6-14 `13`, never retrieved `28`.
- The 35 remaining facts have no exact substring in their own expected document and therefore have no meaningful retrieval rank.

## E. Three Priority Experiments

### 1. Fix query parsing and preference precision

- Hypothesis: distinguishing a target law from a cited law, and `bang luong` from a data table, fixes the two wrong-document cases.
- Single change: modify only `_parse_query_filters()` and `_parse_query_preferences()`; do not change embeddings or K.
- Cases: `labor_wage_scale_027`, `cyber_transition_previous_law_078`, plus all 11 table lookups as regression controls.
- Primary metric: Doc Hit@5 from `89/91` to `91/91`; secondary fact coverage at least `113/186`.
- Stop: both target cases hit the correct document, table facts remain `24/24`, and `unsupported_fake_article_098` remains empty.

### 2. Enable the existing reranker over 20 fused candidates

- Hypothesis: reranking can promote Article 21, 198, tax Article 29 and cyber Article 45 evidence currently outside the final Top-5.
- Single change: set `RERANK_ENABLED=true`; hold parser, corpus, embeddings and final K=5 fixed.
- Cases: 16 retrieval-only cases, with focus on six cases whose evidence appears at expansion rank 6-14.
- Primary metric: source-matchable content coverage from `110/151` to at least `123/151`; monitor Full Case Rate and P95.
- Stop: fewer than 7 additional facts, Doc Hit@5 drops, forbidden context exceeds `4/90`, or warm P95 exceeds 900 ms.

### 3. Repair the measurement ceiling before further model tuning

- Hypothesis: OCR repair plus verbatim fact labels prevents 35 source-unmatchable facts from being misreported as retrieval failures.
- Single change: update only evidence labels/source spans and the nine OCR-primary source strings; do not change retriever settings.
- Cases: all 27 source-unmatchable cases, especially `labor_overtime_pay_028`, `labor_salary_payment_011`, `labor_weekly_rest_032`, and `labor_suspension_work_050`.
- Primary metric: source-matchable fact ceiling from `151/186` to at least `177/186`.
- Stop: every changed fact has a cited source chunk and type `verbatim`, `paraphrase`, `compositional`, or `negative_inference`; no silent conversion of N/A to zero.

## F. Final Recommendation

Do not run full E2E yet. First run experiment 1 because it is deterministic, affects two severe wrong-document failures, and costs no model/API calls. Then run experiment 2 on the 16 retrieval-only cases. Experiment 3 must precede any claim that overall Fact Coverage reflects retriever quality rather than OCR and benchmark wording.
