# Legal QA Failure Analysis

- Created at: 2026-06-20T13:57:16
- Dataset: `data\evaluation\legal_qa_eval_100.jsonl`
- Mode: scored_retrieval_predictions

## Top Failure Categories

| Category | Count | Case IDs |
| --- | --- | --- |
| missing_context_fact | 43 | labor_overtime_002, labor_probation_result_007, labor_contract_types_008, labor_employee_termination_009, labor_employer_termination_010, labor_salary_payment_011, labor_minor_worker_013, labor_social_insurance_014, labor_strike_definition_015, labor_collective_bargaining_016, labor_contract_content_018, labor_prohibited_acts_019, labor_assignment_020, labor_severance_allowance_025, labor_wage_scale_027, labor_overtime_pay_028, labor_salary_stop_work_029, labor_rest_break_031, labor_weekly_rest_032, labor_unpaid_leave_033 |
| wrong_article | 24 | labor_contract_types_008, labor_employee_termination_009, labor_strike_definition_015, labor_collective_bargaining_016, labor_contract_content_018, labor_assignment_020, labor_part_time_022, labor_illegal_termination_024, labor_wage_scale_027, labor_salary_stop_work_029, labor_unpaid_leave_033, labor_retirement_age_040, labor_notice_short_fixed_046, labor_return_documents_048, labor_employee_rights_052, labor_employer_obligations_053, labor_relationship_principles_054, labor_worker_definition_056, labor_forced_labor_definition_057, tax_effective_date_062 |
| forbidden_context_fact | 4 | land_price_cao_ba_quat_084, land_price_chu_manh_trinh_085, land_price_lam_son_086, land_price_me_linh_087 |
| wrong_clause | 2 | labor_working_time_001, labor_annual_leave_005 |
| wrong_doc | 2 | labor_wage_scale_027, cyber_transition_previous_law_078 |

## How To Use

- `wrong_intent`: inspect router prompt and intent labels.
- `wrong_doc`, `wrong_article`, `wrong_clause`: inspect retrieval filters, chunk metadata, and ranking.
- `missing_context_fact`: inspect corpus extraction, chunking, retrieval ranking, and top-k context coverage.
- `forbidden_context_fact`: inspect misleading retrieved context before blaming generation.
- `missing_fact`: inspect generator prompt after confirming retrieved context coverage.
- `unsupported_claim`: inspect hallucination grader and citation grounding.
- `refusal_error`: inspect out-of-scope and unsafe request handling.
- `quota_or_rate_limit`: rerun later with `--skip-existing`; do not treat this as model quality failure.
- `runtime_error`: inspect service dependencies, Qdrant, graph exceptions, and API/runtime logs.