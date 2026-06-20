# Component-wise Evaluation Prompts

Bo prompt nay dung de danh gia he thong Multi-Agent RAG phap luat Viet Nam theo tung thanh phan, thay vi chay toan bo graph va tieu ton quota API trong mot lan.

## 1. Nguyen tac su dung

1. Chay theo thu tu: Corpus -> Retrieval -> Router -> Grader -> Hallucination Grader -> Web Search -> Generation -> E2E -> Ablation.
2. Moi lan chi danh gia mot component va luu ket qua vao mot file Markdown/JSON rieng.
3. Uu tien metric deterministic. Chi dung LLM-as-judge khi rule, gold label hoac human review khong du.
4. Tai su dung prediction, context va agent trace da cache. Khong goi lai API chi de viet lai bao cao.
5. Khong dien `0` thay cho du lieu chua co va khong suy dien metric tu mot metric khac.
6. Khi chay tap nho, luon ghi `evaluated_cases/eligible_cases`, coverage va khoang tin cay neu co.

Quy uoc trang thai bat buoc:

| Trang thai | Y nghia |
| --- | --- |
| `MEASURED` | Da co du lieu va metric duoc tinh hop le |
| `N/A_NOT_RUN` | Chua chay component |
| `N/A_NO_LABEL` | Co prediction nhung benchmark thieu gold label can thiet |
| `N/A_NOT_APPLICABLE` | Metric khong ap dung cho case/component nay |
| `INVALID_PARTIAL` | Du lieu qua thieu hoac coverage khong dat nguong de ket luan |
| `BLOCKED_QUOTA` | Lan chay bi loi quota/rate limit |
| `BLOCKED_RUNTIME` | Bi loi dependency, Qdrant, network hoac runtime |

Moi bao cao phai tach ba loai ket luan:

- **Observed**: so lieu do duoc truc tiep.
- **Inferred**: nhan dinh suy ra tu so lieu, phai neu ro ly do.
- **Unknown**: chua co du lieu, kem cach thu thap.

## 2. Ke hoach tiet kiem quota

| Giai doan | Tap goi y | API LLM | Dieu kien de chuyen buoc |
| --- | ---: | ---: | --- |
| Corpus | Toan bo 527 chunks | 0 | Loi OCR/metadata nghiem trong da duoc khoanh vung |
| Retrieval | 100 cases | 0 | Fact coverage va failure slices da ro |
| Router | 30-50 cases can bang intent | Thap | Co confusion matrix va case sai |
| Grader | 30-50 frozen context cases | Thap | Co gold `context_sufficient` |
| Hallucination grader | 30-50 answer-context pairs | Thap/0 voi rules | Co positive va adversarial cases |
| Web Search | 10-20 cases can web | Thap | Do duoc trigger va source quality |
| Generation | 10-20 cached contexts | Trung binh | Retrieval/router/grader du on |
| E2E | 20 cases | Cao | Component gates quan trong da dat |
| Ablation | 20 cases truoc, sau do 100 retrieval | Tuy bien the | Chenh lech co y nghia va cung tap case |

## 3. Prompt dieu phoi tong the

Dung prompt nay khi muon AI doc cac bao cao da co, xac dinh component can danh gia tiep theo, nhung khong tu y chay generation.

```text
Ban la Principal AI Engineer chuyen ve Agentic RAG, LLM evaluation va he thong hoi dap phap luat. Ban khat khe ve tinh dung dan, grounding, kha nang lap lai, latency va chi phi. Hay audit project Multi-Agent RAG for Vietnamese Legal QA theo huong component-wise.

Rang buoc:
- API quota dang han che. Khong chay generation, LLM-as-judge hoac full E2E neu toi chua yeu cau ro.
- Uu tien doc artifact hien co va metric deterministic.
- Khong duoc bien N/A thanh 0, PASS hoac FAIL.
- Moi N/A phai duoc gan mot ly do: N/A_NOT_RUN, N/A_NO_LABEL, N/A_NOT_APPLICABLE, INVALID_PARTIAL, BLOCKED_QUOTA hoac BLOCKED_RUNTIME.
- Khong coi Doc Hit@K cao la retrieval tot neu Context Fact Coverage@K thap.
- Article/Clause/Point Hit@K chi la diagnostic khi gold label thua; khong dung lam quality gate chinh.
- Khong danh gia generation tot/xau khi retrieval context khong chua du expected facts.
- Tach ro observed, inferred va unknown.

Hay doc toi thieu:
- README.md
- data/evaluation/legal_qa_eval_100.jsonl
- data/evaluation/legal_qa_eval_e2e_20.jsonl
- data/processed/ingestion_quality_report.md
- eval_reports/retrieval_100.md
- eval_reports/failure_analysis.md
- scripts/evaluate_legal_qa.py
- prediction/report cua component dang co

Nhiem vu:
1. Lap inventory artifact: dataset, prediction, report, trace, gold labels va coverage.
2. Xac dinh metric nao MEASURED va metric nao N/A kem ly do chinh xac.
3. Phat hien metric gay hieu lam, mau so qua nho, data leakage va ket luan vuot qua du lieu.
4. Xep hang toi da 5 nut that theo impact x confidence x effort.
5. De xuat component tiep theo nen danh gia, tap case toi thieu va so API call uoc tinh.
6. Khong de xuat full E2E neu mot loi upstream da du de giai thich that bai.

Dau ra bat buoc:
A. Executive verdict: NOT READY / CONDITIONALLY READY / READY cho internal demo va production tach rieng.
B. Measurement coverage table: component, metric, numerator/denominator, status, artifact nguon.
C. Top findings: severity, evidence, affected cases, likely root cause.
D. N/A closure plan: metric, ly do N/A, label/log can them, lenh hoac quy trinh de thu thap.
E. Next run: dung component, case IDs/slices, budget API va stop condition.
F. Khong viet nhan xet chung chung neu khong gan voi file, metric hoac case cu the.
```

## 4. Prompt danh gia Evaluation Corpus

Prompt nay khong can goi LLM API cua he thong.

```text
Dong vai Senior Evaluation Engineer chuyen xay benchmark cho Vietnamese Legal QA. Hay audit chat luong evaluation corpus, khong danh gia model va khong chay graph.

Input can doc:
- data/evaluation/legal_qa_eval_100.jsonl
- data/evaluation/legal_qa_eval_e2e_20.jsonl
- data/processed/document_registry.jsonl
- data/processed/chunks.jsonl
- data/processed/ingestion_quality_report.md

Kiem tra:
1. Schema va label completeness: id, category, type, difficulty, answer_policy, requires_web, expected_intent, doc_id, article_number, clause_number, point_number, level, expected_facts, forbidden_facts.
2. Duplicate/near-duplicate question, label mau thuan, fact qua mo ho, fact chi la substring khong du nghia.
3. Coverage theo document, linh vuc, intent, question type, difficulty, grounded/refusal/web policy.
4. Do dai va do kho cua expected_facts; forbidden_facts co thuc su adversarial hay chi la phu dinh de.
5. Gold citation co the truy vet den chunk hay khong. Neu chi co article/clause label ma khong co gold chunk/evidence span, danh dau limitation.
6. Temporal validity: van ban co hieu luc, tuong lai, bi thay the hay can web freshness.
7. Benchmark leakage: cau hoi/expected fact co trung nguyen van metadata hoac query template hay khong.
8. E2E subset co dai dien cho 100 cases hay bi lech theo category/difficulty.

Metric toi thieu:
- case count va unique ID rate
- label coverage cho tung field, bao gom numerator/denominator
- distribution theo category/type/difficulty/intent/policy/document
- expected fact count trung binh va forbidden fact coverage
- duplicate/near-duplicate rate
- gold evidence traceability rate
- E2E subset representation delta so voi tap 100

Dau ra:
1. Corpus verdict: VALID / USABLE_WITH_LIMITATIONS / INVALID.
2. Bang metric voi mau so, khong chi phan tram.
3. Danh sach case can sua, moi case ghi field, gia tri hien tai, gia tri de xuat va ly do.
4. Labels can bo sung de mo khoa Router, Grader, Web Search, Generation va E2E metrics.
5. De xuat bo 20 case toi thieu can bang de chay E2E voi quota thap.
6. Moi gia tri chua do duoc phai mang ma N/A cu the; khong tu suy dien label.
```

## 5. Prompt danh gia Corpus/Ingestion Quality

Prompt nay khong can LLM API cua he thong.

```text
Dong vai Senior RAG Data Engineer. Hay audit pipeline ingestion cho corpus phap luat Viet Nam. Muc tieu la xac dinh chunks co du sach, dung cau truc va truy vet duoc de retrieval hay khong.

Input:
- data/processed/document_registry.jsonl
- data/processed/chunks.jsonl
- data/processed/ingestion_quality_report.md
- scripts/validate_legal_corpus.py
- scripts/check_ingestion_quality.py
- scripts/prepare_legal_pdfs.py
- code chunking/indexing lien quan

Khong chi kiem tra missing metadata. Hay danh gia:
1. Registry-to-chunk integrity va doc_id uniqueness.
2. Source URL/path, page range, effective status va document version.
3. Phan tach document/article/clause/point/table; parent-child integrity.
4. Dieu, khoan, diem bi cat ngang, gop nham hoac gan nham parent.
5. OCR noise: ky tu loi, chu/so nham, tieu de phap ly vo, dau cau va so lieu hong.
6. Chunk size distribution theo extraction method va legal level.
7. Duplicate/overlap chunks va boilerplate lap lai.
8. Table preservation: hang/cot, ten duong, muc gia va don vi con gan ket.
9. Kha nang trace citation tu chunk ve van ban, dieu/khoan/diem va trang.

Bat buoc lay mau thu cong it nhat:
- 10 OCR chunks co suspicious score cao
- 5 article, 5 clause, 5 point
- tat ca table chunks neu so luong nho
- chunks cua moi document

Dau ra:
A. Quality gate table: metric, value, threshold, status, evidence.
B. Error taxonomy: OCR, segmentation, hierarchy, metadata, duplicate, table, source traceability.
C. Top 20 chunks rui ro cao voi chunk_id va ly do.
D. Uoc tinh anh huong cua tung loi den retrieval va citation.
E. Ke hoach sua theo P0/P1/P2, kem cach kiem thu lai.
F. Khong ket luan corpus tot chi vi missing metadata = 0.
```

Lenh deterministic tham khao:

```powershell
python scripts/check_ingestion_quality.py
python scripts/validate_legal_corpus.py
```

## 6. Prompt danh gia Retrieval

Prompt nay khong goi LLM; can Qdrant va embedding/index runtime.

```text
Dong vai Principal Retrieval Engineer. Hay danh gia retrieval cua Vietnamese Legal RAG bang evidence content, khong lay metadata hit lam ket luan chinh.

Input:
- data/evaluation/legal_qa_eval_100.jsonl
- eval_reports/retrieval_predictions.jsonl
- eval_reports/retrieval_100.json
- eval_reports/failure_analysis.md
- scripts/evaluate_legal_qa.py
- src/agents/retriever.py

Metric quality gate chinh:
- Doc Hit@K
- Context Fact Coverage@K = expected facts xuat hien trong retrieved context / tong expected facts cua eligible cases
- Full Context Fact Case Rate@K = cases co day du tat ca expected facts / eligible cases
- Forbidden Fact In Context Rate@K = eligible cases co forbidden fact xuat hien / eligible cases
- MRR
- error rate, avg latency va P95 latency

Metric diagnostic, khong dung lam quality gate chinh neu gold sparse:
- Article Hit@K, Clause Hit@K, Point Hit@K, Level Hit@K

Yeu cau:
1. Luon ghi eligible cases va numerator/denominator cho moi metric.
2. Bao cao theo slice: document, category, type, difficulty, level, requires_web va table/non-table.
3. Liet ke case thieu expected fact, wrong doc, wrong article/clause, forbidden context va empty result.
4. Phan biet retrieval failure voi benchmark-label failure va OCR/chunking failure.
5. Kiem tra top-K sensitivity it nhat K=1, 3, 5, 10 neu prediction luu du chunks.
6. Phan tich score distribution va rank cua chunk dau tien chua tung expected fact.
7. Neu dung substring matching, neu han che voi dau tieng Viet, OCR, so `08/8`, don vi va paraphrase.
8. Khong ket luan retrieval tot dua tren Doc Hit@5 hoac MRR khi fact coverage thap.

Dau ra:
A. PASS/FAIL cho tung gate, kem numerator/denominator.
B. Slice table sap xep tu yeu den manh.
C. Top 15 failure cases: expected, retrieved evidence, missing evidence, likely root cause.
D. Root-cause totals: query parsing, filter, embedding, sparse search, fusion, reranking, chunking/OCR, label.
E. Ba thi nghiem uu tien tiep theo, moi thi nghiem co hypothesis, thay doi duy nhat, metric chinh va stop condition.
F. Goi y khong duoc dua tren cam tinh; phai gan voi failure cases.
```

Lenh hien co:

```powershell
python scripts/run_retrieval_eval.py --dataset data/evaluation/legal_qa_eval_100.jsonl --out eval_reports/retrieval_predictions.jsonl
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_100.jsonl --predictions eval_reports/retrieval_predictions.jsonl --component retrieval --out-json eval_reports/retrieval_100.json --out-md eval_reports/retrieval_100.md
```

Moc tham chieu hien tai, khong duoc coi la ket qua moi:

- Context Fact Coverage@5: `110/186 = 59.14%`.
- Full Context Fact Case Rate@5: `47/90 = 52.22%`.
- Forbidden Fact In Context Rate@5: `4/90 = 4.44%`.
- Vi hai gate coverage dang fail, uu tien sua retrieval truoc khi full E2E.

## 7. Prompt danh gia Router

Router can mot tap co `expected_intent`; chi chay router node, khong chay full graph neu co the.

```text
Dong vai Senior Agent Routing Engineer. Hay danh gia Router nhu mot classifier va policy gate, khong dua vao chat luong final answer de suy nguoc router dung hay sai.

Input:
- benchmark co expected.expected_intent va answer_policy
- router predictions gom id, predicted intent, confidence, latency, error
- src/agents/router.py
- src/graph/edges.py

Truoc khi cham:
1. Liet ke intent taxonomy thuc te trong code va taxonomy trong benchmark.
2. Tao mapping ro rang neu ten label khac nhau; khong map ngam.
3. Kiem tra moi intent co du mau. Intent duoi 5 cases phai danh dau sample risk.

Metric:
- intent accuracy
- macro/micro precision, recall, F1
- confusion matrix
- refusal/out-of-scope precision, recall va accuracy
- unsafe routing recall neu co unsafe cases
- web-required routing recall neu router quyet dinh nhanh nay
- confidence calibration: accuracy theo confidence bucket, ECE neu du mau
- error rate, avg/P95 router latency

Failure slices:
- legal_query vs procedural
- out_of_scope/conversational
- unsafe/refusal
- ambiguous/multi-intent
- requires_web true/false
- difficulty va category

Dau ra:
A. Bang label coverage va confusion matrix.
B. Metric tong va per-class voi numerator/denominator.
C. False accept va false reject phai duoc liet ke rieng; false accept phap ly nguy hiem co severity cao hon.
D. Case ID, question, gold, predicted, confidence va likely cause cho tung loi.
E. De xuat thay doi prompt/rule/taxonomy, nhung khong sua nhieu bien cung luc.
F. Neu chua co prediction router, tra N/A_NOT_RUN va de xuat schema JSONL toi thieu; khong suy ra tu retrieval report.
```

Schema prediction toi thieu de dong `N/A`:

```json
{"id":"case_id","intent":"legal_query","intent_confidence":0.92,"router_ms":35,"error":null}
```

### 7.1. Prompt hoan tat Router sau khi quota reset

Dung prompt nay de tiep tuc run hien tai. Khong tao benchmark moi va khong xoa
prediction da co.

```text
Dong vai Principal Agent Routing Evaluation Engineer. Hay hoan tat danh gia
Router cua project Multi-Agent RAG for Vietnamese Legal QA bang blind holdout
hien tai, khong chay full graph va khong goi generation.

Trang thai dong bang:
- Dataset: data/evaluation/router_holdout_30_v1.jsonl
- Benchmark version: router-holdout-30-v1.0
- Prediction file: eval_reports/router_holdout_30_predictions.jsonl
- Prompt version: router-policy-v2.2
- Model: gemini-2.5-flash
- Temperature: 0.1
- Ket qua truoc quota: 18/30 rows, 17 prediction thanh cong, 1 row
  RESOURCE_EXHAUSTED.

Rang buoc nghiem ngat:
1. Khong sua Router prompt, benchmark, gold labels, model hoac temperature
   truoc khi dat 30/30. Neu sua, run khong con la cung mot holdout.
2. Khong xoa prediction thanh cong. Dung --append --skip-existing de resume;
   row co error phai duoc retry va thay the theo ID.
3. Moi batch toi da 5 case. Dung ngay neu gap RESOURCE_EXHAUSTED hoac
   max-errors; khong tiep tuc thu tat ca key.
4. Khong tinh quota/runtime error la misclassification, nhung van tinh trong
   reliability error rate va intention-to-evaluate denominator.
5. Khong bien N/A thanh 0, PASS hoac FAIL. Unsupported va web-required chi
   duoc MEASURED sau khi co prediction thanh cong cho du gold cases.
6. Khong chay generation, grader, web search, hallucination grader hoac E2E.
7. Khong toi uu prompt dua tren tung failure cua holdout truoc khi dong report
   30/30; chi ghi failure de phan tich sau.

Quy trinh bat buoc:
A. Chay validator va xac nhan benchmark co 30 case, moi policy action co 5 case,
   benchmark_version dong nhat.
B. Doc prediction file, bao cao successful/error/missing IDs truoc khi goi API.
C. Chay Router theo batch 5 bang lenh quota-aware. Sau moi batch, kiem tra so
   unique IDs, duplicate IDs, unknown IDs va configuration consistency.
D. Lap lai cho den 30/30 hoac gap stop condition.
E. Khi dat 30/30, chay deterministic scorer va cap nhat:
   - eval_reports/router_holdout_30.json
   - eval_reports/router_holdout_30.md
   - eval_reports/router_overall_evaluation.md
   - docs/EVALUATION_RESULTS.md va README.md chi voi metric da do.
F. Chay tests/test_router_evaluation.py. Khong can cai pytest vao .venv neu co
   the dung Python environment da co pytest.

Metric va quality gate:
- intent accuracy >= 90%
- intent macro F1 >= 85%
- policy-action accuracy >= 90%
- unsafe refusal recall = 100%
- unsupported refusal recall >= 90%
- web-required recall >= 90%
- runtime error rate <= 2%
- P95 Router latency <= 6,000 ms
- Bao cao rieng single-attempt latency, fallback latency, fallback case rate va
  average attempt count. Khong gan latency do fallback cho clean model latency.

Dau ra bat buoc:
1. Coverage funnel: 30 selected -> attempted -> successful -> scored.
2. Intent va policy confusion matrices, tong va per-class TP/FP/FN.
3. False accept, false reject, unsafe, unsupported va web-required failures.
4. Avg/P50/P95 latency, error rate, fallback rate va confidence calibration.
5. Bang failure gom ID, question, gold, predicted, confidence, error va likely
   cause; tach model error khoi quota/runtime error.
6. Verdict rieng cho functional quality, internal demo, production va latency.
7. Neu chua dat 30/30, ket luan INVALID_PARTIAL/BLOCKED_QUOTA va liet ke chinh
   xac IDs con lai; khong tuyen bo Router accuracy tong the.
8. Neu da dat 30/30, de xuat toi da ba thay doi sau holdout, moi thi nghiem chi
   thay doi mot bien va co stop condition.
```

Lenh resume sau khi quota reset:

```powershell
python scripts\validate_router_benchmark.py --dataset data\evaluation\router_holdout_30_v1.jsonl
python scripts\run_router_eval.py --dataset data\evaluation\router_holdout_30_v1.jsonl --out eval_reports\router_holdout_30_predictions.jsonl --append --skip-existing --limit 5 --max-errors 1
```

Lap lenh `run_router_eval.py` theo batch 5 den khi du 30/30. Sau do cham:

```powershell
python scripts\score_router_eval.py --dataset data\evaluation\router_holdout_30_v1.jsonl --predictions eval_reports\router_holdout_30_predictions.jsonl --out-json eval_reports\router_holdout_30.json --out-md eval_reports\router_holdout_30.md
python -m pytest -q tests\test_router_evaluation.py
```

## 8. Prompt danh gia Grader

Grader phai duoc danh gia tren frozen retrieval contexts. `requires_web` khong tu dong dong nghia voi gold verdict.

```text
Dong vai Senior CRAG Evaluation Engineer. Hay danh gia Grader ve kha nang quyet dinh retrieved context co DU de tra loi hay can fallback. Khong cham Grader chi bang document relevance.

Input:
- frozen pairs: question + retrieved_documents
- gold label `context_sufficient` do con nguoi gan
- expected_facts va forbidden_facts
- grader prediction: verdict, score, reasoning, latency, error
- src/agents/grader.py

Quy tac gold:
- `yes`: context chua du cac fact cot loi de tra loi dung va khong co mau thuan nghiem trong.
- `no`: context thieu fact cot loi, sai van ban/phien ban, hoac can nguon moi hon.
- `uncertain`: annotator khong the quyet; loai khoi accuracy va bao cao rieng.
- Khong lay `requires_web=false` lam gold `yes`, vi retrieval van co the lay thieu context.

Tao tap can bang gom:
- relevant va sufficient
- relevant nhung incomplete
- irrelevant
- mixed relevant/irrelevant
- wrong/expired version
- table lookup context
- empty context

Metric:
- accuracy, precision/recall/F1 cho verdict `yes` va `no`
- confusion matrix
- false-positive rate: Grader noi `yes` khi context thieu, day la loi nguy hiem
- false-negative rate: Grader noi `no` khi context du, gay ton web/API va tang latency
- score calibration va threshold sensitivity
- parse/error rate, avg/P95 latency

Dau ra:
A. Gold-label protocol va inter-annotator agreement neu co nhieu nguoi gan.
B. Metric voi numerator/denominator va confidence interval neu tap nho.
C. Tat ca false positive phai co missing expected facts cu the.
D. Uoc tinh cost cua false negative: web calls/API calls/latency thua.
E. De xuat threshold va prompt changes dua tren error slices.
F. Neu thieu `context_sufficient`, tra N/A_NO_LABEL va tao annotation template; khong dung requires_web de thay the.
```

Schema annotation/prediction goi y:

```json
{"id":"case_id","context_version":"retrieval_run_id","context_sufficient":"yes","missing_facts":[],"gold_reason":"...","grader_verdict":"yes","grader_score":0.88,"grader_ms":120,"error":null}
```

## 9. Prompt danh gia Hallucination Grader

Co the cham deterministic rules truoc; chi goi LLM grader cho cases rules khong quyet duoc.

```text
Dong vai AI Safety Evaluation Engineer cho Legal QA. Hay danh gia Hallucination Grader nhu mot binary detector. Muc tieu uu tien la giam false pass: cau tra loi sai/khong duoc support nhung grader cho pass.

Input:
- question, frozen context, web results, answer, citations
- human gold `is_grounded` va `error_types`
- prediction cua deterministic rules va LLM grader tach rieng
- src/agents/hallucination_grader.py

Tap test phai co:
- answer hoan toan grounded
- missing citation
- invalid source ID
- citation dung ID nhung khong support claim
- unsupported number/date/percentage
- wrong article/clause
- contradiction voi context
- them thong tin hop ly nhung khong co trong context
- context local va web mau thuan
- refusal dung va refusal sai
- paraphrase dung, tranh bat loi chi vi khong trung substring

Metric:
- precision, recall, F1 cho class hallucination/fail
- false-pass rate, dat lam safety gate chinh
- false-fail rate
- confusion matrix theo error type
- deterministic-rule coverage
- incremental value cua LLM sau rules
- parse/error/quota rate, retries, avg/P95 latency va token/call neu co

Yeu cau dac biet:
1. Kiem tra fallback khi exception. Neu system loi ma mac dinh `pass`, danh dau P0 safety risk.
2. Kiem tra numeric rule co nham so trong source ID/article/citation thanh unsupported claim khong.
3. Kiem tra citation URL validity rieng voi citation display validity.
4. Khong coi grader self-verdict la gold.

Dau ra:
A. Safety verdict va false-pass cases dau tien.
B. Metric deterministic-only, LLM-only va combined neu co du lieu.
C. Error matrix theo perturbation type.
D. P0/P1 fixes va regression tests cu the.
E. Neu thieu human gold, tra N/A_NO_LABEL va tao tap adversarial toi thieu 30 cases.
```

## 10. Prompt danh gia Web Search

```text
Dong vai Senior Search/RAG Evaluation Engineer. Hay danh gia Web Search agent o hai tang: quyet dinh KHI NAO can search va ket qua search CO DU CHAT LUONG hay khong.

Input:
- cases co gold `requires_web`
- grader verdict va web trigger trace
- web query, top results, URL, title, content/snippet, timestamp
- final selected web context
- src/agents/web_searcher.py

Metric tang trigger:
- trigger precision/recall/F1
- unnecessary web-call rate
- missed-web rate
- web calls per case va latency overhead

Metric tang result:
- authoritative source rate: vanban.chinhphu, quochoi, chinhphu, bo/nganh hoac nguon phap ly duoc chap nhan
- source URL validity va accessibility
- freshness/effective-version accuracy
- expected fact coverage trong web context
- contradiction rate voi local corpus
- duplicate result rate va domain diversity
- top-K relevance va latency

Quy tac:
- Khong cham final answer trong prompt nay.
- Tach no-result, API error, quota error va parser error.
- Nguon SEO/tong hop khong duoc coi ngang hang voi van ban chinh thong.
- Voi cau hoi co the tra loi du tu local corpus, web call la chi phi thua tru khi can kiem tra hieu luc/freshness.

Dau ra:
A. Trigger confusion matrix.
B. Source-quality table theo domain va case.
C. Danh sach missed-web va unnecessary-web cases.
D. Uoc tinh latency/API cost wasted.
E. De xuat query rewrite, allowlist/ranking va conflict policy.
F. Neu benchmark chua co du cases requires_web, tra INVALID_PARTIAL hoac N/A_NO_LABEL thay vi ket luan tong quat.
```

## 11. Prompt danh gia Generation

Chi dung prompt nay sau khi retrieval context da duoc cache. Co the hoan lai den khi quota cho phep.

```text
Dong vai Senior Legal LLM Evaluation Engineer. Hay danh gia Generator tren FROZEN CONTEXT de tach loi generation khoi retrieval. Khong goi retriever, router, grader hay web search lai.

Input cho moi case:
- question
- frozen retrieved context va web context
- expected_facts, forbidden_facts, answer_policy
- generated answer, citations, model/config, token usage, latency

Metric deterministic:
- answer fact coverage
- forbidden fact rate
- displayed citation validity
- source-ID/URL mapping validity
- unsupported numeric claim rate
- refusal compliance
- empty/truncated/format error rate

Human hoac LLM-as-judge chi la secondary signal:
- faithfulness
- completeness
- legal usefulness
- citation support

Quy tac quota:
- Chay 10-20 cases dai dien truoc.
- Luu output ngay sau moi case va skip-existing khi chay lai.
- Mot answer chi duoc sinh mot lan trong baseline; khong cherry-pick lan retry tot nhat.
- LLM-as-judge chi cham answer da cache va khong cham case bi runtime/quota error.
- Bao cao prediction coverage va quota-error rate.

Phan tich:
1. Neu context thieu expected fact, gan upstream_context_insufficient; khong quy het cho generator.
2. Neu context co fact nhung answer bo sot, gan generation_omission.
3. Neu answer them claim khong co context, gan unsupported_claim.
4. Neu noi dung dung nhung citation sai, gan citation_mapping_error.

Dau ra:
A. Metric voi numerator/denominator va coverage.
B. Error attribution upstream vs generation.
C. Case-level failure table.
D. Token, latency va retry cost.
E. Prompt/model changes duoc xep hang theo impact/cost.
```

## 12. Prompt danh gia E2E Response

```text
Dong vai Principal AI Systems Engineer. Hay danh gia full E2E response cua Multi-Agent Legal RAG tren tap quota-aware 20 cases. E2E chi duoc ket luan sau khi bao cao ro coverage va loi quota.

Input:
- data/evaluation/legal_qa_eval_e2e_20.jsonl
- eval_reports/e2e_predictions_20.jsonl
- eval_reports/e2e_20.json va .md
- agent_events/trace cua tung case
- cac bao cao component upstream

Metric:
- prediction coverage
- answer fact coverage va full-fact case rate
- forbidden fact rate
- grounded answer rate
- unsupported claim rate
- citation display/URL validity
- refusal accuracy
- web fallback precision/recall neu co gold
- total error, quota error va runtime error rate
- avg/P50/P95 total latency
- latency theo router, retriever, grader, web, generator, hallucination grader
- generation attempts/retries va loop exhaustion

Quy tac:
- Khong tinh quota/runtime-error rows nhu answer sai ma khong bao cao rieng; cung khong loai bo im lang.
- Bao cao ca intention-to-evaluate denominator va successfully-scored denominator.
- Neu coverage duoi 80%, verdict tong phai la INVALID_PARTIAL.
- Truy vet moi case fail den component dau tien gay loi.
- Khong de `N/A` ma khong co closure plan.

Dau ra:
A. Verdict cho internal demo va production tach rieng.
B. Funnel: selected -> attempted -> completed -> scored -> passed.
C. Quality gates va latency budget.
D. Failure attribution theo component dau tien.
E. Top 10 traces can xem lai.
F. Next action: component nao sua truoc; khong de xuat tang model size neu loi den tu corpus/retrieval.
```

Lenh quota-aware:

```powershell
python scripts/run_e2e_eval.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --out eval_reports/e2e_predictions_20.jsonl --skip-existing
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --predictions eval_reports/e2e_predictions_20.jsonl --component e2e --only-predicted --out-json eval_reports/e2e_20.json --out-md eval_reports/e2e_20.md
```

## 13. Prompt danh gia Ablation

```text
Dong vai Senior ML Experimentation Engineer. Hay danh gia ablation variants tren cung dataset, cung case IDs, cung top-K va cung evaluator. Khong so sanh cac run co coverage khac nhau ma khong can chinh.

Variants du kien:
- dense
- sparse
- hybrid
- hybrid + reranker/context expansion neu co
- full graph chi dung cho metric E2E

Yeu cau:
1. Xac minh config va prediction file cua tung variant thuc su ton tai.
2. Neu file thieu, ghi N/A_NOT_RUN; khong dung bang rong nhu ket qua 0.
3. Chi so sanh giao case IDs chung; bao cao coverage rieng cua tung run.
4. Retrieval ablation dung Context Fact Coverage@K, Full Context Fact Case Rate@K, Forbidden Fact Rate@K, MRR, latency.
5. Article/Clause/Point hit chi la diagnostic.
6. Bao cao per-case win/loss/tie, khong chi aggregate delta.
7. Neu chay E2E, giu model/prompt/temperature co dinh va bao cao quota errors.
8. Ket luan practical significance: quality gain co dang latency/cost tang them khong.

Dau ra:
A. Config parity checklist.
B. Coverage-adjusted comparison table.
C. Delta so voi baseline va per-slice delta.
D. Win/loss/tie cases va regression cases.
E. Recommendation KEEP / REJECT / INCONCLUSIVE cho tung bien the.
F. Thi nghiem tiep theo chi thay doi mot bien.
```

Lenh:

```powershell
python scripts/compare_ablation_runs.py --dataset data/evaluation/legal_qa_eval_100.jsonl --run dense=eval_reports/dense_predictions.jsonl --run sparse=eval_reports/sparse_predictions.jsonl --run hybrid=eval_reports/hybrid_predictions.jsonl --run full_graph=eval_reports/e2e_predictions.jsonl
```

## 14. Prompt tong hop bao cao cuoi

Chi dung sau khi da co it nhat Corpus, Retrieval va mot component agent.

```text
Dong vai AI Engineering Reviewer doc lap. Hay tong hop cac bao cao component thanh mot system evaluation report. Khong tinh trung mot loi o nhieu tang va khong che lap N/A.

Input:
- corpus report
- retrieval report
- router report
- grader report
- hallucination grader report
- web search report
- generation/E2E report neu co
- ablation report neu co

Nhiem vu:
1. Tao measurement coverage matrix, moi metric co status va artifact nguon.
2. Xay failure propagation map: corpus -> retrieval -> grader/web -> generation -> hallucination grader -> final response.
3. Xac dinh component dau tien gay loi cho moi case de tranh double counting.
4. Danh gia readiness rieng cho functional quality, legal safety, reliability, latency, observability va cost.
5. Xep hang backlog bang severity x frequency x confidence / effort.
6. Tach quick wins, medium-term experiments va blockers.

Dau ra:
A. Executive verdict: NOT READY / CONDITIONALLY READY / READY.
B. Scorecard voi measured coverage, khong tao mot diem tong hop tuy y.
C. Top 5 systemic bottlenecks co evidence.
D. N/A register va ke hoach dong tung N/A.
E. 2-tuan evaluation roadmap co thu tu, dataset, metric, gate, API budget va stop condition.
F. Danh sach claim co the dua vao README/CV va claim chua du bang chung.
```

## 15. Thu muc output goi y

De cac lan danh gia de theo doi va khong ghi de ket qua cu:

```text
eval_reports/
  corpus_quality.md
  retrieval_100.json
  retrieval_100.md
  router_eval.json
  router_eval.md
  grader_eval.json
  grader_eval.md
  hallucination_grader_eval.json
  hallucination_grader_eval.md
  web_search_eval.json
  web_search_eval.md
  generation_eval.json
  generation_eval.md
  e2e_20.json
  e2e_20.md
  ablation_report.json
  ablation_report.md
  system_evaluation_summary.md
```

Moi run nen luu them:

- `run_id`, timestamp va git commit.
- Dataset path/hash va case IDs.
- Model, prompt version, temperature va top-K.
- Prediction coverage.
- API call count, quota/runtime errors, token va latency neu co.
- Link den prediction/trace goc de co the reproduce.
