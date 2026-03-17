# Human Evaluation Guide: EpiQAL-C

## Overview

EpiQAL-C evaluates **conclusion reconstruction under masked inputs**. The Discussion section of each research article is withheld at test time. Each question asks the reader to reconstruct an author-stated conclusion from the Discussion by reasoning over the remaining sections (Abstract, Methods, Results, etc.). The correct answer is a conclusion extracted from the Discussion, but it must be derivable from the non-Discussion content alone.

You will evaluate 20 sampled questions across three difficulty tiers (Easy / Medium / Hard).

## What You Will See

For each question, you are provided with:

| Field | Description |
|-------|-------------|
| **paragraph** | The article text **without** the Discussion section (this is what models see at test time) |
| **discussion** | The original Discussion section (provided to you for verification, but NOT shown to models) |
| **question** | The question stem (may have been rewritten to reduce surface-level shortcuts) |
| **ori_question** | The original question stem before any rewriting |
| **evidence** | Passages from the non-Discussion sections used to construct the question |
| **rationale** | The intended reasoning chain explaining why the reference answer is correct |
| **choices** | The answer options (typically one correct answer) |
| **ref_answers** | The reference correct answer(s), given as option indices |

## About Stem Refinement

Some questions have undergone **stem refinement**, a rewriting step designed to reduce surface-level shortcuts. In the original question, disease names or key entities appear directly (e.g., "cutaneous leishmaniasis"). After refinement, these entities are replaced with descriptive phrases (e.g., "a vector-borne skin disorder caused by Leishmania parasites transmitted via sandfly bites"). This forces models to reason about the concept rather than pattern-match on entity names.

You will see two fields:
- **ori_question**: The original question before rewriting. This preserves the original evidence and rationale.
- **question**: The rewritten version used for actual evaluation. This is what models see at test time.

If both fields are identical, the question was not rewritten (it already met the difficulty threshold without refinement).

When evaluating, **always evaluate the `question` field** (the rewritten version). The `ori_question` is provided only to help you understand the original intent if the rewritten version feels unclear.

## Evaluation Criteria

Rate each of the following dimensions on a **1-3 scale**.

### 1. Answer Correctness

*Is the reference answer actually correct?*

| Score | Meaning |
|-------|---------|
| 1 | The reference answer is incorrect or misleading |
| 2 | The reference answer is partially correct (e.g., the answer is correct but imprecise) |
| 3 | The reference answer is fully correct and consistent with the Discussion section |

### 2. Distractor Quality

*Are the incorrect options plausible yet clearly distinguishable from the correct answer through reasoning?*

| Score | Meaning |
|-------|---------|
| 1 | Distractors are obviously wrong or irrelevant (no reasoning needed to eliminate them) |
| 2 | Distractors are reasonable but not strongly misleading |
| 3 | Distractors are highly plausible and require careful reasoning to rule out |

### 3. Question Clarity

*Is the question stem clear, unambiguous, and well-formed?*

| Score | Meaning |
|-------|---------|
| 1 | The question is confusing, ambiguous, or poorly worded |
| 2 | The question is understandable but could be clearer |
| 3 | The question is clear, precise, and unambiguous |

### 4. Evidence Sufficiency

*Do the provided evidence passages (from non-Discussion sections) adequately support the reasoning chain needed to reach the correct answer?*

| Score | Meaning |
|-------|---------|
| 1 | The evidence does not support the correct answer |
| 2 | The evidence partially supports the answer but key steps are missing |
| 3 | The evidence fully supports the complete reasoning chain |

### 5. Reasoning Depth

*How many reasoning steps are required to arrive at the correct answer?*

| Score | Meaning |
|-------|---------|
| 1 | Single-step: the answer can be inferred from one observation |
| 2 | Two-step: requires combining two pieces of evidence or applying one epidemiological principle to one piece of evidence |
| 3 | Multi-step: requires integrating three or more pieces of evidence and/or applying domain knowledge to synthesize a conclusion |

### 6. Answerability

*Can the correct answer be derived from the non-Discussion sections alone?*

| Score | Meaning |
|-------|---------|
| 1 | The answer cannot be derived without the Discussion (requires information only present in the Discussion) |
| 2 | The answer can be partially derived but requires a significant reasoning leap beyond what the non-Discussion sections provide |
| 3 | The answer can be fully derived from the non-Discussion sections through logical reasoning |

## Evaluation Procedure

1. **Read the article** (paragraph field, which does NOT include the Discussion). Get a general understanding of the study design, methods, and results.
2. **Read the question and choices**. Try to answer it yourself using only the paragraph (non-Discussion content).
3. **Check the reference answer** against your own judgment. Review the evidence and rationale fields.
4. **Read the Discussion section** to verify whether the reference answer is indeed a conclusion stated by the authors, and whether it could reasonably be derived without the Discussion.
5. **Rate each dimension** (1-3) based on the criteria above.
6. Fill in your scores in the `evaluation` field of each item.

## Important Notes

- The **discussion** field is provided so you can verify the answer. Models do NOT see this at test time. Your Answerability rating should reflect whether the answer is derivable *without* the Discussion.
- Some questions have been **rewritten** (stem refinement) to reduce surface-level shortcuts. If the question seems unusually verbose or describes a disease indirectly, this is intentional. Evaluate the rewritten version (the `question` field), not the `ori_question`.
- The correct answer is typically a **single conclusion** extracted from the Discussion. Check whether it is genuinely a key finding rather than a minor observation or speculation.
- If you believe a distractor could also be a valid conclusion derivable from the non-Discussion sections, rate Answer Correctness as 2 and note this.
- The article texts are from epidemiological research. You are not expected to have deep domain expertise. Focus on whether the reasoning chain is logical and well-supported, not on whether you personally know the medical facts.