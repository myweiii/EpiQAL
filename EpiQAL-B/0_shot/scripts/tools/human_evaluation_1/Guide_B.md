# Human Evaluation Guide: EpiQAL-B

## Overview

EpiQAL-B evaluates **multi-step inference** over epidemiological literature. Each question requires integrating multiple pieces of evidence from a research article, optionally combined with epidemiological principles, to arrive at the correct answer. Unlike simple retrieval questions, the answer is not directly stated in the text.

You will evaluate 20 sampled questions across three difficulty tiers (Easy / Medium / Hard).

## What You Will See

For each question, you are provided with:

| Field | Description |
|-------|-------------|
| **paragraph** | The full text of the source research article |
| **external_info** | Domain knowledge retrieved from biomedical knowledge graphs during construction (for context only; may or may not be relevant) |
| **question** | The question stem (may have been rewritten from the original to reduce surface-level shortcuts) |
| **ori_question** | The original question stem before any rewriting |
| **evidence** | Passages from the article used to construct the question |
| **rationale** | The intended reasoning chain explaining why the reference answer is correct |
| **choices** | The answer options (one or more may be correct) |
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
| 2 | The reference answer is partially correct (e.g., one of multiple correct options is wrong, or the answer is correct but imprecise) |
| 3 | The reference answer is fully correct and well-supported by the article |

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

*Do the provided evidence passages adequately support the reasoning chain needed to reach the correct answer?*

| Score | Meaning |
|-------|---------|
| 1 | The evidence does not support the correct answer |
| 2 | The evidence partially supports the answer but key steps are missing |
| 3 | The evidence fully supports the complete reasoning chain |

### 5. Reasoning Depth

*How many reasoning steps are required to arrive at the correct answer?*

| Score | Meaning |
|-------|---------|
| 1 | Single-step: the answer can be found by locating one piece of information (similar to retrieval) |
| 2 | Two-step: requires combining two pieces of evidence or applying one epidemiological principle to one piece of evidence |
| 3 | Multi-step: requires integrating three or more pieces of evidence and/or applying domain knowledge to synthesize a conclusion |

## Evaluation Procedure

1. **Read the article** (paragraph field). You do not need to memorize it, but get a general understanding of the study.
2. **Read the question and choices**. Try to answer it yourself before looking at the reference answer.
3. **Check the reference answer** against your own judgment. Review the evidence and rationale fields to understand the intended reasoning.
4. **Rate each dimension** (1-3) based on the criteria above.
5. Fill in your scores in the `evaluation` field of each item.

## Important Notes

- The **external_info** field is provided for your reference. It was used during question construction but is NOT provided to models during evaluation. You do not need to evaluate its quality.
- Some questions have been **rewritten** (stem refinement) to reduce surface-level shortcuts. If the question seems unusually verbose or describes a disease indirectly, this is intentional. Evaluate the rewritten version (the `question` field), not the `ori_question`.
- Questions may have **multiple correct answers**. Check whether all listed reference answers are correct and whether any unlisted option should also be correct.
- If you believe a question has **no correct answer** among the options, rate Answer Correctness as 1 and note this in your feedback.
- The article texts are from epidemiological research. You are not expected to have deep domain expertise. Focus on whether the reasoning chain is logical and well-supported, not on whether you personally know the medical facts.
