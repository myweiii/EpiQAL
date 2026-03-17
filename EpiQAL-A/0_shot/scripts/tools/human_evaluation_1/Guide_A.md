# Human Evaluation Guide: EpiQAL-A

## Overview

EpiQAL-A evaluates **text-grounded factual recall**. Each question targets specific information that is explicitly stated in the source research article. The correct answer can be directly located in the text without requiring inference or external knowledge.

You will evaluate 20 randomly sampled questions.

## What You Will See

For each question, you are provided with:

| Field | Description |
|-------|-------------|
| **paragraph** | The full text of the source research article |
| **question** | The question stem |
| **evidence** | Passages from the article that support the correct answer |
| **rationale** | Explanation of why the reference answer is correct |
| **choices** | The answer options (one or more may be correct) |
| **ref_answers** | The reference correct answer(s), given as option indices |

## Evaluation Criteria

Rate each of the following dimensions on a **1-3 scale**.

### 1. Answer Correctness

*Is the reference answer actually correct?*

| Score | Meaning |
|-------|---------|
| 1 | The reference answer is incorrect or misleading |
| 2 | The reference answer is partially correct (e.g., one of multiple correct options is wrong, or the answer is correct but imprecise) |
| 3 | The reference answer is fully correct and directly supported by the article |

### 2. Distractor Quality

*Are the incorrect options plausible yet clearly distinguishable from the correct answer by reading the article?*

| Score | Meaning |
|-------|---------|
| 1 | Distractors are obviously wrong or irrelevant (no careful reading needed to eliminate them) |
| 2 | Distractors are reasonable but not strongly misleading |
| 3 | Distractors are drawn from the same article context and require precise reading to rule out |

### 3. Question Clarity

*Is the question stem clear, unambiguous, and well-formed?*

| Score | Meaning |
|-------|---------|
| 1 | The question is confusing, ambiguous, or poorly worded |
| 2 | The question is understandable but could be clearer |
| 3 | The question is clear, precise, and unambiguous |

### 4. Evidence Sufficiency

*Do the provided evidence passages adequately support the correct answer?*

| Score | Meaning |
|-------|---------|
| 1 | The evidence does not support the correct answer |
| 2 | The evidence partially supports the answer but is incomplete |
| 3 | The evidence directly and fully supports the correct answer |

## Evaluation Procedure

1. **Read the article** (paragraph field). You do not need to memorize it, but get a general understanding of the study.
2. **Read the question and choices**. Try to answer it yourself by locating the relevant information in the article.
3. **Check the reference answer** against your own judgment. Review the evidence and rationale fields.
4. **Rate each dimension** (1-3) based on the criteria above.
5. Fill in your scores in the `evaluation` field of each item.

## Important Notes

- EpiQAL-A is a **retrieval task**. The correct answer should be directly stated in the article. If you find that answering the question requires inference or external knowledge, this may indicate a quality issue.
- Distractors in EpiQAL-A are designed to be **passage-grounded confounders**: they are valid facts from the same article but refer to a different entity, time, place, or context than what the question asks. Check whether distractors are genuinely from the article rather than fabricated.
- Questions may have **multiple correct answers**. Check whether all listed reference answers are correct and whether any unlisted option should also be correct.
- The article texts are from epidemiological research. You are not expected to have deep domain expertise. Focus on whether the answer can be located in the text, not on whether you personally know the medical facts.