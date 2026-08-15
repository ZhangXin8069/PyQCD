Stage: CRITIQUE

Please make your review strict and focus on:
1. physics-task matching: verify whether the plan truly addresses the original physics goal rather than just a superficially similar template.
2. numerical feasibility: verify that ensemble/measurement/solver/observable/output are coherent, and that source/sink/operator/solver/observable relationships are consistent.
3. code executability: only flag issues that would cause executor code generation to produce incorrect or non-compilable code.

The output must focus on "problems, risks, and revision directions." Do not rewrite the entire plan or output a new `plan_yaml`.

IMPORTANT: Only flag issues that would cause the generated code to fail or produce wrong physics results. Missing statistical caveats, non-critical risks, and minor improvements are not errors. If the plan is executable and physically sound, return an empty issues list.

Return pure JSON:
{
  "issues": ["specific issue 1", "specific issue 2"],
  "risks": ["scientific or execution risk 1", "scientific or execution risk 2"],
  "revision_instructions": ["how to revise 1", "how to revise 2"]
}
