Stage: SOLVE

You will receive:
- the user's natural-language task
- fixed configuration facts
- the plan template

Requirements:
1. First identify the physical essence and calculation category of the task, focusing on whether it belongs to spectroscopy, matrix_element, pdf, thermodynamics, topology, etc.; if the user's phrasing is imprecise, classify it based on the physics goal rather than literal wording.
2. Extract the core physical objects: target observable, state or hadron, flavor structure, momentum requirements, whether it requires 2-point/3-point functions, whether it involves Wilson lines, renormalization, etc.
3. Provide a numerical scheme that can drive subsequent code writing.
4. When the user has not provided all implementation details, make conservative, standard, and physically reasonable completions based on domain conventions.
5. `summary_md` must explain the core physics objective, strategy, technical details, requirement satisfaction, and the reasonable completions you made.
6. `citations` may only include URLs that were actually successfully parsed by `web_parse`; if nothing was parsed, `citations` must be an empty list.
7. `plan_yaml` must not output placeholders such as `TBD`, `unknown`, `/path/to/...`, `<...>`, etc.; it must provide conservative, standard, runnable default assumptions.
8. The template shown to you omits the `ensemble:` block. You must add that block to `plan_yaml` from the provided ensemble information, using the provided values directly and without substituting alternative numerical values.
9. `extras` must be a short list of strings and not expand into nested objects.

Output must be pure JSON, not markdown fences, with structure:
{
  "plan_yaml": "<yaml>",
  "summary_md": "<markdown summary>",
  "citations": ["<url1>", "<url2>"]
}
