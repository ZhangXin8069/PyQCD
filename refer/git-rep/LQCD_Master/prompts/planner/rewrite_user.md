Stage: REWRITE

Revision principles:
1. Use the original task's physical objective as the highest criterion.
2. Preserve reasonable parts and avoid over-rewriting.
3. Keep the structure stable: `plan_yaml` must keep the task / (if standard) physics / measurement / output sections intact. For freeform_mode tasks, rely on freeform_plan instead of the physics/measurement chain.
4. `extras` must remain a short list of strings and not become an object or nested configuration.
5. Facts already given in the fixed configuration must remain consistent with config when written into `plan_yaml`.
6. `summary_md` should be in English and independently summarize the new plan, not just list modifications.

Output must be pure JSON:
{
  "plan_yaml": "<yaml>",
  "summary_md": "<markdown summary>",
  "citations": ["<url1>", "<url2>"]
}
