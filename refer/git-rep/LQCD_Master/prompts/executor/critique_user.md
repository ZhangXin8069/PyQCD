Stage: EXECUTOR_CRITIQUE

Review criteria:
- `errors` should only list clear issues that will cause implementation failure, deviation from the physics target, invalid PyQUDA usage, or obvious submission flow errors.
- `warnings` should only list risks that require human confirmation or are not safe to automatically fix.
- do not automatically repair the code or output a revised implementation.
- strictly incorporate static analysis results; if you confirm some findings are valid, reflect them in `errors` or `warnings`.

Please output pure JSON:
{
  "summary": "<overall judgment, concise and direct>",
  "errors": ["<clear error 1>", "<clear error 2>"],
  "warnings": ["<risk 1>", "<risk 2>"],
  "notes": "<optional>"
}
