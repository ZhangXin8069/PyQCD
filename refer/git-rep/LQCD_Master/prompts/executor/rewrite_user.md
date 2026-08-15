Stage: EXECUTOR_REWRITER

Requirements:

- continue to maintain a single shared main script `main.py`;
- the test/full submission commands should ideally only pass the configuration path/configuration label, configuration number, and parallelism; other runtime details should be written back into `main.py`, not make the command line excessively long;
- do not output the complete Slurm script text, only output the submit spec;
- if there are only warnings and no errors, and the user has not requested a modification, do not perform a major rewrite just to fix warnings;
- if static analysis results are provided, prioritize fixing problems that are clearly valid;
- you must strictly follow the same JSON keys as in the executor generate stage, without changing key names, translating keys, using synonyms, or omitting `main_program`, `test_submit`, or `full_submit`.

The output format is identical to executor generate stage:
{
"main_program": "<python code>",
"test_submit": {
"job": {"name": "<job name>", "output": "<log path>", "error": "<log path>", "array": "<optional array spec>", "time": "<HH:MM:SS>"},
"run": {"program": "main.py", "args": ["<arg1>", "<arg2>"]}
},
"full_submit": {
"job": {"name": "<job name>", "output": "<log path>", "error": "<log path>", "array": "<optional array spec>", "time": "<HH:MM:SS>"},
"run": {"program": "main.py", "args": ["<arg1>", "<arg2>"]}
},
"notes": "<brief>"
}
