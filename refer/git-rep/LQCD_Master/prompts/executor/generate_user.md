Stage: EXECUTOR

Requirements:

1. Generate a single shared Python main script, such as `main.py`, containing the actual physics computation logic.
2. The test and full submission commands should ideally only pass configuration path/label, configuration number, and parallelism; other runtime details should be placed back into `main.py`, not make the command line excessively long.
3. The test submit is for quick verification of implementation correctness; the full submit is for production-scale computation.
4. For input files, cfg availability, sample file naming, and directory structure, prioritize input facts and avoid assumptions without evidence.
5. Use clear, stable Python structure; make parameters, paths, and cfg ranges configurable where possible; clearly reflect the organization of source/sink/operator/solver/observable.
6. Do not output the full Slurm script text; the system will render submission scripts from fixed configuration.
7. You must strictly output the following JSON keys, without renaming keys, translating keys, using synonyms, or omitting `main_program`, `test_submit`, or `full_submit`.

The output must be pure JSON:
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
