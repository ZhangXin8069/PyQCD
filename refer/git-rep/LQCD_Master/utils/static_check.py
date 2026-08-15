from __future__ import annotations

import ast
import builtins
import importlib
import inspect
from typing import Any


def analyze_bundle_static(
    *,
    main_program: str,
    test_submit_script: str,
    full_submit_script: str,
) -> dict[str, Any]:
    program_report = analyze_python_static(main_program)
    pyquda_report = analyze_pyquda_usage(main_program)
    test_report = analyze_shell_static(test_submit_script, script_name="submit_test.sh")
    full_report = analyze_shell_static(full_submit_script, script_name="submit_full.sh")

    return {
        "main_errors": list(program_report["errors"]) + list(pyquda_report["errors"]),
        "test_sh_errors": list(test_report["errors"]),
        "full_sh_errors": list(full_report["errors"]),
    }


def analyze_python_static(source: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename="main.py")
    except SyntaxError as exc:
        line = exc.lineno or 0
        col = exc.offset or 0
        message = exc.msg or "syntax error"
        errors.append(f"main.py:{line}:{col} syntax error: {message}")
        return {"errors": errors}

    visitor = _ModuleStaticAnalyzer()
    visitor.visit(tree)
    errors.extend(visitor.errors)

    compile_error = compile_python_source(source)
    if compile_error:
        errors.append(compile_error)

    return {"errors": errors}


def compile_python_source(source: str) -> str:
    try:
        compile(source, "main.py", "exec")
    except SyntaxError as exc:
        line = exc.lineno or 0
        col = exc.offset or 0
        message = exc.msg or "syntax error"
        return f"main.py:{line}:{col} syntax error during compile: {message}"
    return ""


def analyze_shell_static(source: str, *, script_name: str) -> dict[str, Any]:
    errors: list[str] = []
    stripped = source.strip()
    if not stripped:
        errors.append(f"{script_name}: empty script")
        return {"errors": errors}
    if "main.py" not in source:
        errors.append(f"{script_name}: expected main.py launch target not found")
    return {"errors": errors}


def analyze_pyquda_usage(source: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename="main.py")
    except SyntaxError:
        return {"errors": errors}

    analyzer = _PyQUDACallAnalyzer()
    analyzer.visit(tree)

    errors.extend(analyzer.errors)

    runtime_errors = _check_pyquda_runtime_signatures(analyzer.calls)
    errors.extend(runtime_errors)

    return {
        "errors": errors,
    }


def _check_pyquda_runtime_signatures(calls: list["_PyQUDACall"]) -> list[str]:
    errors: list[str] = []
    cache: dict[str, Any] = {}

    for call in calls:
        if not call.module_name:
            continue
        module = _import_module_cached(call.module_name, cache)
        if module is None:
            continue

        target = getattr(module, call.attr_name, _MISSING)
        if target is _MISSING:
            errors.append(
                f"main.py:{call.lineno} PyQUDA API missing: {call.module_name}.{call.attr_name}"
            )
            continue

        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue

        args = [object() for _ in range(call.positional_arg_count)]
        kwargs = {name: object() for name in call.keyword_arg_names}
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            errors.append(
                f"main.py:{call.lineno} invalid call to {call.module_name}.{call.attr_name}: {exc}"
            )
    return errors


def _import_module_cached(module_name: str, cache: dict[str, Any]) -> Any | None:
    if module_name in cache:
        return cache[module_name]
    try:
        cache[module_name] = importlib.import_module(module_name)
    except Exception:
        cache[module_name] = None
    return cache[module_name]


_MISSING = object()
_PYQUDA_ALIAS_TO_MODULE = {
    "core": "pyquda_utils.core",
    "io": "pyquda_utils.io",
    "source": "pyquda_utils.source",
    "phase_v2": "pyquda_utils.phase_v2",
    "gamma": "pyquda_utils.gamma",
    "pyquda": "pyquda",
}
_PATH_PARAMETER_NAMES = {
    "resource_path",
    "path",
    "cfg_path",
    "gauge_path",
    "output_path",
    "cache_path",
    "tunecache",
    "quda_resource_path",
}


class _PyQUDACall:
    def __init__(
        self,
        *,
        lineno: int,
        module_alias: str,
        module_name: str | None,
        attr_name: str,
        positional_arg_count: int,
        keyword_arg_names: list[str],
    ) -> None:
        self.lineno = lineno
        self.module_alias = module_alias
        self.module_name = module_name
        self.attr_name = attr_name
        self.positional_arg_count = positional_arg_count
        self.keyword_arg_names = keyword_arg_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineno": self.lineno,
            "module_alias": self.module_alias,
            "module_name": self.module_name,
            "attr_name": self.attr_name,
            "positional_arg_count": self.positional_arg_count,
            "keyword_arg_names": self.keyword_arg_names,
        }


class _PyQUDACallAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.calls: list[_PyQUDACall] = []
        self.alias_to_module: dict[str, str] = {}
        self.none_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            full_name = alias.name
            bound_name = alias.asname or full_name.split(".", 1)[0]
            if full_name in _PYQUDA_ALIAS_TO_MODULE.values():
                self.alias_to_module[bound_name] = full_name
            elif full_name == "pyquda_utils":
                self.alias_to_module[bound_name] = "pyquda_utils"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            bound_name = alias.asname or alias.name
            if module == "pyquda_utils" and alias.name in _PYQUDA_ALIAS_TO_MODULE:
                real_module = _PYQUDA_ALIAS_TO_MODULE[alias.name]
                self.alias_to_module[bound_name] = real_module
            elif module in _PYQUDA_ALIAS_TO_MODULE.values() and alias.name == "*":
                continue
            elif module in _PYQUDA_ALIAS_TO_MODULE.values():
                self.alias_to_module[bound_name] = module
            elif module == "pyquda":
                self.alias_to_module[bound_name] = "pyquda"
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value_is_none = isinstance(node.value, ast.Constant) and node.value.value is None
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value_is_none:
                    self.none_names.add(target.id)
                else:
                    self.none_names.discard(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                self.none_names.add(node.target.id)
            else:
                self.none_names.discard(node.target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = self._extract_call(node)
        if call is not None:
            self.calls.append(call)
            self._check_semantics(node, call)
        self.generic_visit(node)

    def _extract_call(self, node: ast.Call) -> _PyQUDACall | None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        if not isinstance(func.value, ast.Name):
            return None
        module_alias = func.value.id
        module_name = self.alias_to_module.get(module_alias)
        if module_name is None and module_alias not in _PYQUDA_ALIAS_TO_MODULE:
            return None
        if module_name is None:
            module_name = _PYQUDA_ALIAS_TO_MODULE.get(module_alias)
        keyword_names = [kw.arg for kw in node.keywords if kw.arg]
        return _PyQUDACall(
            lineno=node.lineno,
            module_alias=module_alias,
            module_name=module_name,
            attr_name=func.attr,
            positional_arg_count=len(node.args),
            keyword_arg_names=keyword_names,
        )

    def _check_semantics(self, node: ast.Call, call: _PyQUDACall) -> None:
        for kw in node.keywords:
            if kw.arg in _PATH_PARAMETER_NAMES:
                if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                    self.errors.append(
                        f"main.py:{node.lineno} {kw.arg} should be omitted or set to a valid path, not None"
                    )
                elif isinstance(kw.value, ast.Constant) and kw.value.value == "None":
                    self.errors.append(
                        f"main.py:{node.lineno} {kw.arg} should be omitted or set to a valid path, not string 'None'"
                    )
                elif isinstance(kw.value, ast.Name) and kw.value.id in self.none_names:
                    self.errors.append(
                        f"main.py:{node.lineno} {kw.arg} uses name {kw.value.id!r} known to be None"
                    )

        if call.module_name == "pyquda_utils.core" and call.attr_name == "invert":
            for kw in node.keywords:
                if kw.arg == "phase":
                    self.errors.append(
                        f"main.py:{node.lineno} core.invert should not receive keyword argument 'phase'"
                    )

        if call.module_name == "pyquda_utils.core" and call.attr_name == "getArrayModule":
            self.errors.append(
                f"main.py:{node.lineno} pyquda_utils.core.getArrayModule is not a supported API in this workflow"
            )


class _ModuleStaticAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.defined_names: set[str] = set(dir(builtins))
        self.none_names: set[str] = set()
        self.local_scopes: list[set[str]] = []

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_statements(node.body, reachable=True)

    def _visit_statements(self, statements: list[ast.stmt], *, reachable: bool) -> None:
        is_reachable = reachable
        for stmt in statements:
            if not is_reachable:
                continue
            self.visit(stmt)
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                is_reachable = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._define_name(alias.asname or alias.name.split(".", 1)[0], value_is_none=False)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._define_name(alias.asname or alias.name, value_is_none=False)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value_is_none = self._expr_is_none(node.value)
        for target in node.targets:
            self._register_target(target, value_is_none=value_is_none)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._register_target(node.target, value_is_none=self._expr_is_none(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._register_target(node.target, value_is_none=False)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._register_target(node.target, value_is_none=False)
        self._visit_statements(node.body, reachable=True)
        self._visit_statements(node.orelse, reachable=True)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._register_target(item.optional_vars, value_is_none=False)
        self._visit_statements(node.body, reachable=True)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        truth_value = self._constant_truth_value(node.test)
        if truth_value is True:
            self._visit_statements(node.body, reachable=True)
            self._visit_statements(node.orelse, reachable=False)
            return
        if truth_value is False:
            self._visit_statements(node.body, reachable=False)
            self._visit_statements(node.orelse, reachable=True)
            return
        self._visit_statements(node.body, reachable=True)
        self._visit_statements(node.orelse, reachable=True)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._check_builtin_signature(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)
        if isinstance(node.value, ast.Name) and node.value.id in self.none_names:
            self.errors.append(
                f"main.py:{node.lineno} attribute access on name {node.value.id!r} known to be None"
            )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.value, ast.Name) and node.value.id in self.none_names:
            self.errors.append(
                f"main.py:{node.lineno} subscript on name {node.value.id!r} known to be None"
            )

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.defined_names and not self._is_locally_defined(node.id):
            self.errors.append(f"main.py:{node.lineno} undefined name: {node.id}")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._define_name(node.name, value_is_none=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._define_name(node.name, value_is_none=False)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._define_name(node.name, value_is_none=False)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        truth_value = self._constant_truth_value(node.test)
        if truth_value is False:
            self._visit_statements(node.body, reachable=False)
            self._visit_statements(node.orelse, reachable=True)
            return
        self._visit_statements(node.body, reachable=True)
        self._visit_statements(node.orelse, reachable=True)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_statements(node.body, reachable=True)
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name:
                self._define_name(handler.name, value_is_none=False)
            self._visit_statements(handler.body, reachable=True)
        self._visit_statements(node.orelse, reachable=True)
        self._visit_statements(node.finalbody, reachable=True)

    def _register_target(self, target: ast.expr, *, value_is_none: bool) -> None:
        if isinstance(target, ast.Name):
            self._define_name(target.id, value_is_none=value_is_none)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_target(elt, value_is_none=False)
            return
        if isinstance(target, ast.Attribute):
            self.visit(target.value)
            return
        if isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)

    def _register_local_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            if self.local_scopes:
                self.local_scopes[-1].add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_local_target(elt)
            return
        if isinstance(target, ast.Starred):
            self._register_local_target(target.value)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_nodes: list[ast.expr],
    ) -> None:
        self.local_scopes.append(set())
        try:
            for generator in generators:
                self.visit(generator.iter)
                self._register_local_target(generator.target)
                for if_clause in generator.ifs:
                    self.visit(if_clause)
            for result_node in result_nodes:
                self.visit(result_node)
        finally:
            self.local_scopes.pop()

    def _define_name(self, name: str, *, value_is_none: bool) -> None:
        self.defined_names.add(name)
        if value_is_none:
            self.none_names.add(name)
        else:
            self.none_names.discard(name)

    def _expr_is_none(self, node: ast.AST | None) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    def _is_locally_defined(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.local_scopes))

    def _constant_truth_value(self, node: ast.AST) -> bool | None:
        if isinstance(node, ast.Constant):
            try:
                return bool(node.value)
            except Exception:
                return None
        if isinstance(node, ast.Name) and node.id in self.none_names:
            return False
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left = node.left
            right = node.comparators[0]
            op = node.ops[0]
            if isinstance(op, ast.Is):
                if self._expr_is_none(left) and self._expr_is_none(right):
                    return True
                if isinstance(left, ast.Name) and left.id in self.none_names and self._expr_is_none(right):
                    return True
                if isinstance(right, ast.Name) and right.id in self.none_names and self._expr_is_none(left):
                    return True
            if isinstance(op, ast.IsNot):
                if self._expr_is_none(left) and self._expr_is_none(right):
                    return False
                if isinstance(left, ast.Name) and left.id in self.none_names and self._expr_is_none(right):
                    return False
                if isinstance(right, ast.Name) and right.id in self.none_names and self._expr_is_none(left):
                    return False
        return None

    def _check_builtin_signature(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            return
        name = node.func.id
        positional_count = len(node.args)
        keyword_names = {kw.arg for kw in node.keywords if kw.arg}

        if name == "len" and positional_count != 1:
            self.errors.append(f"main.py:{node.lineno} len() expects exactly 1 positional argument")
        elif name == "range" and positional_count not in {1, 2, 3}:
            self.errors.append(f"main.py:{node.lineno} range() expects 1 to 3 positional arguments")
        elif name == "print" and any(kw not in {"sep", "end", "file", "flush"} for kw in keyword_names):
            self.errors.append(f"main.py:{node.lineno} print() received unsupported keyword argument")
        elif name == "open":
            if positional_count < 1 or positional_count > 8:
                self.errors.append(f"main.py:{node.lineno} open() received an invalid number of positional arguments")
