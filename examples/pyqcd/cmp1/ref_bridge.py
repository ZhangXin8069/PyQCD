"""参考库加载桥：lqcddb 直接导入；donghx 脚本 AST 提取函数（剔除顶层驱动 IO）。"""
import ast
import os
import sys

ROOT = '/root/PyQCD'
LQCDBB_SRC = os.path.join(ROOT, 'refer/sush/lqcddb/src')
DONGHX_DIR = os.path.join(ROOT, 'refer/donghx')

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')


def load_lqcddb():
    if LQCDBB_SRC not in sys.path:
        sys.path.insert(0, LQCDBB_SRC)
    import lqcddb
    return lqcddb


def load_donghx(filename, func_names=None):
    path = os.path.join(DONGHX_DIR, filename)
    tree = ast.parse(open(path).read())
    def has_io(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                name = getattr(f, 'id', None) or getattr(f, 'attr', '') or ''
                if name in {'open', 'fromfile', 'savez', 'savetxt', 'print',
                            'input', 'exists', 'system', 'makedirs',
                            'getsize'}:
                    return True
            if isinstance(n, ast.Attribute) and n.attr == 'argv':
                return True
        return False

    def is_pure(node):
        return not any(isinstance(n, ast.Call) or
                       (isinstance(n, ast.Attribute) and n.attr == 'argv')
                       for n in ast.walk(node))

    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.ClassDef)):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if not has_io(node):
                body.append(node)
        elif isinstance(node, ast.For) and is_pure(node):
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    ns = {'__name__': 'donghx_' + filename.replace('.py', '').replace('.', '_')}
    if DONGHX_DIR not in sys.path:
        sys.path.insert(0, DONGHX_DIR)
    exec(compile(mod, path, 'exec'), ns)
    if func_names is not None:
        missing = [k for k in func_names if k not in ns]
        if missing:
            raise KeyError(f'{filename} 缺函数: {missing}')
        return {k: ns[k] for k in func_names}
    return ns
