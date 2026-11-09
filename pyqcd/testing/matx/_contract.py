"""MatX `einsum` 契约数据（matx 1.1.0）与测试接口层。

本模块只写测试侧代码：不改动 `pyqcd/einsum/**`，也不假装能调用 MatX
（本机没有 MatX 源码，也没有 Python 绑定，`einsum` 是 C++ 模板接口）。
唯一权威依据是官方文档页 `DOC_URL`；页面自相矛盾之处原样登记在
`CONTRADICTIONS`，规范化解释取正文（较新、带示例），API 段记为待核实项。

三层内容：

- `DOC_CASES` / `LIMITATIONS` / `CONTRADICTIONS`：可断言的数据语料；
- `implicit_output()` / `infer_output_shape()` / `einsum_into()`：接口层，
  把 MatX 语义（显式输出、`...` 尺寸必须完全一致、不支持 size-1 广播）
  映射到 `numpy` 或 `pyqcd` backend；
- `availability()`：原生 MatX 可用性只读探针，任何异常都转成 reason 字符串。

异常层级：`MatXInterfaceError` > `MatXUnsupportedError` > `MatXShapeError`，
语义由宽到窄——「接口用错」宽于「MatX 语义不支持」宽于「尺寸不满足」。
"""
from __future__ import annotations

import ctypes.util
import os
from collections import Counter

import numpy as np

DOC_URL = 'https://nvidia.github.io/MatX/api/linalg/matvec/einsum.html'
DOC_VERSION = 'matx 1.1.0'
DOC_NAMESPACE = 'matx::cutensor'
DOC_SIGNATURE = 'einsum(const std::string &subscripts, const InT&... ops)'
FEATURE_MACRO = '-DMATX_EN_CUTENSOR=ON'
REQUIRED_CUTENSOR = 'cuTENSOR>=1.7.0'
REQUIRED_CUTENSORNET = 'cuTensorNet>=23.03.0.20'
MATX_ROOT_ENV = 'PYQCD_MATX_ROOT'

REQUIRED_ENVIRONMENT = (
    '需 cuTENSOR≥1.7.0 + cuTensorNet≥23.03.0.20 且 '
    '`-DMATX_EN_CUTENSOR=ON` 编译，并须在 GPU executor 上运行')

CALL_CONVENTION = (
    '本接口固定为 out 优先：einsum_into(out, subscripts, *operands)。文档页 '
    f'API 段给出的签名是 {DOC_SIGNATURE}，输出张量是 ops 之一；测试接口不改'
    '动该签名，只把「输出张量必须随调用给出、并接受类型与尺寸检查」这一点'
    '固定成第一个位置参数。')

MODE_EXPLICIT = 'explicit'
MODE_IMPLICIT = 'implicit'

STATUS_TESTED = 'tested'
STATUS_UNTESTED = 'untested'
STATUS_UNSUPPORTED = 'unsupported'
#: 文档声明已被实际源码推翻（文档段落过时），保留记录但不再作为断言依据。
STATUS_STALE = 'stale'

CASE_FIELDS = ('name', 'group', 'subscripts', 'shapes', 'out_shape', 'mode',
               'matx_status', 'source_note')

DOC_CASES = (
    {
        'name': 'contraction_3d',
        'group': 'contraction',
        'subscripts': 'ijk,jil->kl',
        'shapes': ((3, 4, 5), (4, 3, 2)),
        'out_shape': (5, 2),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文三维张量收缩示例：a{3,4,5}, b{4,3,2} → c{5,2}，'
            'einsum("ijk,jil->kl", a, b, c)。属于文档声明「支持并测试」的 '
            'tensor contraction。'),
    },
    {
        'name': 'inner_product',
        'group': 'inner',
        'subscripts': 'i,i->',
        'shapes': ((60,), (60,)),
        'out_shape': (),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文内积示例：a1{60}, b1{60} → c0{}，'
            'einsum("i,i->", a1, b1, c0)。属于文档声明「支持并测试」的 '
            'inner product。'),
    },
    {
        'name': 'gemm',
        'group': 'gemm',
        'subscripts': 'mk,kn->mn',
        'shapes': ((10, 20), (20, 10)),
        'out_shape': (10, 10),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文 GEMM 示例：a2{10,20}, b2{20,10} → c2{10,10}，'
            'einsum("mk,kn->mn", a2, b2, c2)。属于文档声明「支持并测试」的 '
            'GEMM。'),
    },
    {
        'name': 'gemm_transposed_out',
        'group': 'gemm',
        'subscripts': 'mk,kn->nm',
        'shapes': ((5, 20), (20, 10)),
        'out_shape': (10, 5),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文转置输出 GEMM 示例：a2{5,20}, b2{20,10} → c2{10,5}，'
            'einsum("mk,kn->nm", a2, b2, c2)。属于文档声明「支持并测试」的 '
            'GEMM 变体。'),
    },
    {
        'name': 'permute',
        'group': 'permute',
        'subscripts': 'ijkl->jlki',
        'shapes': ((5, 20, 4, 3),),
        'out_shape': (20, 3, 4, 5),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文置换示例：a{5,20,4,3} → b{20,3,4,5}，'
            'einsum("ijkl->jlki", a, b)。上游 '
            'test/00_tensor/EinsumTests.cu::Permute 覆盖了同一方程，'
            '按源码核验记为 tested。'),
    },
    {
        'name': 'sum_axis',
        'group': 'reduction',
        'subscripts': 'ij->j',
        'shapes': ((2, 3),),
        'out_shape': (3,),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文求和示例：a{2,3} → b{3}，einsum("ij->j", a, b)。'
            '上游 test/00_tensor/EinsumTests.cu::Sum 覆盖了同一方程，'
            '按源码核验记为 tested。'),
    },
    {
        'name': 'trace',
        'group': 'reduction',
        'subscripts': 'ii->',
        'shapes': ((10, 10),),
        'out_shape': (),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文迹示例：a2{10,10} → 标量{}，einsum("ii->", a2, c)。'
            '上游 test/00_tensor/EinsumTests.cu::Trace 与 '
            'TraceImplicitOutput 覆盖了该方程，按源码核验记为 tested。'
            '注意文档没有 `ii->i` 对角输出示例。'),
    },
    {
        'name': 'ellipsis_batch_gemm',
        'group': 'broadcast',
        'subscripts': '...ij,...jk->...ik',
        'shapes': ((5, 2, 3, 4), (5, 2, 4, 6)),
        'out_shape': (5, 2, 3, 6),
        'mode': MODE_EXPLICIT,
        'matx_status': STATUS_TESTED,
        'source_note': (
            '文档正文广播示例："...ij,...jk->...ik"，两个张量的 ... 均为 '
            '[5,2]，要求各张量中 ... 代表的尺寸完全一致。按正文（较新、带'
            '示例）记为 tested；页面末尾 API 段却称省略号尚未支持，'
            '见 CONTRADICTIONS。'),
    },
)

DOC_IMPLICIT_EXAMPLE = {
    'subscripts': 'ij,jk',
    'inferred_output': 'ik',
    'source_note': (
        '文档正文隐式模式段：省略 `->` 时按 NumPy 规则推断输出——在所有输入'
        '中恰好出现一次的字母按字母序进入输出，重复出现的字母求和；'
        '例 `ij,jk` 推断为 `ik`。'),
}

IMPLICIT_EXAMPLES = (
    (DOC_IMPLICIT_EXAMPLE['subscripts'],
     DOC_IMPLICIT_EXAMPLE['inferred_output']),
    ('i,i', ''),
    ('ij', 'ij'),
    ('ii', ''),
    ('ij,ij', ''),
    ('ij,jk,kl', 'il'),
    ('ijk,kj', 'i'),
    ('ab,bc,cd', 'ad'),
)

LIMITATIONS = (
    {
        'key': 'cpu-executor-unsupported',
        'kind': 'constraint',
        'statement': '不支持 host-based（CPU）executor，调用必须落在 GPU '
                     'executor 上。',
        'impact': '本机无 GPU/cuTENSOR，无法复现原生结果；测试只能用 numpy '
                  '参考实现验证语义，不能声称验证了 MatX 自身。',
        'source': '文档正文 executor 段（einsum 不支持 host-based executor）',
        'enforced_by': 'availability()',
    },
    {
        'key': 'ellipsis-shape-must-match',
        'kind': 'constraint',
        'statement': '`...` 代表的批维在各输入张量中尺寸必须完全一致'
                     '（cuTensorNet 要求共享维尺寸相等）。',
        'impact': '`...ij,...jk->...ik` 只在两侧 `...` 完全相同时可用；'
                  '批维维数不同即报错。',
        'source': '文档正文广播省略号段',
        'enforced_by': 'infer_output_shape(..., matx_semantics=True)',
    },
    {
        'key': 'no-size-1-broadcast',
        'kind': 'constraint',
        'statement': '不支持 NumPy 的 size-1 广播：共享维尺寸不等即报错，'
                     '不会把 1 拉伸到 N。',
        'impact': '`...i,...i->...` 在 (2,1,3)×(2,4,3) 上 numpy 返回 (2,4)，'
                  '而 MatX 语义下 unsupported；该差异必须显式记录，不能当作 '
                  'MatX 通过。',
        'source': '文档正文「cuTensorNet 要求共享维尺寸相等」段',
        'enforced_by': 'infer_output_shape(..., matx_semantics=True)',
    },
    {
        'key': 'optional-feature-and-versions',
        'kind': 'constraint',
        'statement': 'einsum 属可选特性：需 `-DMATX_EN_CUTENSOR=ON` 编译，'
                     '依赖 cuTENSOR>=1.7.0 与 cuTensorNet>=23.03.0.20；'
                     '`cutensor_DIR`/`cutensornet_DIR` 未指定时 CMake 会联网'
                     '下载依赖。',
        'impact': '本机未编译该特性，`availability()` 恒返回不可用；禁止用'
                  '联网安装替代。',
        'source': '文档正文可选特性与构建段',
        'enforced_by': 'availability()',
    },
    {
        'key': 'first-call-autotuning',
        'kind': 'engineering',
        'statement': '同一签名的首次调用会做 cuTENSOR/cuTensorNet 优化启发式'
                     '（明显更慢），后续调用只做收缩。',
        'impact': '任何计时类断言必须取第二次及以后的调用；本验证不做性能'
                  '断言，只登记该开销。',
        'source': '文档正文性能说明段',
        'enforced_by': 'record-only',
    },
    {
        'key': 'tested-scope-contraction-inner-gemm',
        'kind': 'constraint',
        'statement': '文档声明目前只支持并测试了 tensor contractions、'
                     'inner products、GEMMs；其他操作可能可用但未测试。'
                     '实际源码（refer/git-rep/MatX @ 0ac8664）的上游测试 '
                     'test/00_tensor/EinsumTests.cu 另覆盖 Permute、Sum、'
                     'Trace、隐式输出与 ellipsis 广播。',
        'impact': '文档措辞比实现保守；DOC_CASES 的 permute/sum/trace 按源码'
                  '核验记为 tested，但本机无 cuTENSOR，仍属源码级而非执行级证据。',
        'source': '文档正文「目前只支持并测试了 ...」段；'
                  'test/00_tensor/EinsumTests.cu 测试清单',
        'enforced_by': 'DOC_CASES[*].matx_status',
    },
    {
        'key': 'output-tensor-required',
        'kind': 'constraint',
        'statement': '显式模式必须随调用给出输出张量（用 `->` 直接命名输出'
                     '下标），MatX 会对输出张量做类型与尺寸检查。',
        'impact': '接口层 `einsum_into(out, ...)` 先校验 out 的尺寸/dtype/'
                  '可写性再计算；API 段进一步称 `->` 强制必需，见 '
                  'CONTRADICTIONS。',
        'source': '文档正文显式模式段 + 页面末尾自动生成 API 段',
        'enforced_by': 'einsum_into()',
    },
    {
        'key': 'diagonal-output-not-in-doc',
        'kind': 'constraint',
        'statement': '`ii->i` 这类对角输出不在文档用例内（只有 `ii->` 迹），'
                     '本接口在 MatX 语义下按不支持处理。',
        'impact': '对角提取在 numpy/pyqcd 上可用，但不能当作 MatX 通过；'
                  '登记为 unsupported。',
        'source': '文档正文用例清单（含 `ii->` 迹，无 `ii->i`）',
        'enforced_by': 'infer_output_shape(..., matx_semantics=True)',
    },
)

CONTRADICTIONS = (
    {
        'key': 'ellipsis-and-explicit-output',
        'topic': '`...` 广播是否可用，以及 `->` 是否强制必需',
        'doc_body_claim': (
            '正文（描述段，较新且附带示例）称支持 NumPy 风格广播省略号 `...`，'
            '要求各张量中 `...` 代表的尺寸完全一致，并可用 `->` 显式命名输出；'
            '示例 einsum("...ij,...jk->...ik", ...) 的两个张量 `...` 均为 '
            '[5,2]。'),
        'doc_api_claim': (
            '页面末尾自动生成的 API 段称 “Ellipses are not supported yet, but '
            'a variadic list of tensors for contraction is supported. '
            "The output operator '->' is required in MatX currently”。"),
        'resolution': (
            '以实际源码为准（refer/git-rep/MatX @ 0ac8664）：'
            'include/matx/transforms/einsum.h 的 InferImplicitOutput 与 '
            'ellipsis 解析同时支持隐式输出与 `...` 广播；'
            'include/matx/operators/einsum.h 的 EinsumOp::Exec 直接调用同一 '
            'einsum_impl。因此正文与实现一致，API 段的两条说法是过时注释，'
            '登记为 stale，不再作为待核实项。'),
        'api_claim_status': STATUS_STALE,
        'affected_cases': ('ellipsis_batch_gemm',),
        'affected_topics': ('ellipsis-broadcast', 'implicit-output-mode'),
        'recorded_in': 'pyqcd/testing/matx/_contract.py:CONTRADICTIONS',
    },
)


class MatXInterfaceError(ValueError):
    """接口层错误基类：MatX 语义下无法完成该调用。"""


class MatXUnsupportedError(MatXInterfaceError):
    """MatX 语义明确不支持的操作（size-1 广播、对角输出、隐式 + 省略号）。"""


class MatXShapeError(MatXUnsupportedError):
    """尺寸/形状不满足 MatX 要求（尺寸不一致、维数不符、输出形状不对）。"""


def _is_label(char):
    return ('a' <= char <= 'z') or ('A' <= char <= 'Z')


def _split_term(term):
    """把单个下标项拆成 (前缀标签, 是否含省略号, 后缀标签)。"""
    if '...' in term:
        if term.count('...') > 1:
            raise MatXInterfaceError(f'下标项 {term!r} 含多个省略号')
        pre, _, post = term.partition('...')
        has_ellipsis = True
    else:
        pre, post, has_ellipsis = term, '', False
    if '.' in pre or '.' in post:
        raise MatXInterfaceError(f'下标项 {term!r} 含非法省略号（应为连续 ...）')
    for char in pre + post:
        if not _is_label(char):
            raise MatXInterfaceError(
                f'下标项 {term!r} 含非法字符 {char!r}（只允许 a-z/A-Z 与 ...）')
    return pre, has_ellipsis, post


def _split_subscripts(subscripts):
    """返回 (输入项列表, 输出项 | None)；None 表示隐式模式。"""
    if not isinstance(subscripts, str):
        raise MatXInterfaceError(
            f'subscripts 必须是 str，收到 {type(subscripts).__name__}')
    if subscripts.count('->') > 1:
        raise MatXInterfaceError(f'subscripts 含多个 ->: {subscripts!r}')
    if any(char.isspace() for char in subscripts):
        raise MatXInterfaceError(f'subscripts 含空白字符: {subscripts!r}')
    if '->' in subscripts:
        left, _, right = subscripts.partition('->')
        if ',' in right:
            raise MatXInterfaceError(f'输出项不得含逗号: {subscripts!r}')
        out_term = right
    else:
        left, out_term = subscripts, None
    terms = left.split(',')
    for term in terms:
        _split_term(term)
    return terms, out_term


def implicit_output(subscripts):
    """按文档的 NumPy 隐式规则推断输出下标。

    规则：在所有输入中恰好出现一次的字母进入输出，按字母序排列；重复出现
    的字母被求和。含 `->` 的显式下标会被拒绝；含 `...` 的输入一律报错——
    本接口不处理隐式省略号推断，请改用显式 `->` 写法。
    """
    if not isinstance(subscripts, str):
        raise MatXInterfaceError(
            f'subscripts 必须是 str，收到 {type(subscripts).__name__}')
    if '->' in subscripts:
        raise MatXInterfaceError(
            f'implicit_output 只接受隐式模式（不含 ->），收到 {subscripts!r}')
    if '...' in subscripts:
        raise MatXUnsupportedError(
            f'隐式模式含省略号 ...，本接口不处理: {subscripts!r}。`...` 代表'
            '若干未命名批维，纯字母规则无法唯一表示其位置；请改用显式模式，'
            '例如 "...ij,...jk->...ik"。')
    terms = subscripts.split(',')
    counts = Counter()
    for term in terms:
        pre, has_ellipsis, post = _split_term(term)
        if has_ellipsis:  # pragma: no cover - 上面已拦截
            raise MatXUnsupportedError(f'隐式模式含省略号 ...: {subscripts!r}')
        counts.update(pre + post)
    return ''.join(sorted(label for label, count in counts.items()
                          if count == 1))


def normalize_subscripts(subscripts):
    """把隐式模式补成显式（`ij,jk` → `ij,jk->ik`）；显式模式原样返回。"""
    terms, out_term = _split_subscripts(subscripts)
    if out_term is not None:
        return subscripts
    return f'{subscripts}->{implicit_output(subscripts)}'


def _merge_label(label, old, new, matx_semantics, subscripts):
    """合并同一标签在不同操作数中的尺寸。"""
    if old == new:
        return old
    if matx_semantics:
        if old == 1 or new == 1:
            raise MatXUnsupportedError(
                f'标签 {label!r} 的共享维尺寸 {old} vs {new}（{subscripts!r}）'
                '：MatX 不支持 NumPy 的 size-1 广播，cuTensorNet 要求共享维'
                '尺寸相等。')
        raise MatXShapeError(
            f'标签 {label!r} 的共享维尺寸 {old} vs {new}（{subscripts!r}）：'
            'MatX 要求共享维尺寸相等。')
    if old == 1:
        return new
    if new == 1:
        return old
    raise MatXShapeError(
        f'标签 {label!r} 的尺寸 {old} 与 {new} 无法广播（{subscripts!r}）。')


def _merge_ellipsis(old, new, matx_semantics, subscripts):
    """合并 `...` 代表的批维形状。"""
    if tuple(old) == tuple(new):
        return tuple(old)
    if matx_semantics:
        if len(old) != len(new):
            raise MatXShapeError(
                f'... 的批维维数不一致：{tuple(old)} vs {tuple(new)}'
                f'（{subscripts!r}），MatX 要求各张量中 ... 尺寸完全一致。')
        if 1 in tuple(old) or 1 in tuple(new):
            raise MatXUnsupportedError(
                f'... 的批维 {tuple(old)} vs {tuple(new)}（{subscripts!r}）：'
                'MatX 不支持 size-1 广播。')
        raise MatXShapeError(
            f'... 的批维 {tuple(old)} vs {tuple(new)}（{subscripts!r}）：'
            'MatX 要求各张量中 ... 尺寸完全一致。')
    try:
        return tuple(np.broadcast_shapes(tuple(old), tuple(new)))
    except ValueError as exc:
        raise MatXShapeError(
            f'... 的批维 {tuple(old)} 与 {tuple(new)} 无法广播'
            f'（{subscripts!r}）: {exc}')


def _repeated_in_term(terms, label):
    """该标签是否在某一项内出现 >= 2 次（即被输入侧求和/取对角）。"""
    return any(term.count(label) > 1 for term in terms)


def infer_output_shape(subscripts, shapes, matx_semantics=True):
    """按 subscripts 与各操作数形状推断输出形状（显式/隐式均可）。

    `matx_semantics=True` 时执行 doc 契约里的 MatX 规则：共享维尺寸必须
    相等、`...` 批维必须完全一致、输出下标不得重复（无对角输出）。
    `matx_semantics=False` 时退回 NumPy 广播规则，仅用于对照记录。
    """
    terms, out_term = _split_subscripts(subscripts)
    shapes = [tuple(int(size) for size in shape) for shape in shapes]
    if out_term is None:
        out_term = implicit_output(subscripts)
    if len(terms) != len(shapes):
        raise MatXInterfaceError(
            f'操作数个数 {len(shapes)} 与下标项数 {len(terms)} 不一致: '
            f'{subscripts!r}')
    sizes = {}
    ellipsis_shape = ()
    ellipsis_seen = False
    for index, (term, shape) in enumerate(zip(terms, shapes)):
        pre, has_ellipsis, post = _split_term(term)
        labels = pre + post
        if has_ellipsis:
            ellipsis_ndim = len(shape) - len(labels)
            if ellipsis_ndim < 0:
                raise MatXShapeError(
                    f'第 {index} 个操作数形状 {shape} 比下标 {term!r} 的维数 '
                    f'{len(labels)} 还小（{subscripts!r}）')
            batch = shape[:ellipsis_ndim]
            if ellipsis_seen:
                ellipsis_shape = _merge_ellipsis(
                    ellipsis_shape, batch, matx_semantics, subscripts)
            else:
                ellipsis_shape, ellipsis_seen = batch, True
            dims = shape[ellipsis_ndim:]
        else:
            if len(shape) != len(labels):
                raise MatXShapeError(
                    f'第 {index} 个操作数形状 {shape} 与下标 {term!r} 的维数 '
                    f'{len(labels)} 不一致（{subscripts!r}）')
            dims = shape
        for label, size in zip(labels, dims):
            if label in sizes:
                sizes[label] = _merge_label(
                    label, sizes[label], size, matx_semantics, subscripts)
            else:
                sizes[label] = size
    out_pre, out_has_ellipsis, out_post = _split_term(out_term)
    out_labels = out_pre + out_post
    if matx_semantics and len(set(out_labels)) != len(out_labels):
        raise MatXUnsupportedError(
            f'输出下标重复（对角输出）不在 MatX 文档用例内: {out_term!r}'
            f'（{subscripts!r}），仅 `ii->` 这类迹被文档给出。')
    if matx_semantics:
        partial_trace = sorted({label for label in out_labels
                                if _repeated_in_term(terms, label)})
        if partial_trace:
            raise MatXUnsupportedError(
                f'输入下标在同一项内重复且出现在输出（对角/部分迹）不在 MatX '
                f'文档用例内: {partial_trace}（{subscripts!r}），文档只给出 '
                '`ii->` 全迹。')
    missing = [label for label in out_labels if label not in sizes]
    if missing:
        raise MatXInterfaceError(
            f'输出下标 {missing} 未在输入中出现（{subscripts!r}）')
    head = tuple(ellipsis_shape) if out_has_ellipsis else ()
    return head + tuple(sizes[label] for label in out_labels)


def _probe_once(environ, finder):
    notes = []
    root = (environ.get(MATX_ROOT_ENV) or '').strip()
    root_ok = False
    if root:
        root_ok = os.path.isdir(root)
        notes.append(f'{MATX_ROOT_ENV}={root}' + ('' if root_ok else ' 不是目录'))
    else:
        notes.append(f'未设置 {MATX_ROOT_ENV}')
    libraries = {}
    for name in ('cutensor', 'cutensornet'):
        try:
            libraries[name] = finder(name)
        except Exception as exc:  # noqa: BLE001 - 探针只报告，不抛出
            libraries[name] = None
            notes.append(f'find_library({name!r}) 失败: '
                         f'{type(exc).__name__}: {exc}')
        notes.append(f'find_library({name!r})={libraries[name]!r}')
    usable = root_ok and all(libraries.values())
    if usable:
        return True, ('; '.join(notes)
                      + '；检测依据仅限只读环境探针，本机未编译 '
                        'MATX_EN_CUTENSOR 特性、也未验证 GPU executor。')
    return False, '; '.join(notes) + f'；{REQUIRED_ENVIRONMENT}'


def availability(env=None, find_library=None):
    """原生 MatX `einsum` 可用性探针，返回 `(bool, reason)`。

    只做只读检测：`PYQCD_MATX_ROOT` 是否指向一个存在的目录、
    `ctypes.util.find_library` 能否解析 `cutensor`/`cutensornet`。任何异常
    都被吞掉并转成 reason 字符串；本函数不导入、不加载、不调用 MatX。
    """
    environ = os.environ if env is None else env
    finder = ctypes.util.find_library if find_library is None else find_library
    try:
        return _probe_once(environ, finder)
    except Exception as exc:  # noqa: BLE001 - 探针只报告，不抛出
        return False, (f'可用性探针异常: {type(exc).__name__}: {exc}；'
                       f'{REQUIRED_ENVIRONMENT}')


def _pyqcd_backend():
    """按需加载被测实现（本模块 import 时不触碰 `pyqcd.einsum`）。"""
    from ..einsum import _target

    module = _target.ready()
    if module is None:
        raise MatXInterfaceError(
            f'pyqcd.einsum 未就绪: {_target.skip_reason()}')
    return module


def einsum_into(out, subscripts, *operands, backend='numpy',
                matx_semantics=True):
    """按 MatX 的 out 优先约定把结果写入 `out` 并返回 `out`。

    `out` 为输出张量（第一个位置参数，见 `CALL_CONVENTION`）。显式模式要求
    subscripts 含 `->`，并先校验 `out` 的尺寸、可写性与 dtype；隐式模式先经
    `implicit_output()` 归一化成显式模式再走同一路径。
    """
    if not isinstance(out, np.ndarray):
        raise MatXInterfaceError(
            f'out 必须是 numpy.ndarray，收到 {type(out).__name__}')
    if not operands:
        raise MatXInterfaceError('至少需要一个输入张量')
    array = out
    normalized = normalize_subscripts(subscripts)
    shapes = [np.shape(operand) for operand in operands]
    expected = infer_output_shape(normalized, shapes,
                                 matx_semantics=matx_semantics)
    if tuple(array.shape) != tuple(expected):
        raise MatXShapeError(
            f'输出张量形状 {tuple(array.shape)} 与推断结果 {tuple(expected)} '
            f'不一致（{normalized!r}）')
    if not array.flags.writeable:
        raise MatXInterfaceError('输出张量不可写（writeable=False）')
    dtype = np.result_type(*[np.asarray(op).dtype for op in operands])
    if not np.can_cast(dtype, array.dtype, casting='same_kind'):
        raise MatXInterfaceError(
            f'输出张量 dtype {array.dtype} 无法按 same_kind 规则容纳计算 '
            f'dtype {dtype}')
    if backend == 'numpy':
        result = np.einsum(normalized, *operands)
    elif backend == 'pyqcd':
        result = _pyqcd_backend().contract(normalized, *operands, out=out)
    else:
        raise MatXInterfaceError(
            f'未知 backend {backend!r}（只支持 "numpy" 与 "pyqcd"）')
    if result is not out:
        np.copyto(array, np.asarray(result), casting='same_kind')
    return out
