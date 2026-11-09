"""Einsums 字符串 ``einsum`` 的契约数据与 spec 层（不加载任何原生库）。

本模块只做三件事：

- 登记可断言的数据语料：``DOC_URL`` / ``SPEC_SYNTAX`` / ``DOC_CASES`` /
  ``LIMITATIONS``；
- 解析与规范化 spec：``parse_einsums_spec`` / ``numpy_to_einsums`` /
  ``normalize_spec`` / ``einsums_to_numpy``；
- 按标签一致性校验形状：``check_shapes`` / ``infer_output_shape``。

Einsums 有两套 einsum：编译期模板版与运行时字符串版，Python 与
ComputeGraph 只能用后者（``einsums::compute_graph::einsum``）。字符串 spec
在 Einsums 里是**数据**而不是类型，图优化可以重写它，所以 Python 侧原样透传，
只做语法与形状校验，绝不改写用户写法。

本模块没有原生依赖，import 时不触碰文件系统。
"""
from __future__ import annotations

import re
from collections import Counter

DOC_URL = 'https://einsums.github.io/Einsums/latest/'
DOC_VERSION = 'Einsums main (docs latest)'
DOC_EINSUM_URL = (
    'https://einsums.github.io/Einsums/latest/libs/Einsums/ComputeGraph/docs/'
    'string_einsum.html')
DOC_SOURCE_URL = 'https://github.com/Einsums/Einsums'
DOC_NAMESPACE = 'einsums::compute_graph'
DOC_SIGNATURE = 'cg::einsum("ij <- ik ; kj", &C, A, B)'

#: 文档给出的签名变体（前因子版与记号版），桥只实现基础版。
SIGNATURE_VARIANTS = (
    {
        'form': 'cg::einsum(spec, &C, A, B)',
        'meaning': 'C = A B（默认 beta=0、alpha=1）',
        'source': 'string_einsum.html#prefactors',
        'bridge': 'supported',
    },
    {
        'form': 'cg::einsum(spec, beta, &C, alpha, A, B)',
        'meaning': 'C = beta * C + alpha * A B',
        'source': 'string_einsum.html#prefactors',
        'bridge': 'unsupported（ABI 无 alpha/beta 槽位）',
    },
    {
        'form': 'cg::einsum("ik;kj -> ij", &C, A, B)',
        'meaning': 'numpy 记号，输出写在右侧',
        'source': 'string_einsum.html#notation',
        'bridge': 'supported（Python 层 normalize 成箭头写法）',
    },
    {
        'form': 'cg::einsum(cg::EinsumFormatString(spec), &C, A, B)',
        'meaning': '运行时构造的 spec 需要显式校验包装',
        'source': 'string_einsum.html#compile-time-validation',
        'bridge': 'n/a（桥在 C++ 侧直接接收 const char*）',
    },
)

ARROW_LEFT = '<-'
ARROW_RIGHT = '->'
OPERAND_SEPARATOR = ';'

SPEC_SYNTAX = (
    'Einsums 字符串 einsum 语法（docs latest，ComputeGraph/string_einsum）：\n'
    '  箭头写法   : "<输出> <- <输入1> ; <输入2>"   输出写在最前面，`<-` 分隔\n'
    '  numpy 写法 : "<输入1> ; <输入2> -> <输出>"   输出写在最后面，`->` 分隔\n'
    '  操作数分隔 : `;`（两种写法都是）；空白全部忽略\n'
    '  单字符模式 : 没有逗号时每个字母是一个标签，如 "ij <- ik ; kj"\n'
    '  多字符模式 : 出现逗号时逗号是**标签**分隔符（不是操作数分隔符），\n'
    '               如 "mu,nu <- mu,rho ; rho,nu"；本桥未验证该模式\n'
    '  标量输出   : 输出段留空，如 " <- i ; i"（DOT/内积）\n'
    '  置换算子   : P(i/j) 等只出现在多字符/单字符箭头写法里，本桥不支持\n'
    '  本测试层   : 只接受字母标签，`...` 一律拒绝（桥只支持字母标签）'
)

STATUS_TESTED = 'tested'
STATUS_UNTESTED = 'untested'

SOURCE_DOCUMENTED = 'documented'
SOURCE_INFERRED = 'inferred'

#: ``DOC_CASES`` 每条必备字段；缺字段由 ``verify.py`` 的 doc-case 组判失败。
CASE_FIELDS = ('name', 'group', 'spec', 'numpy_spec', 'shapes', 'out_shape',
               'dtype', 'source', 'status', 'source_note', 'native_check',
               'native_note')

_ASCII_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_LETTERS_RE = re.compile(r'^[A-Za-z]+$')
_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')


class EinsumsError(ValueError):
    """本包所有契约/接口错误的基类。"""


class EinsumsSpecError(EinsumsError):
    """spec 字符串不合法（箭头、分号、标签集合等）。"""


class EinsumsShapeError(EinsumsError):
    """标签与形状不一致（rank、同名标签尺寸、负维等）。"""


def _require_spec(spec):
    """校验 spec 是非空字符串，返回去掉首尾空白的文本。"""
    if not isinstance(spec, str):
        raise EinsumsSpecError(
            f'spec 必须是 str，收到 {type(spec).__name__}: {spec!r}')
    text = spec.strip()
    if not text:
        raise EinsumsSpecError('spec 为空字符串')
    return text


def _reject_unsupported(text):
    """显式拒绝 `...` 与 P(...) 置换算子，并说明桥的能力边界。"""
    if '...' in text or '\u2026' in text:
        raise EinsumsSpecError(
            f'本桥只支持字母标签，不支持 `...` 省略号批维（spec={text!r}）；'
            'Einsums 字符串 spec 也没有省略号语法，请先把批维展开成显式字母')
    if 'P(' in text:
        raise EinsumsSpecError(
            f'P(...) 置换算子不在本桥范围内（spec={text!r}）；'
            '它是耦合簇残差写法，需要 Einsums 的完整 spec 解析器，'
            '本测试层只覆盖字母标签的收缩/转置/求和')


def _duplicates(labels):
    """返回按首次出现顺序排列的重复标签。"""
    seen = set()
    repeated = []
    for label in labels:
        if label in seen and label not in repeated:
            repeated.append(label)
        seen.add(label)
    return repeated


def _parse_labels(segment, multi, role, allow_empty):
    """把一段（输出段或某个输入段）切成标签元组。"""
    text = segment.strip()
    if not text:
        if allow_empty:
            return ()
        raise EinsumsSpecError(f'{role}段为空（操作数不能为空）: {segment!r}')
    if multi:
        tokens = [token.strip() for token in text.split(',')]
        for token in tokens:
            if not token:
                raise EinsumsSpecError(f'{role}段含空标签: {segment!r}')
            if not _NAME_RE.match(token):
                raise EinsumsSpecError(
                    f'{role}段含非法标签名 {token!r}: {segment!r}'
                    '（多字符模式只接受字母开头的字母/数字/下划线）')
        return tuple(tokens)
    if not _LETTERS_RE.match(text):
        bad = sorted({ch for ch in text if ch not in _ASCII_LETTERS})
        raise EinsumsSpecError(
            f'{role}段含非字母字符 {bad}: {segment!r}'
            '（单字符模式只接受 a-zA-Z 字母标签）')
    return tuple(text)


def parse_einsums_spec(spec):
    """解析 Einsums 箭头写法，返回 ``(out_labels, [in_labels, ...])``。

    只接受 ``"<输出> <- <输入1> ; <输入2>"``：output 在前、``;`` 分隔输入。
    标量输出合法（输出段为空，例如 ``" <- i ; i"``）。numpy 写法请先过
    :func:`normalize_spec` 或 :func:`numpy_to_einsums`。

    拒绝的情形（抛 :class:`EinsumsSpecError`）：缺箭头、箭头重复、两种箭头
    并存、输入段为空、没有输入、输出标签重复、输出标签不在输入里、非字母
    标签、``...`` 与 ``P(...)``。
    """
    text = _require_spec(spec)
    _reject_unsupported(text)
    if ARROW_LEFT in text and ARROW_RIGHT in text:
        raise EinsumsSpecError(
            f'`<-` 与 `->` 不能同时出现: {text!r}'
            '（Einsums 两种写法二选一，自动识别）')
    if ARROW_RIGHT in text:
        raise EinsumsSpecError(
            f'parse_einsums_spec 只接受箭头写法 "<输出> <- <输入1> ; <输入2>"，'
            f'收到 numpy 写法 {text!r}；请先调用 normalize_spec()/'
            'numpy_to_einsums() 转换')
    if ARROW_LEFT not in text:
        raise EinsumsSpecError(
            f'spec 缺少箭头: {text!r}；Einsums 写法是 "out <- in1 ; in2"，'
            'numpy 写法是 "in1,in2->out"')
    if text.count(ARROW_LEFT) != 1:
        raise EinsumsSpecError(f'箭头 `<-` 只能出现一次: {text!r}')

    left, right = text.split(ARROW_LEFT)
    multi = ',' in text
    out_labels = _parse_labels(left, multi, '输出', allow_empty=True)
    if not right.strip():
        raise EinsumsSpecError(
            f'缺少输入操作数（`<-` 右侧为空）: {text!r}；'
            '标量输出也要写成 " <- i ; i"')

    inputs = []
    for index, segment in enumerate(right.split(OPERAND_SEPARATOR)):
        inputs.append(_parse_labels(segment, multi, f'第 {index + 1} 个输入',
                                    allow_empty=False))

    repeated = _duplicates(out_labels)
    if repeated:
        raise EinsumsSpecError(
            f'输出标签重复 {repeated}: {text!r}（对角输出不是 Einsums 写法）')
    seen = set()
    for labels in inputs:
        seen.update(labels)
    surplus = [label for label in out_labels if label not in seen]
    if surplus:
        raise EinsumsSpecError(
            f'输出标签 {surplus} 没有出现在任何输入里: {text!r}')
    return out_labels, inputs


def _canonical(out_labels, in_labels, multi=False):
    """拼回规范 Einsums 写法（``<-``、``;``、单空格）。"""
    joiner = ',' if multi else ''
    head = joiner.join(out_labels)
    body = ' ; '.join(joiner.join(labels) for labels in in_labels)
    return f'{head} {ARROW_LEFT} {body}'


def numpy_to_einsums(spec):
    """把 numpy 写法 ``"ik,kj->ij"``（或隐式输出）转成 ``"ij <- ik ; kj"``。

    操作数分隔符接受 ``,`` 与 ``;``（文档里两种都出现过）。省略 ``->`` 时按
    numpy 隐式规则推断输出：在每个输入中恰好出现一次的标签进入输出，并按
    字母序排列。
    """
    text = _require_spec(spec)
    _reject_unsupported(text)
    if ARROW_LEFT in text:
        raise EinsumsSpecError(
            f'{text!r} 已经是 Einsums 箭头写法；请直接用 normalize_spec()')
    if text.count(ARROW_RIGHT) > 1:
        raise EinsumsSpecError(f'箭头 `->` 只能出现一次: {text!r}')

    if ARROW_RIGHT in text:
        left, right = text.split(ARROW_RIGHT)
        explicit = True
    else:
        left, right, explicit = text, None, False

    operands = [part.strip() for part in re.split(r'[;,]', left.strip())]
    if not operands or any(not part for part in operands):
        raise EinsumsSpecError(
            f'输入操作数列表为空或含空段: {text!r}；'
            '写法是 "in1,in2->out" 或隐式 "in1,in2"')

    in_labels = []
    for index, operand in enumerate(operands):
        if not _LETTERS_RE.match(operand):
            bad = sorted({ch for ch in operand if ch not in _ASCII_LETTERS})
            raise EinsumsSpecError(
                f'第 {index + 1} 个输入含非字母字符 {bad}: {operand!r}'
                '（本桥只支持字母标签）')
        in_labels.append(tuple(operand))

    if explicit:
        head = right.strip()
        if head and not _LETTERS_RE.match(head):
            bad = sorted({ch for ch in head if ch not in _ASCII_LETTERS})
            raise EinsumsSpecError(
                f'输出含非字母字符 {bad}: {head!r}（本桥只支持字母标签）')
        out_labels = tuple(head)
    else:
        counts = Counter(label for labels in in_labels for label in labels)
        out_labels = tuple(sorted(label for label, count in counts.items()
                                  if count == 1))

    repeated = _duplicates(out_labels)
    if repeated:
        raise EinsumsSpecError(
            f'输出标签重复 {repeated}: {text!r}（numpy 对角输出不是 Einsums 写法）')
    seen = set()
    for labels in in_labels:
        seen.update(labels)
    surplus = [label for label in out_labels if label not in seen]
    if surplus:
        raise EinsumsSpecError(
            f'输出标签 {surplus} 没有出现在任何输入里: {text!r}')
    return _canonical(out_labels, in_labels)


def normalize_spec(spec):
    """两种写法都接受，统一返回 Einsums 写法 ``"out <- in1 ; in2"``。

    箭头写法原样规范化（校验后重排空白）；numpy 写法经
    :func:`numpy_to_einsums` 转换。``...`` 一律拒绝：本桥只支持字母标签。
    """
    text = _require_spec(spec)
    _reject_unsupported(text)
    if ARROW_LEFT in text:
        out_labels, in_labels = parse_einsums_spec(text)
        return _canonical(out_labels, in_labels, multi=',' in text)
    converted = numpy_to_einsums(text)
    out_labels, in_labels = parse_einsums_spec(converted)
    return _canonical(out_labels, in_labels)


def einsums_to_numpy(spec):
    """反向转换：``"ij <- ik ; kj"`` → ``"ik,kj->ij"``（只支持单字符标签）。"""
    out_labels, in_labels = parse_einsums_spec(normalize_spec(spec))
    labels = list(out_labels) + [label for labels in in_labels
                                 for label in labels]
    multi = sorted({label for label in labels if len(label) != 1})
    if multi:
        raise EinsumsSpecError(
            f'多字符标签 {multi} 无法映射到 numpy 单字符下标')
    body = ','.join(''.join(labels) for labels in in_labels)
    return f'{body}{ARROW_RIGHT}{"".join(out_labels)}'


def _extent(value, where):
    """把一维尺寸标准化成非负 int；非法抛 ``EinsumsShapeError``。"""
    try:
        extent = int(value)
    except (TypeError, ValueError):
        raise EinsumsShapeError(f'{where} 不是整数: {value!r}') from None
    if extent != value:
        raise EinsumsShapeError(f'{where} 不是整数值: {value!r}')
    if extent < 0:
        raise EinsumsShapeError(f'{where} 为负: {extent}')
    return extent


def _merge(sizes, label, extent, where):
    """把 ``label -> extent`` 并入已知尺寸表，冲突时抛错。"""
    known = sizes.get(label)
    if known is None:
        sizes[label] = extent
    elif known != extent:
        raise EinsumsShapeError(
            f'标签 {label!r} 尺寸不一致: 已见 {known}，{where} 给出 {extent}')


def check_shapes(out_labels, in_labels, shapes, label_sizes=None):
    """按标签一致性校验每个输入维与标签尺寸，返回 ``{label: size}``。

    ``in_labels`` 为每个输入的标签元组（与 ``shapes`` 一一对应），``shapes``
    为每个输入的形状。``label_sizes`` 可额外给出已确定的尺寸（例如外部已知
    的批维），与本轮观测冲突时抛 :class:`EinsumsShapeError`。

    非法情形：标签组数与形状个数不符、rank 与标签数不符、同一标签在不同
    位置尺寸不同、负维、非整数维、输出标签在任何输入里都没有出现过。
    """
    if in_labels is None and label_sizes is None:
        raise EinsumsShapeError(
            '至少需要 in_labels 或 label_sizes 之一，否则无法定位标签尺寸')

    sizes = {}
    if label_sizes is not None:
        for label, size in dict(label_sizes).items():
            _merge(sizes, str(label), _extent(size, f'label_sizes[{label!r}]'),
                   'label_sizes')

    if in_labels is not None:
        in_labels = [tuple(labels) for labels in in_labels]
        shapes = [tuple(shape) for shape in shapes]
        if len(in_labels) != len(shapes):
            raise EinsumsShapeError(
                f'标签组数 {len(in_labels)} 与形状个数 {len(shapes)} 不一致')
        for index, (labels, shape) in enumerate(zip(in_labels, shapes),
                                               start=1):
            if len(labels) != len(shape):
                raise EinsumsShapeError(
                    f'第 {index} 个操作数 rank={len(shape)} 与标签 '
                    f'{labels} 的个数 {len(labels)} 不一致')
            for label, extent in zip(labels, shape):
                _merge(sizes, label,
                       _extent(extent, f'第 {index} 个操作数的 {label!r} 维'),
                       f'第 {index} 个操作数')

    out_labels = tuple(out_labels)
    repeated = _duplicates(out_labels)
    if repeated:
        raise EinsumsShapeError(f'输出标签重复 {repeated}')
    missing = [label for label in out_labels if label not in sizes]
    if missing:
        raise EinsumsShapeError(
            f'输出标签 {missing} 没有任何已知尺寸，无法推断输出形状')
    return sizes


def infer_output_shape(out_labels, shapes, label_sizes=None, in_labels=None):
    """推断输出形状：``tuple(sizes[label] for label in out_labels)``。

    ``in_labels`` 给出时从 ``shapes`` 推导标签尺寸；否则必须有
    ``label_sizes``（或两者都给，此时交叉校验）。
    """
    sizes = check_shapes(out_labels, in_labels, shapes, label_sizes)
    return tuple(sizes[label] for label in out_labels)


#: 语义语料。``source='documented'`` 只用于文档原文出现的写法；
#: ``status='untested'`` 表示本机没有 Einsums（也没有可执行的原生桥）时
#: 一律未实测——不冒充官方已测。
DOC_CASES = (
    {
        'name': 'gemm_7x7',
        'group': 'gemm',
        'spec': 'ij <- ik ; kj',
        'numpy_spec': 'ik,kj->ij',
        'shapes': ((7, 7), (7, 7)),
        'out_shape': (7, 7),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            '文档首页（Getting Started 段，与仓库 README 同源）的入呼示例：'
            'A/B/C 都是 7×7，cg::einsum("ij <- ik ; kj", &C, A, B)；该例声明'
            '路由到单次 gemm。string_einsum 页的 Supported Patterns 也以它'
            '作为 GEMM 范式。'),
        'native_check': True,
        'native_note': 'nops=2、rank=2、out_ndim=2，落在冻结 ABI 内',
    },
    {
        'name': 'contraction_3d',
        'group': 'contraction',
        'spec': 'kl <- ijk ; jil',
        'numpy_spec': 'ijk,jil->kl',
        'shapes': ((3, 4, 5), (4, 3, 2)),
        'out_shape': (5, 2),
        'dtype': 'complex128',
        'source': SOURCE_INFERRED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'numpy 常见三维收缩 a{3,4,5}、b{4,3,2} → c{5,2}；'
            'Einsums 文档只承诺字符串派发能处理 rank 3+，未给这条具体用例。'),
        'native_check': True,
        'native_note': 'nops=2、rank=3、out_ndim=2；两个输入 rank 相同',
    },
    {
        'name': 'inner_product',
        'group': 'inner',
        'spec': ' <- i ; i',
        'numpy_spec': 'i,i->',
        'shapes': ((60,), (60,)),
        'out_shape': (),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 DOT 范式 " <- i ; i"；'
            'tutorial_einsum.html 说明空输出写进 rank-0 张量并走单次 DOT。'
            '本机无 Einsums，未实测。'),
        'native_check': True,
        'native_note': 'nops=2、rank=1、out_ndim=0（标量输出）；未在本机实测',
    },
    {
        'name': 'transpose_4d',
        'group': 'permutation',
        'spec': 'jlki <- ijkl',
        'numpy_spec': 'ijkl->jlki',
        'shapes': ((2, 3, 4, 5),),
        'out_shape': (3, 5, 4, 2),
        'dtype': 'complex128',
        'source': SOURCE_INFERRED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'numpy 四维转置 ijkl->jlki；Einsums 文档把转置写成 '
            'permute("ijkl->jlki", ...) 而不是 einsum，且实际源码 '
            'ComputeGraph/Operations.hpp 没有一元 cg::einsum 重载。'),
        'native_check': False,
        'native_note': 'Einsums 无一元 cg::einsum；转置须走 cg::permute，'
                       '故不向桥断言',
    },
    {
        'name': 'sum_axis',
        'group': 'reduction',
        'spec': 'j <- ij',
        'numpy_spec': 'ij->j',
        'shapes': ((3, 4),),
        'out_shape': (4,),
        'dtype': 'complex128',
        'source': SOURCE_INFERRED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'numpy 按轴求和 ij->j；Einsums 的 cg::einsum 只有二元重载，'
            '单操作数归约不在其 API 内（源码 Operations.hpp 已核实）。'),
        'native_check': False,
        'native_note': 'Einsums 无一元 cg::einsum；一元归约不在桥内',
    },
    {
        'name': 'trace',
        'group': 'reduction',
        'spec': ' <- ii',
        'numpy_spec': 'ii->',
        'shapes': ((5, 5),),
        'out_shape': (),
        'dtype': 'complex128',
        'source': SOURCE_INFERRED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'numpy 迹 ii->（输入标签重复）；Einsums 文档未列出重复输入标签，'
            '本测试层解析器支持它，但不向原生桥断言。'),
        'native_check': False,
        'native_note': '重复输入标签（对角/迹）超出 Einsums 文档给出的模式，'
                       '原生语义未验证，故不向桥断言',
    },
    {
        'name': 'gemv',
        'group': 'gemv',
        'spec': 'i <- ik ; k',
        'numpy_spec': 'ik,k->i',
        'shapes': ((4, 5), (5,)),
        'out_shape': (4,),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 GEMV 范式 "i <- ik ; k"；'
            '文档声明该模式由字符串派发处理。本机无 Einsums，未实测。'),
        'native_check': False,
        'native_note': '冻结 ABI 只有单个 INPUT_RANK 槽位，两个操作数 rank 必须'
                       '相同（2 vs 1 无法表达），故不向桥断言',
    },
    {
        'name': 'outer_product',
        'group': 'outer',
        'spec': 'ij <- i ; j',
        'numpy_spec': 'i,j->ij',
        'shapes': ((4,), (6,)),
        'out_shape': (4, 6),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 GER 范式 "ij <- i ; j"；'
            'tutorial_einsum.html 亦给出 "i;j->ij"。本机无 Einsums，未实测。'),
        'native_check': True,
        'native_note': 'nops=2、rank=1、out_ndim=2；未在本机实测',
    },
    {
        'name': 'hadamard',
        'group': 'elementwise',
        'spec': 'ij <- ij ; ij',
        'numpy_spec': 'ij,ij->ij',
        'shapes': ((4, 5), (4, 5)),
        'out_shape': (4, 5),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 direct product 范式'
            ' "ij <- ij ; ij"（Python 写法 "ij;ij->ij"）。本机无 Einsums，'
            '未实测。'),
        'native_check': True,
        'native_note': 'nops=2、rank=2、out_ndim=2；未在本机实测',
    },
    {
        'name': 'contraction_rank3_pattern',
        'group': 'contraction',
        'spec': 'il <- ijk ; jkl',
        'numpy_spec': 'ijk,jkl->il',
        'shapes': ((3, 4, 5), (4, 5, 6)),
        'out_shape': (3, 6),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 "Higher-rank contractions '
            '(rank 3+)" 第一条原文：cg::einsum("il <- ijk ; jkl", &C, A, B)。'),
        'native_check': True,
        'native_note': 'nops=2、两个输入同 rank=3、out_ndim=2；未在本机实测',
    },
    {
        'name': 'contraction_rank4_pattern',
        'group': 'contraction',
        'spec': 'ijkl <- ijp ; klp',
        'numpy_spec': 'ijp,klp->ijkl',
        'shapes': ((4, 4, 3), (4, 4, 3)),
        'out_shape': (4, 4, 4, 4),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'string_einsum.html Supported Patterns 的 "Higher-rank contractions '
            '(rank 3+)" 第二条原文：cg::einsum("ijkl <- ijp ; klp", &C, A, B)。'),
        'native_check': True,
        'native_note': 'nops=2、两个输入同 rank=3、out_ndim=4（桥上限）；未实测',
    },
    {
        'name': 'multi_operand_three',
        'group': 'contraction',
        'spec': 'il <- ij ; jk ; kl',
        'numpy_spec': 'ij,jk,kl->il',
        'shapes': ((100, 2), (2, 100), (100, 3)),
        'out_shape': (100, 3),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'tutorial_einsum.html "More Than Two Operands" 原文：'
            'cg::einsum("ij;jk;kl->il", &D, A, B, C)，并说明 Einsums 按最省'
            '算术的次序两两收缩。'),
        'native_check': False,
        'native_note': '桥上限 nops ∈ {1,2}，三操作数无法经冻结 ABI 表达',
    },
    {
        'name': 'diagonal_in_operand',
        'group': 'contraction',
        'spec': 'ij <- iik ; kj',
        'numpy_spec': 'iik,kj->ij',
        'shapes': ((4, 4, 6), (6, 5)),
        'out_shape': (4, 5),
        'dtype': 'complex128',
        'source': SOURCE_DOCUMENTED,
        'status': STATUS_UNTESTED,
        'source_note': (
            'tutorial_einsum.html 原文 "An index repeated within one operand is '
            'a diagonal rather than a contraction ... \\"iik;kj->ij\\" still runs '
            'on the packed contraction backend"，说明重复输入标签按跨步对角视图'
            '处理而不复制。'),
        'native_check': False,
        'native_note': '两输入 rank 3 vs 2 不同，冻结 ABI 的单一 INPUT_RANK '
                       '槽位无法表达；本层只做解析与形状断言',
    },
)

#: 桥与上游的硬边界。``enforced_by`` 指明约束落在哪一层。
LIMITATIONS = (
    {
        'key': 'cxx20-required',
        'statement': 'Einsums 是 C++20 张量代数库，少数组件经 CXX23 兼容模块'
                     '使用 C++23 特性（其内部提供 C++20 回退）。',
        'impact': '编译器/标准库必须支持 C++20，否则头文件无法编译。',
        'source': 'docs latest 首页：A C++20 tensor algebra library',
        'enforced_by': 'C++ 侧构建脚本（Python 侧不检查编译器）',
    },
    {
        'key': 'host-cpu-eager-api',
        'statement': '本桥只做主机（CPU）指针 + eager 调用 '
                     'einsums::compute_graph::einsum；一次调用内 Einsums 解析'
                     'spec、派发并执行，没有 ComputeGraph 捕获与跨调用优化。',
        'impact': '重复调用会重复解析与派发；想复用必须自己在 C++ 侧建图。',
        'source': 'docs latest：cg::einsum 是 eager 入口；Graph 捕获需 Graph '
                  '对象',
        'enforced_by': '桥 ABI 只有 pyqceinsums_einsum 一个数值入口',
    },
    {
        'key': 'deps-blas-lapack-hdf5-openmp',
        'statement': 'Einsums 依赖 BLAS/LAPACK、HDF5、OpenMP，另需 '
                     'fmt/spdlog/mimalloc/Stripes/Waggle/Apiary。',
        'impact': '缺任一依赖时只能构建返回 UNAVAILABLE 的桩库；'
                  '当前 libpyqceinsums.so 的可用性以 ABI 探针为准。',
        'source': 'docs latest building/index 与仓库依赖清单',
        'enforced_by': 'CMake configure + 构建脚本',
    },
    {
        'key': 'first-configure-network',
        'statement': '首次 CMake configure 可能联网抓取依赖（FetchContent）。',
        'impact': '离线机器必须先预置依赖或使用系统包，否则 configure 失败。',
        'source': 'Einsums 构建文档',
        'enforced_by': 'C++ 侧 CMake（Python 侧不做任何联网）',
    },
    {
        'key': 'python-package-opt-in-pybind11',
        'statement': '官方 Python 包需 -DEINSUMS_BUILD_PYTHON=ON 且 '
                     'pybind11>=2.11 才生成 einsums 模块。',
        'impact': '本机未安装 einsums，import einsums 失败；本包因此只走 '
                  'ctypes + C ABI，不依赖 Python 绑定。',
        'source': 'docs latest Python 绑定说明',
        'enforced_by': 'C++/打包配置；Python 侧不 import einsums',
    },
    {
        'key': 'bridge-bounded-nops-rank',
        'statement': '桥上限 nops ∈ {1,2}、rank <= 4、out_ndim <= 4；'
                     'params[1]=INPUT_RANK 只有一个槽位，两个输入的 rank 必须'
                     '相同；越界 fail-closed 返回 INVALID(1)。',
        'impact': 'GEMV 这类异 rank 组合、rank>=5 的收缩无法经桥执行；'
                  'Python 侧在进入 C ABI 前就按同一边界拒绝。',
        'source': 'cpython 侧冻结 ABI（本任务给定）',
        'enforced_by': '_bridge.check_abi_bounds + C 侧参数校验',
    },
    {
        'key': 'spec-is-data-passthrough',
        'statement': 'spec 在 Einsums 里是数据而不是类型，图优化会原地重写它；'
                     '本桥把用户 spec 原样透传，只做语法/形状校验。',
        'impact': 'Python 侧不能假设自己的字符串就是最终执行的字符串；'
                  '也不能依赖 Python 侧重写来获得性能。',
        'source': 'docs latest 首页：The spec is data rather than types',
        'enforced_by': '_bridge.einsum_into 只 normalize 不改写模式',
    },
    {
        'key': 'string-vs-template-einsum',
        'statement': 'Einsums 有两套 einsum：编译期模板版与运行时字符串版；'
                     'Python 只能走字符串版（ComputeGraph）。',
        'impact': '模板版的编译期校验与优化在 Python 路径上不存在。',
        'source': 'docs latest：string_einsum.html 与 ComputeGraph API',
        'enforced_by': '桥只导出字符串入口',
    },
    {
        'key': 'multi-char-and-permutation-untested',
        'statement': '多字符索引模式（逗号分隔标签名）与 P(...) 置换算子'
                     '在本桥未验证；P(...) 在 Python 侧被显式拒绝。',
        'impact': '耦合簇残差这类 spec 目前不能经本桥执行。',
        'source': 'string_einsum.html：Index Modes / Permutation Operators',
        'enforced_by': '_contract._reject_unsupported + 未列入 DOC_CASES',
    },
    {
        'key': 'no-unary-einsum',
        'statement': 'Einsums 的 compute_graph::einsum 只有二元重载'
                     '（另有 sizeof...(Operands)>=3 的 n 元重载），没有一元版本；'
                     '转置用 cg::permute，标量内积用 cg::dot。',
        'impact': '本桥只接受恰好两个操作数；numpy 风格的 ijkl->jlki、'
                  'ii->、ij->j 等一元写法在 Einsums 侧应改写成 permute/dot，'
                  '不能经本 ABI 执行。',
        'source': 'refer/git-rep/Einsums @ 628971c：'
                  'libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/'
                  'Operations.hpp:5711（二元）与 NaryEinsum.hpp:208'
                  '（>=3 元）；tutorial_einsum.rst 亦说明转置用 permute',
        'enforced_by': 'C 侧 num_ops!=2 校验 + _bridge.check_abi_bounds',
    },
    {
        'key': 'prefactors-alpha-beta',
        'statement': '文档的六参数形式 cg::einsum(spec, beta, &C, alpha, A, B) '
                     '计算 C = beta*C + alpha*A*B，默认 beta=0、alpha=1；'
                     '本桥 ABI 没有 alpha/beta 槽位，只实现默认的 C = A*B。',
        'impact': '需要累加或缩放的耦合簇式调用无法经本桥表达；'
                  'Python 侧也没有 c_pf/ab_pf 关键字。',
        'source': 'string_einsum.html#prefactors 与 tutorial_einsum.html'
                  '#scaling-with-prefactors',
        'enforced_by': 'SIGNATURE_VARIANTS + 冻结 ABI（params 无前因子槽位）',
    },
    {
        'key': 'multi-operand-nary',
        'statement': 'Einsums 允许任意多个操作数并按最省算术次序两两收缩；'
                     '本桥只接受 1 或 2 个操作数。',
        'impact': '如 "ij;jk;kl->il" 的三操作数调用必须由调用方拆成两次；'
                  '拆分手序影响性能但不影响数值。',
        'source': 'tutorial_einsum.html#more-than-two-operands',
        'enforced_by': 'PYQCEINSUMS_MAX_OPERANDS=2 + C 侧参数校验',
    },
    {
        'key': 'compile-time-format-validation',
        'statement': 'C++ 字面量 spec 由 EinsumFormatString 在编译期校验（缺分号'
                     '即编译错误）；运行时构造的字符串必须显式包成 '
                     'cg::EinsumFormatString(spec) 才校验。',
        'impact': '本桥经 const char* 传入的是运行时字符串，校验只能靠 Python '
                  '侧 parse/normalize 与 C 侧参数检查，不能依赖编译期保护。',
        'source': 'string_einsum.html#compile-time-validation',
        'enforced_by': '_contract.parse_einsums_spec + C 侧 fail-closed 校验',
    },
    {
        'key': 'conj-in-spec',
        'statement': '文档允许在 spec 里用 conj(...) 包裹某个操作数的下标表示复共轭'
                     '（tutorial_einsum.html "an operand is conjugated by wrapping '
                     'its indices in conj(...)"）。',
        'impact': '本层只处理纯字母标签，conj(...) 被当作文法外内容拒绝；'
                  '复数共轭需调用方在传入前自行处理。',
        'source': 'tutorial_einsum.html#more-than-two-operands',
        'enforced_by': '_contract 的标签解析（只接受字母/逗号分隔标签）',
    },
    {
        'key': 'python-boundary-copies',
        'statement': '官方 Python 绑定用 einsums.asarray 把 NumPy 数组**拷贝**成 '
                     'Einsums 张量，运算本身只接受 Einsums 张量，不直接吃 NumPy 数组。',
        'impact': '本桥的 CPU 指针 ABI 也要在 C++ 侧拷贝进出；'
                  '两边都不是零拷贝，不能宣称与 NumPy 共享内存。',
        'source': 'docs latest 首页 Capabilities and Limitations：'
                  'NumPy arrays cross the boundary through einsums.asarray, '
                  'which copies them into a tensor',
        'enforced_by': 'C++ 侧 std::copy；Python 侧只传 .ctypes.data',
    },
    {
        'key': 'gpu-mpi-disk-status',
        'statement': 'GPU 后端（CUDA/Metal）、MPI 分布式执行与磁盘张量接口的状态：'
                     'GPU 为可用特性但本桥不涉及；MPI 仍在开发；'
                     '磁盘张量接口尚未稳定。',
        'impact': '本桥只覆盖主机 CPU 的 eager einsum，不能声称支持 GPU/MPI/'
                  'out-of-core 路径。',
        'source': 'docs latest 首页 Capabilities and Limitations；'
                  'README（CUDA GPU backend and MPI are works in progress）',
        'enforced_by': 'C ABI 只有主机指针 + 单进程调用',
    },
)


def limitation_keys():
    """返回 ``LIMITATIONS`` 的 key 列表（保持登记顺序）。"""
    return [entry['key'] for entry in LIMITATIONS]


def documented_cases():
    """返回 ``source == 'documented'`` 的用例名列表。"""
    return [case['name'] for case in DOC_CASES
            if case['source'] == SOURCE_DOCUMENTED]


def inferred_cases():
    """返回 ``source == 'inferred'`` 的用例名列表。"""
    return [case['name'] for case in DOC_CASES
            if case['source'] == SOURCE_INFERRED]
