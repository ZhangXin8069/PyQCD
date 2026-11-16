"""契约测试：``verify_specialized`` 的 ``--language`` / ``--equivalence``（纯接口层，
零执行路径改动）。

守护八件事：

1. **缺省零行为变化**：不传 ``--language`` 时报告与改造前**逐字节相同**——四条
   分支（语料 HIT / 无等价 MISS / 带等价 MISS / 不可解析）的历史输出原样嵌在
   :data:`GOLDEN` 里，是改造前实跑抓下来的原文。唯一的增量是 ``--equivalence``
   引入的 ``结果调整`` 一行（见 :data:`GOLDEN` 的 ``equiv`` 条目注释）。
2. **英文侧无中文**：``--language en`` 的报告里没有汉字。语料自带的 ``case`` /
   ``subtable``（12 条中文情形名）是**语料数据**而不是本模块文案，单独按"掩掉这两个
   数据字段后无汉字"断言——把数据也翻成本地化字段等于在本模块里复制一份语料。
3. **中文侧仍是中文**：``--language zh-cn`` 与缺省输出逐字节相同且含汉字。
4. **语言不改判定**：同一方程在缺省 / en / zh-cn 下退出码相同（0/1/2 全覆盖）。
5. **语言不改 --json**：``--json`` 的文本与解析结果逐字不变。``query()`` 根本不
   接收语言参数，结构化字段在构造上就与语言无关；``--json`` 仍是 ``query()`` 的
   原样序列化。
6. **非法语言退出码 2**：``--language fr`` 由 argparse 的 ``choices`` 拦下。
   ``zh`` 是 :func:`resolve_language` 的 API 别名，与 ``test.py`` 一致不进 choices。
7. **等价映射段双语**：en 下输入侧（``张量调整``→``operands``）与输出侧
   （``结果调整``→``output``）都给，且说明文字不夹中文。
8. **``--equivalence`` 三态**：``false`` 时等价段完全不出现且 MISS 原因**逐字**回到
   引入等价映射之前的那句；``true`` 时输入侧与输出侧**两行都在**（某一侧无需调整
   也写 ``-``，不省略整行）；``auto`` 跟随 ``PYQCD_EINSUM_EQUIVALENCE``（未设即
   开启）而显式 ``true``/``false`` 压过它。``--json`` 的 ``equivalence`` 字段开启
   时非空、关闭时缺该字段（与 ``test.py`` 同口径）；开关**不影响退出码**。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re

from ..einsum import verify_specialized as module

#: 语料 HIT（seq=1650，13 操作数，输出秩 2）。
EQUATION_HIT = "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
#: 形状特化 HIT（非语料，`_shape_paths`）——这一支没有 `corpus` 段。
EQUATION_SHAPE = "x,ax,bx,cx->abc"
#: 语料外的方程（既无语料命中，也无等价写法）。
EQUATION_MISS = "ij,jk->ik"
#: 语料外但存在等价写法（输入侧轴序）。
EQUATION_EQUIV = "acdb,menf,ac,em,bd,fn->"
#: 语料外但存在等价写法（输出侧轴序）。
EQUATION_EQUIV_OUTPUT = "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->sa"
#: 语料外但存在等价写法（输入侧与输出侧**同时**调整，`operand-axes+output-axes`）。
EQUATION_EQUIV_BOTH = "gabh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->sa"
#: 无法解析（输出指标重复）。
EQUATION_FAIL = "ij->ii"

#: 引入等价映射**之前**的 MISS 原因原文
#: （``git show fd26357e:pyqcd/einsum/verify_specialized.py``）。``--equivalence
#: false`` 必须逐字回到这一句——那是"关闭即无此功能"的证据。
LEGACY_MISS_REASON = ("既不在 wick_12cases 语料里，也没有形状特化路径，"
                      "走通用分步路径（无特化内核）")

#: 等价段里会出现的人读标签。``--equivalence false`` 时一个都不许出现。
EQUIVALENCE_LABELS = ("等价写法", "变换类型", "变换说明", "张量调整",
                      "结果调整", "提示")

ZH = re.compile(r"[\u4e00-\u9fff]")

#: 改造前实跑抓取的原文（缺省不带 `--language`）。
#:
#: ``equiv`` 条目比改造前多一行 ``结果调整``：2026-11-16 起等价段**必须**同时给出
#: 输入侧与输出侧（只报输入侧会让人算出轴序错了的结果），无回填时显式写
#: ``-（改写后输出轴序已一致）``。其余每一行仍与改造前逐字节相同。
GOLDEN = {
    "hit": '方程        : agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as\n归一化      : agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as\n拓扑签名    : ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19), (4, 8), (5, 12), (1, 16), (17, 9), (2, 6, 10), (7, 14), (3, 18), (19, 11, 15), (0, 13))\n元数/槽位   : 13 个操作数 / 40 个指标槽（输出秩 2）\n摘要        : 8006c502f8072c0e0a6db314\n特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）\n结果        : HIT\n  特化路径  : topology:8006c502  [kernel(topology)]\n  声明内核  : kupl_topology, cuda_topology\n  语料出处  : seq=1650  T00028  10. 重子四点函数\n  代表写法  : agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as\n  操作数角色: peram_u, peram_u, peram_u, peram_u, peram_u, C * gamma_B, gamma_J1, gamma_J2, C * gamma_BH, VVV_0, VDV_1, VDV_2, VVV_3\n  说明      : 声明内核只表示契约允许；实际是否下发取决于 out=、dtype、原生库可用性。\n              数值验证见 python -m pyqcd.testing.einsum.verify_specialized\n',
    "miss": '方程        : ij,jk->ik\n归一化      : ij,jk->ik\n拓扑签名    : ((0, 1), (1, 2), (0, 2))\n元数/槽位   : 2 个操作数 / 6 个指标槽（输出秩 2）\n摘要        : 77309c2e72f7ec9e54e4f294\n特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）\n结果        : MISS\n  原因      : 既不在 wick_12cases 语料里，也没有形状特化路径，且不存在可落到已知形式的等价写法，走通用分步路径（无特化内核）\n',
    "equiv": '方程        : acdb,menf,ac,em,bd,fn->\n归一化      : acdb,menf,ac,em,bd,fn->\n拓扑签名    : ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1), (5, 4), (3, 2), (7, 6), ())\n元数/槽位   : 6 个操作数 / 16 个指标槽（输出秩 0）\n摘要        : 91104304ce1d8e05ab572591\n特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）\n结果        : MISS\n  原因      : 直接写法不在语料里，但存在等价写法可命中特化路径（操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)））；需按给出的张量调整同步转换后再算，结果与直接写逐元素相同\n  等价写法  : cadb,menf,ac,em,bd,fn->\n  变换类型  : operand-axes\n  变换说明  : 操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）\n  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -\n  结果调整  : -（改写后输出轴序已一致）\n  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，\n              用等价方程调用 contract() 即可走上特化路径，结果与直接写法逐元素相同。\n',
    "fail": '方程        : ij->ii\n归一化      : ij->ii\n结果        : FAIL\n  原因      : 方程无法解析（非法字符、空操作数、多重 -> 或输出指标不合法）\n',
}

_GOLDEN_EQUATION = {
    "hit": EQUATION_HIT,
    "miss": EQUATION_MISS,
    "equiv": EQUATION_EQUIV,
    "fail": EQUATION_FAIL,
}


@contextlib.contextmanager
def _env(**values):
    """临时设定环境变量（``None`` = 删掉），退出时原样恢复。"""
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _clean():
    """本模块的判定与文案只应取决于这三个环境变量，其余一律清干净。

    ``PYQCD_EINSUM_EQUIVALENCE="1"`` 是刻意的：缺省 golden 是**开启**等价映射时的
    输出，宿主机环境里若留着 ``=0`` 会让整组用例静默换口径。关闭态的用例自己用
    :func:`_env` 显式设定。
    """
    with _env(PYQCD_EINSUM_LANG=None, PYQCD_EINSUM_SPECIALIZE="1",
              PYQCD_EINSUM_EQUIVALENCE="1"):
        yield


def _capture(argv):
    """进程内跑 ``main()``，返回 ``(退出码, stdout, stderr)``。"""
    out, err = io.StringIO(), io.StringIO()
    code = 1
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = module.main(list(argv))
        except SystemExit as exc:            # argparse 的 --help 与 choices 错误
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- 1. 零行为变化

def test_default_language_reproduces_the_legacy_report_byte_for_byte():
    """不传 --language 时四条分支的输出必须与改造前逐字节相同。"""
    expected_codes = {"hit": 0, "miss": 1, "equiv": 1, "fail": 2}
    with _clean():
        for name, equation in _GOLDEN_EQUATION.items():
            code, text, _ = _capture([equation])
            assert code == expected_codes[name], (name, code)
            assert text == GOLDEN[name], (
                f"{name} 分支的输出与改造前不一致：\n--- got ---\n{text}"
                f"--- expected ---\n{GOLDEN[name]}")


def test_chinese_report_stays_chinese_and_equals_the_default():
    """--language zh-cn 与缺省输出逐字节相同，且确实含中文。"""
    with _clean():
        for equation in (EQUATION_HIT, EQUATION_SHAPE, EQUATION_MISS,
                         EQUATION_EQUIV, EQUATION_FAIL):
            default = _capture([equation])
            chosen = _capture([equation, "--language", "zh-cn"])
            assert chosen == default, (equation, chosen, default)
            assert ZH.search(chosen[1]), (equation, chosen[1])


# ---------------------------------------------------------------- 2/3. 语言本身

def test_english_report_has_no_chinese_text():
    """en 的报告里不得出现汉字（语料数据字段除外，单独验）。"""
    with _clean():
        for equation in (EQUATION_SHAPE, EQUATION_MISS, EQUATION_EQUIV,
                         EQUATION_EQUIV_OUTPUT, EQUATION_FAIL):
            code, text, _ = _capture([equation, "--language", "en"])
            assert not ZH.search(text), (
                f"{equation} 的英文报告里仍有中文：\n{text}")

        # 语料 HIT：corpus.case / corpus.subtable 是语料自带的中文情形名（12 条），
        # 属于数据而非本模块文案；把这两段掩掉后仍不许有汉字。
        code, text, _ = _capture([EQUATION_HIT, "--language", "en"])
        assert code == 0
        corpus = module.query(EQUATION_HIT)["corpus"]
        masked = text
        for token in (corpus["case"], corpus["subtable"]):
            if token:
                masked = masked.replace(token, "<corpus-data>")
        assert not ZH.search(masked), f"除语料数据外仍有中文：\n{masked}"
        assert ZH.search(text), "语料自带的情形名应原样保留"


def test_help_follows_the_language_option():
    """--help 也双语；缺省（中文）与 --language en 各自成体系。"""
    with _clean():
        assert ZH.search(module.build_parser().format_help())
        english = module.build_parser("en").format_help()
        assert not ZH.search(english), english
        for token in ("--language {auto,en,zh-cn}", "Exit codes", "equation"):
            assert token in english, (token, english)
        assert "--language {auto,en,zh-cn}" in module.build_parser().format_help()

        code, text, _ = _capture(["--help", "--language", "en"])
        assert code == 0 and not ZH.search(text), text


def test_language_never_changes_the_exit_code():
    """同一方程在缺省 / en / zh-cn 下的退出码必须相同。"""
    expected = ((EQUATION_HIT, 0), (EQUATION_SHAPE, 0), (EQUATION_MISS, 1),
                (EQUATION_EQUIV, 1), (EQUATION_FAIL, 2))
    with _clean():
        for equation, code_expected in expected:
            codes = {_capture([equation])[0],
                     _capture([equation, "--language", "en"])[0],
                     _capture([equation, "--language", "zh-cn"])[0]}
            assert codes == {code_expected}, (equation, codes)


def test_language_never_changes_the_json_payload():
    """--json 是 query() 的原样序列化：加语言支持后文本与字段都不许变。"""
    with _clean():
        for equation in (EQUATION_HIT, EQUATION_MISS, EQUATION_EQUIV,
                         EQUATION_FAIL):
            base = _capture([equation, "--json"])
            for language in ("en", "zh-cn"):
                other = _capture([equation, "--json", "--language", language])
                assert other == base, (equation, language, other, base)
            payload = json.loads(base[1])
            assert payload == json.loads(json.dumps(module.query(equation),
                                                   ensure_ascii=False))


def test_invalid_language_is_a_usage_error():
    """非法语言走 argparse 的 choices（退出码 2），与已有 --json 风格一致。"""
    with _clean():
        code, _, err = _capture([EQUATION_MISS, "--language", "fr"])
        assert code == 2, code
        assert "invalid choice" in err and "zh-cn" in err, err
        assert not ZH.search(err), f"参数错误文案不该是中文：{err}"

        # `zh` 只在 API 层是别名；CLI 的 choices 与 test.py 一致不含它。
        assert _capture([EQUATION_MISS, "--language", "zh"])[0] == 2
        assert module.resolve_language("zh") == "zh"
        assert module.LANGUAGE_MODES == ("auto", "en", "zh-cn")


def test_auto_only_follows_the_explicit_env_variable():
    """auto 不跟随 locale：未设 PYQCD_EINSUM_LANG 时保持中文。"""
    with _env(PYQCD_EINSUM_SPECIALIZE="1", PYQCD_EINSUM_LANG=None,
              LC_ALL=None, LANG="C.UTF-8"):
        assert module.resolve_language("auto") == "zh"
        assert module.resolve_language(None) == "zh"
        assert ZH.search(_capture([EQUATION_MISS])[1])

    with _env(PYQCD_EINSUM_SPECIALIZE="1", PYQCD_EINSUM_LANG="en_US.UTF-8"):
        assert module.resolve_language("auto") == "en"
        assert not ZH.search(_capture([EQUATION_MISS])[1])

    # 显式 --language 覆盖环境变量。
    with _env(PYQCD_EINSUM_SPECIALIZE="1", PYQCD_EINSUM_LANG="en"):
        assert ZH.search(_capture([EQUATION_MISS, "--language", "zh-cn"])[1])


# ---------------------------------------------------------------- 4. 等价映射段

def test_english_equivalence_section_shows_both_input_and_output_side():
    """en 下输入侧（张量调整）与输出侧（结果调整）都要给，且全都是英文。"""
    with _clean():
        # 输入侧轴序：operand_permutations
        code, text, _ = _capture([EQUATION_EQUIV, "--language", "en"])
        assert code == 1
        assert "  operands  : transpose(1, 0, 2, 3), -, -, -, -, -" in text, text
        assert "  rewrite   : cadb,menf,ac,em,bd,fn->" in text, text
        assert not ZH.search(text), text

        # 输出侧轴序：output_permutation
        code, text, _ = _capture([EQUATION_EQUIV_OUTPUT, "--language", "en"])
        assert code == 1
        assert "  output    : transpose(1, 0) the result afterwards" in text, text
        assert not ZH.search(text), text

        # 两侧同时存在的情形（真实方程 `EQUATION_EQUIV_BOTH` 已由下面的 --equivalence
        # 用例覆盖；这里再用合成报告钉住打印逻辑：en 的说明文字由结构化字段自己
        # 渲染，不夹中文）。`rewritten` 必须为真——报告只在确有改写时出这一段。
        report = dict(module.query(EQUATION_MISS))
        report["equivalence"] = {
            "equation": "cabd,menf,ac,em,bd,fn->",
            "kind": "combined",
            "rewritten": True,
            "description": "合成用例：输入侧与输出侧同时调整",
            "operand_order": None,
            "operand_permutations": [[1, 0, 2, 3], None, None, None, None, None],
            "output_permutation": [1, 0],
        }
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module._print_report(report, "en")
        combined = buffer.getvalue()
        assert "  operands  : transpose(1, 0, 2, 3), -, -, -, -, -" in combined, combined
        assert "  output    : transpose(1, 0) the result afterwards" in combined, combined
        assert not ZH.search(combined), combined


# ---------------------------------------------------------------- 5. 机器字段

def test_reason_stays_machine_stable_and_matches_its_zh_render():
    """reason 是历史中文文案，恒等于 reason_key 的 zh 渲染；不新增本地化字段。"""
    with _clean():
        expected = ((EQUATION_MISS, "reason.no_specialized_path"),
                    (EQUATION_FAIL, "reason.unparsable"),
                    (EQUATION_EQUIV, "reason.equivalent_rewrite"))
        for equation, key in expected:
            payload = module.query(equation)
            assert payload["reason_key"] == key, (equation, payload)
            assert payload["reason"] == module.translate(
                key, "zh", **payload["reason_args"]), payload
            assert "reason_localized" not in payload, payload
        assert module.query(123)["reason_key"] == "reason.not_a_string"
        assert module.query(123).get("matched") is False


def test_every_message_has_both_variants_and_missing_keys_raise():
    """每个 key 都要 en/zh；缺 key / 缺变体必须抛异常，不静默回退。"""
    for key, variants in module.MESSAGES.items():
        assert set(variants) == {"en", "zh"}, key
        assert variants["en"].strip() and variants["zh"].strip(), key
        assert variants["en"] != variants["zh"], f"{key} 的两种语言文案相同"

    try:
        module.translate("no.such.key", "en")
    except KeyError:
        pass
    else:
        raise AssertionError("缺 key 时必须抛 KeyError")

    module.MESSAGES.setdefault("probe.only_zh", {"zh": "只有中文"})
    try:
        module.translate("probe.only_zh", "en")
    except KeyError:
        pass
    else:
        raise AssertionError("缺语言变体时必须抛 KeyError")
    finally:
        module.MESSAGES.pop("probe.only_zh", None)


# ------------------------------------------------------- 6. --equivalence 三态

def _reason_line(text):
    """抽出 ``  原因      :`` 那一行的值（逐字比对用）。"""
    prefix = "  原因      : "
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, f"原因行不唯一：\n{text}"
    return lines[0][len(prefix):]


def test_equivalence_false_hides_the_section_and_restores_the_legacy_wording():
    """``--equivalence false``：等价段完全消失，MISS 原因逐字回到旧版那句。"""
    with _clean():
        for equation in (EQUATION_MISS, EQUATION_EQUIV, EQUATION_EQUIV_BOTH):
            code, text, _ = _capture([equation, "--equivalence", "false"])
            assert code == 1, (equation, code)
            assert _reason_line(text) == LEGACY_MISS_REASON, (
                f"{equation} 关闭后不是旧版措辞：\n{text}")
            for label in EQUIVALENCE_LABELS:
                assert label not in text, (equation, label, text)
            # 这一支连"等价"二字都不该出现（与引入本功能之前逐字节同级）
            assert "等价" not in text, (equation, text)

        # --json 侧：关闭时没有 equivalence 字段（与 test.py 同口径）
        payload = json.loads(_capture(
            [EQUATION_EQUIV, "--json", "--equivalence", "false"])[1])
        assert "equivalence" not in payload, payload
        assert payload["reason"] == LEGACY_MISS_REASON, payload
        assert payload["reason_key"] == "reason.no_specialized_path_disabled"

        # 环境变量关闭等价于 --equivalence false
        with _env(PYQCD_EINSUM_LANG=None, PYQCD_EINSUM_SPECIALIZE="1",
                  PYQCD_EINSUM_EQUIVALENCE="0"):
            _, text, _ = _capture([EQUATION_EQUIV])
            assert _reason_line(text) == LEGACY_MISS_REASON, text
            assert "等价写法" not in text, text


def test_equivalence_true_shows_both_input_and_output_side():
    """``--equivalence true``：输入侧与输出侧两行都必须在场。"""
    with _clean():
        # 两侧同时调整：operand-axes + output-axes（13 个操作数，只有第 0 路要转）
        code, text, _ = _capture([EQUATION_EQUIV_BOTH, "--equivalence", "true"])
        assert code == 1
        assert ("  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -, -, -, -, -, "
                "-, -, -") in text, text
        assert "  结果调整  : 拿到结果后需 transpose(1, 0)" in text, text
        payload = module.query(EQUATION_EQUIV_BOTH, equivalence="true")
        assert payload["equivalence"]["kind"] == "operand-axes+output-axes", payload
        assert payload["equivalence"]["operand_permutations"][0] == [1, 0, 2, 3]
        assert payload["equivalence"]["output_permutation"] == [1, 0]

        # 只需输出侧：输入侧显式写 `-`，不省略整行
        _, text, _ = _capture([EQUATION_EQUIV_OUTPUT, "--equivalence", "true"])
        assert "  张量调整  : -" in text, text
        assert "  结果调整  : 拿到结果后需 transpose(1, 0)" in text, text

        # 只需输入侧：输出侧显式写"已一致"，不省略整行
        _, text, _ = _capture([EQUATION_EQUIV, "--equivalence", "true"])
        assert "  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -" in text, text
        assert "  结果调整  : -（改写后输出轴序已一致）" in text, text

        # 英文侧同样两行都在
        _, text, _ = _capture([EQUATION_EQUIV_BOTH, "--equivalence", "true",
                               "--language", "en"])
        assert ("  operands  : transpose(1, 0, 2, 3), -, -, -, -, -, -, -, -, -, "
                "-" in text), text
        assert "  output    : transpose(1, 0) the result afterwards" in text, text
        assert not ZH.search(text), text


def test_equivalence_auto_follows_the_env_and_explicit_beats_it():
    """``auto`` 跟随 ``PYQCD_EINSUM_EQUIVALENCE``；显式 true/false 压过它。"""
    with _env(PYQCD_EINSUM_LANG=None, PYQCD_EINSUM_SPECIALIZE="1",
              PYQCD_EINSUM_EQUIVALENCE="0"):
        assert module.resolve_equivalence("auto") is False
        assert module.resolve_equivalence(None) is False
        assert "等价写法" not in _capture([EQUATION_EQUIV])[1]
        assert "等价写法" not in _capture([EQUATION_EQUIV, "--equivalence", "auto"])[1]
        forced = _capture([EQUATION_EQUIV, "--equivalence", "true"])[1]
        assert "等价写法" in forced and "结果调整" in forced, forced
        # 强制开启是**临时**压过环境变量：调用结束后必须原样还回去，
        # 否则同进程里后续查询与 test.py 的判定会被悄悄改写。
        assert os.environ.get(module.EQUIVALENCE_ENV) == "0", (
            f"{module.EQUIVALENCE_ENV} 被泄漏成 "
            f"{os.environ.get(module.EQUIVALENCE_ENV)!r}")

    with _env(PYQCD_EINSUM_LANG=None, PYQCD_EINSUM_SPECIALIZE="1",
              PYQCD_EINSUM_EQUIVALENCE="1"):
        assert module.resolve_equivalence("auto") is True
        assert "等价写法" in _capture([EQUATION_EQUIV])[1]
        assert "等价写法" not in _capture([EQUATION_EQUIV, "--equivalence", "false"])[1]

    # 未设 = 开启（缺省行为）
    with _env(PYQCD_EINSUM_LANG=None, PYQCD_EINSUM_SPECIALIZE="1",
              PYQCD_EINSUM_EQUIVALENCE=None):
        assert module.resolve_equivalence("auto") is True
        assert "等价写法" in _capture([EQUATION_EQUIV])[1]


def test_equivalence_json_field_is_present_only_when_enabled():
    """``--json`` 的 ``equivalence`` 字段：开启时非空，关闭时缺该字段。"""
    with _clean():
        on = json.loads(_capture([EQUATION_EQUIV, "--json"])[1])
        assert on["equivalence"]["rewritten"] is True, on
        assert on["equivalence"]["equation"] == "cadb,menf,ac,em,bd,fn->", on
        assert on["equivalence"]["operand_permutations"][0] == [1, 0, 2, 3], on

        # 开启但查过没有等价写法：字段仍在（用来区分"查过"与"根本没查"）
        miss = json.loads(_capture([EQUATION_MISS, "--json"])[1])
        assert miss["equivalence"]["rewritten"] is False, miss

        off = json.loads(_capture([EQUATION_MISS, "--json",
                                   "--equivalence", "false"])[1])
        assert "equivalence" not in off, off

        # 结构化字段不受 --language 影响
        for language in ("en", "zh-cn"):
            assert (_capture([EQUATION_EQUIV, "--json", "--language", language])
                    == _capture([EQUATION_EQUIV, "--json"])), language

        # 命中特化的方程不查等价写法，也就没有该字段
        hit = json.loads(_capture([EQUATION_HIT, "--json"])[1])
        assert "equivalence" not in hit, sorted(hit)


def test_equivalence_never_changes_the_exit_code():
    """开关只影响附加信息：同一方程的退出码在 auto/true/false 下必须相同。"""
    expected = ((EQUATION_HIT, 0), (EQUATION_SHAPE, 0), (EQUATION_MISS, 1),
                (EQUATION_EQUIV, 1), (EQUATION_EQUIV_BOTH, 1), (EQUATION_FAIL, 2))
    with _clean():
        for equation, code_expected in expected:
            codes = {_capture([equation, "--equivalence", mode])[0]
                     for mode in ("auto", "true", "false")}
            assert codes == {code_expected}, (equation, codes)


def test_equivalence_option_is_documented_and_matches_the_shared_switch():
    """``--help`` 双语给出该选项；常量与 ``_equivalence`` 的开关同名同值。"""
    with _clean():
        for text in (module.build_parser().format_help(),
                     module.build_parser("en").format_help()):
            assert "--equivalence {auto,true,false}" in text, text
            assert module.EQUIVALENCE_ENV in text, text
        assert not ZH.search(module.build_parser("en").format_help())

        assert module.EQUIVALENCE_MODES == ("auto", "true", "false")
        assert module.DEFAULT_EQUIVALENCE == "auto"
        assert module.EQUIVALENCE_ENV == "PYQCD_EINSUM_EQUIVALENCE"

        from ..einsum import _equivalence

        assert module.EQUIVALENCE_ENV == getattr(
            _equivalence, "EQUIVALENCE_ENV", module.EQUIVALENCE_ENV), (
                "环境变量名必须与 _equivalence 的开关一致")

        # 非法三态在 API 层抛 ValueError；在 CLI 层由 choices 拦成退出码 2
        try:
            module.resolve_equivalence("maybe")
        except ValueError:
            pass
        else:
            raise AssertionError("非法三态必须抛 ValueError")
        code, _, err = _capture([EQUATION_MISS, "--equivalence", "maybe"])
        assert code == 2, code
        assert "invalid choice" in err, err


def test_auto_still_works_when_the_switch_function_is_missing():
    """旧版 ``_equivalence`` 没有 ``enabled()`` 时降级为"开启"（改造前行为）。

    降级方向是**故意**的：只会"关不掉"，不会把已有功能静默关死。
    """
    from ..einsum import _equivalence

    saved = getattr(_equivalence, "enabled", None)
    if saved is None:                     # 已经是旧版形态，直接验降级口径
        assert module.resolve_equivalence("auto") is True
        return
    del _equivalence.enabled
    try:
        with _env(PYQCD_EINSUM_EQUIVALENCE="0"):
            assert module.resolve_equivalence("auto") is True, (
                "没有 enabled() 时不应凭环境变量把等价段静默关掉")
    finally:
        _equivalence.enabled = saved


if __name__ == "__main__":
    from .conftest import main

    raise SystemExit(main())
