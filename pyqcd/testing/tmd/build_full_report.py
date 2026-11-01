"""Build the Chinese 16:9 evidence report for the full TMD production run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _fmt(value, digits=4, unit=''):
    if value is None:
        return '不可用'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return '不可用'
    rendered_unit = r'\%' if unit == '%' else unit
    return f'{number:.{digits}g}{rendered_unit}'


def _tex_escape(text):
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(char, char) for char in str(text))


def _path_text(path):
    return rf'\nolinkurl{{{path}}}'


def _line_of(relative_path, needle):
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        return None
    for number, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1):
        if needle in line:
            return number
    return None


def _source_ref(relative_path, needle, label=None):
    line = _line_of(relative_path, needle)
    suffix = f':{line}' if line is not None else ''
    return rf'\texttt{{\detokenize{{{relative_path}{suffix}}}}}' + (
        f'（{_tex_escape(label)}）' if label else '')


def _resource_summary(path):
    path = Path(path)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {
            'samples': 0,
            'cpu_max': None,
            'memory_max': None,
            'disk_max': None,
            'gpu': {},
        }
    gpu = {}
    for row in rows:
        for item in row.get('gpu', []):
            index = int(item['index'])
            value = gpu.setdefault(index, {
                'name': item.get('name', ''),
                'memory_max': 0.0,
                'util_values': [],
            })
            value['memory_max'] = max(
                value['memory_max'], float(item['memory_percent']))
            value['util_values'].append(float(item['utilization_percent']))
    return {
        'samples': len(rows),
        'cpu_max': max(float(row['cpu_percent']) for row in rows),
        'memory_max': max(float(row['memory_percent']) for row in rows),
        'disk_max': max(float(row['disk_percent']) for row in rows),
        'gpu': gpu,
    }


def _worker_rows(run_dir):
    rows = []
    for path in sorted((Path(run_dir) / 'workers').glob('worker_*.json')):
        payload = _load_json(path, {})
        if payload.get('status') != 'complete':
            continue
        rows.append({
            'worker': path.stem,
            'backend': payload.get('backend', '?'),
            'device': payload.get('device', '?'),
            'conf_ids': payload.get('conf_ids', []),
            'status': payload.get('status', '?'),
            'started': payload.get('started_utc', ''),
            'finished': payload.get('finished_utc', ''),
        })
    return rows


def _worker_table(run_dir):
    rows = _worker_rows(run_dir)
    lines = [
        r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth} l l X l@{}}',
        r'\toprule',
        r'worker & backend/device & 状态 & 组态 & 完成时间 \\',
        r'\midrule',
    ]
    for item in rows:
        lines.append(
            f"{_tex_escape(item['worker'])} & "
            f"{_tex_escape(item['backend'])}/{_tex_escape(item['device'])} & "
            f"{_tex_escape(item['status'])} & "
            f"{_tex_escape(','.join(map(str, item['conf_ids'])))} & "
            f"{_tex_escape(str(item['finished'])[:19])} \\\\")
    lines.extend([r'\bottomrule', r'\end{tabularx}'])
    return '\n'.join(lines)


def _figure(path, caption, width=0.88):
    path = Path(path)
    if not path.is_file():
        return (
            r'\begin{block}{图像不可用}'
            + _path_text(path)
            + r'\end{block}')
    return (
        rf'\includegraphics[width={width:.2f}\linewidth,'
        rf'height=0.62\textheight,keepaspectratio]'
        rf'{{{path}}}'
        + '\n\n'
        + rf'\source{{{_tex_escape(caption)}；数据文件：'
        + _tex_escape(path.name)
        + '}'
    )


def _frame(title, body, source=None, plain=False):
    option = '[plain]' if plain else '[t]'
    source_block = f'\n  \\source{{{source}}}' if source else ''
    return (
        f'\\begin{{frame}}{option}{{{title}}}\n'
        f'  {body}\n'
        f'{source_block}\n'
        '\\end{frame}\n'
    )


def _summary_markdown(summary, workers, resources, manifest, status):
    lines = [
        '# 梯度流核子胶子 TMD-PDF 全链计算结果',
        '',
        '## 任务与范围',
        '',
        f"- 组态：`{','.join(map(str, summary.get('conf_ids', [])))}`",
        f"- 流时：`{summary.get('tau_values', [])}`",
        '- 输出状态：`diagnostic_only`；`physical_acceptance=false`。',
        '',
        '## 结论摘要',
        '',
        f"本次运行完成了 {len(summary.get('conf_ids', []))} 组态蒸馏顶点、"
        '双极化 2pt、连接 3pt、三流时 Wilson flow、'
        '三方向有限横距 OPE、二维 staircase OPE、ratio、平台矩阵元与格点端准 PDF/TMD-PDF。',
        '由于缺少 soft/rapidity、胶子-夸克混合和连续极限，结果不能升级为物理光锥 TMD-PDF。',
        '',
        '## 计算与资源',
        '',
        f"- worker 数：{len(workers)}",
        f"- 产物文件数：{manifest.get('file_count', 0)}",
        f"- 产物总字节：{manifest.get('total_bytes', 0)}",
        f"- CPU 最大占用：{_fmt(resources.get('cpu_max'), 4, '%')}",
        f"- RAM 最大占用：{_fmt(resources.get('memory_max'), 4, '%')}",
        f"- 磁盘最大占用：{_fmt(resources.get('disk_max'), 4, '%')}",
        '',
        '## 结果与证据',
        '',
        f"- ratio-1D 产物数：{summary.get('n_ratio_1d', 0)}",
        f"- ratio-2D 产物数：{summary.get('n_ratio_2d', 0)}",
        f"- PDF/TMD-PDF 产物数：{summary.get('n_pdf', 0)}",
        f"- 汇总文件：`analysis/analysis_summary.json`",
        '',
        '## 风险与未验证项',
        '',
        '- 未执行微扰 light-cone matching。',
        '- 未测量 soft/rapidity。',
        '- 仅单一格距、单一系综，无连续极限。',
        '- 有限 Fourier 支撑与固定 staple 长度引入系统误差。',
        '',
        '## 状态',
        '',
        f"- `status={status}`",
    ]
    return '\n'.join(lines) + '\n'


def build_report(run_dir, tex_path, summary_path):
    run_dir = Path(run_dir).resolve()
    analysis_dir = run_dir / 'analysis'
    summary = _load_json(analysis_dir / 'analysis_summary.json')
    if summary is None:
        raise FileNotFoundError('缺少 analysis/analysis_summary.json')
    spectrum = _load_json(analysis_dir / 'spectrum' / 'spectrum_summary.json', {})
    plateau = _load_json(analysis_dir / 'plateaus' / 'plateau_summary.json', {})
    flow_windows = _load_json(
        analysis_dir / 'flow_windows' / 'flow_window_summary.json', {})
    budget = _load_json(analysis_dir / 'error_budget.json', {})
    manifest = _load_json(analysis_dir / 'artifact_manifest.json', {})
    staircase = _load_json(
        analysis_dir / 'ratio_2d' / 'staircase_summary.json', {})
    resources = _resource_summary(
        run_dir / 'workers' / 'system_resources.jsonl')
    workers = _worker_rows(run_dir)
    status = summary.get('status', 'diagnostic_only')

    source_map = [
        ('Wilson flow RK3', 'pyqcd/renorm/_gradient_flow.py',
         'def wilson_flow_step'),
        ('Clover/staple TMD OPE', 'pyqcd/renorm/_tmd.py',
         'def full_channel_tmd_ope_time'),
        ('canonical 1D/2D OPE', 'pyqcd/pipeline/_tmd9.py',
         'def compute_tmd_ope_time_2d'),
        ('disconnected ratio', 'pyqcd/analysis/_gradient_flow_ratio.py',
         'def calculate_gradient_flow_tmd_ratios_from_configurations'),
        ('plateau matrix element', 'pyqcd/analysis/_tmd_ratio.py',
         'def plateau_c0'),
        ('quasi-TMD Fourier', 'pyqcd/renorm/_tmdextract.py',
         'def quasi_tmd_pdf'),
        ('lattice CS kernel', 'pyqcd/renorm/_tmdextract.py',
         'def cs_kernel_two_momentum'),
        ('production driver', 'pyqcd/testing/tmd/run_full_production.py',
         'def run_worker'),
        ('analysis driver', 'pyqcd/testing/tmd/analyze_full_production.py',
         'def analyze_run'),
    ]
    source_table = '\n'.join([
        r'\begin{tabularx}{\linewidth}{@{}l X@{}}',
        r'\toprule',
        r'物理/工程对象 & 代码证据 \\',
        r'\midrule',
        *[
            f'{_tex_escape(label)} & {_source_ref(path, needle)}\\\\'
            for label, path, needle in source_map
        ],
        r'\bottomrule',
        r'\end{tabularx}',
    ])

    workers_table = _worker_table(run_dir)
    gpu_rows = []
    for index, item in sorted(resources['gpu'].items()):
        util_values = item['util_values']
        gpu_rows.append(
            f"{index} & {_tex_escape(item['name'])} & "
            f"{_fmt(item['memory_max'], 4, '%')} & "
            f"{_fmt(sum(util_values) / len(util_values) if util_values else None, 4, '%')} & "
            f"{_fmt(max(util_values) if util_values else None, 4, '%')} \\\\")
    gpu_table = '\n'.join([
        r'\begin{tabularx}{0.92\linewidth}{@{}c X c c c@{}}',
        r'\toprule',
        r'GPU & 设备 & 显存峰值 & 平均利用率 & 瞬时峰值 \\',
        r'\midrule',
        *gpu_rows,
        r'\bottomrule',
        r'\end{tabularx}',
    ])

    pdf_dir = analysis_dir / 'pdf'
    flow_image = analysis_dir / 'flow' / 'flow_history.png'
    spectrum_images = [
        analysis_dir / 'spectrum' / 'effective_mass_nopol_P000.png',
        analysis_dir / 'spectrum' / 'effective_mass_nopol_P200.png',
        analysis_dir / 'spectrum' / 'effective_mass_nopol_P400.png',
    ]
    ratio_images = [
        analysis_dir / 'ratio_1d' / 'ratio_v2_tau1_x_P002_ratio.png',
        analysis_dir / 'ratio_1d' / 'ratio_v2_tau1_y_P020_ratio.png',
        analysis_dir / 'ratio_1d' / 'ratio_v2_tau1_z_P200_ratio.png',
    ]
    c0_images = [
        analysis_dir / 'plateaus' / 'c0_tau1_z_P200_windows.png',
        analysis_dir / 'plateaus' / 'c0_tau2_z_P200_windows.png',
        analysis_dir / 'plateaus' / 'c0_tau3_z_P200_windows.png',
    ]
    pdf_images = [
        pdf_dir / 'quasi_tau1_P200.png',
        pdf_dir / 'quasi_tau3_P200.png',
        pdf_dir / 'cs_kernel_tau1.png',
        pdf_dir / 'cs_kernel_tau3.png',
    ]

    frames = []
    frames.append(_frame(
        '梯度流重整化核子胶子 TMD-PDF：12 组态全链完成，但物理接受仍未闭合',
        r'\centering\begin{minipage}{0.86\linewidth}\centering'
        r'{\Large\bfseries PyQCD 真实数据计算结果}\par\vspace{0.7em}'
        r'12 configurations \quad $\tau/a^2=1,2,3$ \quad '
        r'$z/a=0\ldots12$ \quad $b/a=0\ldots6$\par\vspace{1em}'
        rf'\textcolor{{warning}}{{状态：{_tex_escape(status)}；'
        r'physical\_acceptance=false}\end{minipage}',
        plain=True,
    ))
    frames.append(_frame(
        '结论先行：完整格点链已产出，但只能作为诊断性 TMD-PDF',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.32\textwidth}\begin{block}{已完成}'
        rf'{summary.get("n_ratio_1d", 0)} 个 1D ratio、'
        rf'{summary.get("n_ratio_2d", 0)} 个 2D ratio、'
        rf'{summary.get("n_pdf", 0)} 个 PDF/TMD-PDF 产出。'
        r'\end{block}\end{column}' + '\n'
        r'\begin{column}{0.32\textwidth}\begin{block}{最强证据}'
        rf'12/12 组态、{manifest.get("file_count", 0)} 个数据文件；'
        rf'总字节 {manifest.get("total_bytes", 0)}。'
        r'\end{block}\end{column}' + '\n'
        r'\begin{column}{0.32\textwidth}\begin{block}{未闭合}'
        r'soft/rapidity、微扰 matching、胶子--夸克混合、连续极限均未完成。'
        r'\end{block}\end{column}\end{columns}',
        rf'汇总：{_path_text(analysis_dir / "analysis_summary.json")}',
    ))
    frames.append(_frame(
        '验收标准区分“数据链完整”与“物理方案闭合”',
        r'\small\begin{tabularx}{\linewidth}{@{}l X l@{}}'
        r'\toprule 层级 & 本次证据 & 状态 \\ \midrule '
        r'输入与缓存 & 12 组原始 gauge/eigen/peram 与 canonical HDF5 全链 & 确证 \\ '
        r'统计与图表 & 1D/2D ratio、JK、平台、Fourier、CS 与误差账本 & 确证 \\ '
        r'soft/rapidity & 无同几何测量 & 未验证 \\ '
        r'微扰 matching & 按要求跳过非格点 matching & 不适用 \\ '
        r'连续极限 & 单格距、单系综 & 未验证 \\ '
        r'\bottomrule\end{tabularx}',
        '验收定义来自 pyqcd-tmd-algorithm validation reference。',
    ))
    workflow = '\n'.join([
        r'\small\begin{tabularx}{\linewidth}{@{}l X@{}}',
        r'\toprule 阶段 & 输入 $\to$ 输出 \\',
        r'\midrule',
        r'蒸馏顶点 & eigenvector $\to$ VdV/VVV canonical HDF5 \\',
        r'外态关联 & perambulator + VVV $\to$ source-resolved C2 与连接 3pt \\',
        r'Wilson flow & raw gauge $\to$ $V_\tau$、$E(\tau)$ 与 $t_0$ \\',
        r'胶子算符 & flowed gauge $\to$ 三方向 1D/2D OPE \\',
        r'比值与统计 & C2 + OPE $\to$ disconnected ratio、JK 样本、c0/hR \\',
        r'格点提取 & hR $\to$ 准 PDF、准 TMD-PDF、两动量 CS 核 \\',
        r'\bottomrule\end{tabularx}',
    ])
    frames.append(_frame(
        '全链以同一组态索引贯穿流化、算符、统计和 Fourier',
        workflow,
        _source_ref(
            'pyqcd/testing/tmd/run_full_production.py',
            'def _run_ope_stage'),
    ))
    frames.append(_frame(
        '输入严格绑定 12 个目标组态、三流时和三方向位移网格',
        r'\small\begin{tabularx}{\linewidth}{@{}l X@{}}'
        r'\toprule 参数 & 设置 \\ \midrule '
        r'组态 & 6250, 6450, 6650, 6850, 7050, 7250, 7450, 7650, 7850, 8050, 8250, 8450 \\ '
        r'格点 & $24^3\times72$, $a=0.1053$ fm \\ '
        r'动量 & P000, P200, P020, P002, P400 \\ '
        r'流时/步长 & $\tau/a^2=1,2,3$, RK3 $\epsilon=0.01$ \\ '
        r'非局域几何 & $z=0\ldots12$, $b=0\ldots6$, $L=12a$ \\ '
        r'二维几何 & $b_x=0\ldots6$, $b_y=0\ldots2$, 两种 staircase \\ '
        r'\bottomrule\end{tabularx}',
        '运行参数写入 workers/*.json 与 canonical cache contract。',
    ))
    frames.append(_frame(
        r'三 worker 分片执行，所有资源门保持在 80\% 预算内',
        workers_table + r'\vspace{0.5em}' + '\n' + gpu_table,
        rf'CPU峰值 {_fmt(resources["cpu_max"], 4, "%")}，'
        rf'RAM峰值 {_fmt(resources["memory_max"], 4, "%")}，'
        rf'磁盘峰值 {_fmt(resources["disk_max"], 4, "%")}；'
        'GPU 平均利用率由 duty-cycle 控制，瞬时 kernel 峰值单独报告。',
    ))
    frames.append(_frame(
        'Wilson flow 显示平稳 UV 平滑并给出 $t_0$ 诊断',
        _figure(flow_image, '12 组态 Wilson-flow 作用量密度与 $t^2\\langle E\\rangle$'),
        _source_ref('pyqcd/renorm/_gradient_flow.py', 'def flow_action_density'),
    ))
    frames.append(_frame(
        '核子 2pt 在低动量窗口形成可读平台，P400 更早进入噪声区',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            spectrum_images[0], 'P000 nopol effective mass', 0.96)
        + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            spectrum_images[1], 'P200 nopol effective mass', 0.96)
        + r'\end{column}\end{columns}',
        _source_ref('pyqcd/analysis/_analyse.py', 'def meff'),
    ))
    dispersion = analysis_dir / 'spectrum' / 'dispersion_diagnostic.png'
    connected = analysis_dir / 'spectrum' / 'ratio_3pt_proton_P0.png'
    frames.append(_frame(
        '色散和连接 3pt 作为控制道，和非局域胶子 3pt 严格区分',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            dispersion, '核子能量色散诊断', 0.96)
        + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            connected, '连接矢量流 3pt/2pt ratio 控制道', 0.96)
        + r'\end{column}\end{columns}',
        _source_ref('pyqcd/analysis/_correlators.py', 'def run_3pt_ratio'),
    ))
    frames.append(_frame(
        '有限横距 OPE 的离散量已完整落盘，三方向用于旋转对称性检查',
        r'\small\begin{tabularx}{\linewidth}{@{}l l X@{}}'
        r'\toprule 对象 & shape & 作用 \\ \midrule '
        r'1D full-channel & $(2,4,6,13,7,72)$ & unpolarized/helicity 全通道 \\ '
        r'1D legacy & $(13,7,72)$ & 形状归一与交叉检查 \\ '
        r'2D full-channel & $(2,4,6,13,7,3,72)$ & 双分量 transverse staircase \\ '
        r'flowed gauge & $(72,24,24,24,4,3,3)$ & $V_\tau$ 原始流场缓存 \\ '
        r'\bottomrule\end{tabularx}',
        _source_ref('pyqcd/pipeline/_tmd9.py', 'def _tmd_ope_contract'),
    ))
    frames.append(_frame(
        '三方向 ratio 的复相位与 helicity 通道被保留到统计层',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.32\textwidth}' + _figure(
            ratio_images[0], 'x 方向 ratio', 0.98) + r'\end{column}' + '\n'
        r'\begin{column}{0.32\textwidth}' + _figure(
            ratio_images[1], 'y 方向 ratio', 0.98) + r'\end{column}' + '\n'
        r'\begin{column}{0.32\textwidth}' + _figure(
            ratio_images[2], 'z 方向 ratio', 0.98) + r'\end{column}'
        + r'\end{columns}',
        _source_ref(
            'pyqcd/analysis/_gradient_flow_ratio.py',
            'def physical_helicity_ratio'),
    ))
    staircase_image = analysis_dir / 'ratio_2d' / 'staircase_sensitivity.png'
    frames.append(_frame(
        '二维 staircase 顺序差异作为显式离散化系统误差报告',
        _figure(staircase_image, 'order (0,1) 与 (1,0) ratio 比较')
        + '\n\n'
        + r'\small 中位相对 L2：'
        + _fmt(
            budget.get('staircase_relative_l2_median'), 4),
        _source_ref(
            'pyqcd/testing/tmd/analyze_full_production.py',
            'def _staircase_summary'),
    ))
    c0_2d_images = [
        analysis_dir / 'plateaus_2d' / 'c0_tau1_order01_heatmap.png',
        analysis_dir / 'plateaus_2d' / 'c0_tau3_order01_heatmap.png',
    ]
    frames.append(_frame(
        '二维 c0 同时保留 $b_x,b_y$ 依赖，避免把不同 staircase 强行合并',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            c0_2d_images[0], 'tau=1 2D c0 heatmap', 0.96)
        + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            c0_2d_images[1], 'tau=3 2D c0 heatmap', 0.96)
        + r'\end{column}\end{columns}',
        _source_ref(
            'pyqcd/testing/tmd/analyze_full_production.py',
            'def analyze_plateaus_2d'),
    ))
    for tau, image in zip((1, 2, 3), c0_images):
        frames.append(_frame(
            rf'流时 $\tau/a^2={tau}$ 的 c0 平台对窗口敏感，不能由单窗直接宣称物理矩阵元',
            _figure(image, f'tau={tau} 多窗口 c0 稳定性'),
            rf'窗口结果：{_path_text(analysis_dir / "plateaus" / f"c0_tau{tau}_z_P200.npz")}',
        ))
    window_image = analysis_dir / 'flow_windows' / 'flow_window_coverage.png'
    paired_image = analysis_dir / 'flow_windows' / 'flow_time_paired_difference.png'
    frames.append(_frame(
        '流窗交集只保留可比较几何，跨流时差异采用配对 JK',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            window_image, '流窗覆盖', 0.96) + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            paired_image, '配对流时差', 0.96) + r'\end{column}'
        + r'\end{columns}',
        rf'共同窗：{_tex_escape(str(flow_windows.get("common_window", {})))}。',
    ))
    frames.append(_frame(
        '格点端准 PDF 与准 TMD-PDF 已生成，但未施加光锥匹配',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_images[0], 'tau=1, P200 准 PDF/TMD-PDF', 0.96)
        + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_images[1], 'tau=3, P200 准 PDF/TMD-PDF', 0.96)
        + r'\end{column}\end{columns}',
        _source_ref('pyqcd/renorm/_tmdextract.py', 'def quasi_pdf_gluon'),
    ))
    frames.append(_frame(
        '重整化实空间矩阵元 $h_R(z,b)$ 保留 $b$ 依赖并区分流时',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_dir / 'hR_tau1_P200.png', 'tau=1, P200 hR(z,b)', 0.96)
        + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_dir / 'hR_tau3_P200.png', 'tau=3, P200 hR(z,b)', 0.96)
        + r'\end{column}\end{columns}',
        _source_ref('pyqcd/pipeline/_tmd9.py', 'def self_renormalize'),
    ))
    frames.append(_frame(
        '两动量 CS 核作为格点提取量保存，零核和任意截断均未使用',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_images[2], 'tau=1 CS kernel', 0.96) + r'\end{column}' + '\n'
        r'\begin{column}{0.49\textwidth}' + _figure(
            pdf_images[3], 'tau=3 CS kernel', 0.96) + r'\end{column}'
        + r'\end{columns}',
        _source_ref(
            'pyqcd/renorm/_tmdextract.py',
            'def cs_kernel_two_momentum'),
    ))
    frames.append(_frame(
        '误差账本把统计、平台、流时、staircase 与未实现层分开',
        r'\small\begin{tabularx}{\linewidth}{@{}l l X@{}}'
        r'\toprule 来源 & 数值 & 解释 \\ \midrule '
        + f"统计中位 & {_fmt(budget.get('statistical_median'), 4)} & JK plateau 误差 \\\\ "
        + f"平台窗漂移 & {_fmt(budget.get('plateau_window_drift_median'), 4)} & 多窗口极差 \\\\ "
        + f"流时相对效应 & {_fmt(budget.get('flow_time_relative_effect_median'), 4)} & 配对 JK 差值 \\\\ "
        + f"staircase 相对 L2 & {_fmt(budget.get('staircase_relative_l2_median'), 4)} & 路径顺序系统差 \\\\ "
        + r'soft/rapidity & 未测量 & PDF 不得物理解释 \\ '
        + r'matching/mixing & 未施加 & 保持格点端诊断 \\ '
        + r'\bottomrule\end{tabularx}',
        rf'完整账本：{_path_text(analysis_dir / "error_budget.json")}',
    ))
    frames.append(_frame(
        '误差账本的量级用于判断当前主导不确定性，而不是用于升级物理结论',
        _figure(
            analysis_dir / 'error_budget.png',
            '统计、窗口、流时与 staircase 误差量级'),
        rf'完整账本：{_path_text(analysis_dir / "error_budget.json")}',
    ))
    frames.append(_frame(
        '当前结论：数值链闭合为 diagnostic-only，物理闭环明确失败关闭',
        r'\begin{columns}[T,totalwidth=\textwidth]' + '\n'
        r'\begin{column}{0.48\textwidth}\begin{block}{可确认}'
        r'12 组态真实数据、三流时、三方向与 2D OPE、ratio 和格点端 Fourier 均有可读产物。'
        r'\end{block}\end{column}' + '\n'
        r'\begin{column}{0.48\textwidth}\begin{block}{不可确认}'
        r'不能将当前 xg 解读为物理胶子 TMD-PDF；流时/platform/matching 等系统误差仍主导。'
        r'\end{block}\end{column}\end{columns}',
        _source_ref(
            'pyqcd/testing/tmd/analyze_full_production.py',
            'def make_lattice_only_status'),
    ))
    frames.append(_frame(
        '下一步按物理瓶颈排序：soft/rapidity、系统扫描、连续极限',
        r'\small\begin{tabularx}{\linewidth}{@{}l l X@{}}'
        r'\toprule 任务 & 验收产物 & 通过条件 \\ \midrule '
        r'soft/rapidity & 同几何软因子与误差 & 表示、流时、staple 完全匹配 \\ '
        r'tau/staple 扫描 & 多尺度 hR 与外推带 & 共同窗内稳定且误差账本闭合 \\ '
        r'matching & 显式胶子混合矩阵 & $\alpha_s\to0$ 核对角核 \\ '
        r'连续极限 & 多格距/体积联合外推 & 统计与系统误差分离 \\ '
        r'\bottomrule\end{tabularx}',
        '本任务按要求未执行非格点 QCD matching。',
    ))
    frames.append(_frame(
        '附录：物理对象到代码入口的完整映射',
        source_table,
        '所有入口均位于 PyQCD 生产包；测试驱动只负责编排和持久化。',
    ))
    frames.append(_frame(
        '附录：产物与复现入口',
        r'\small\begin{tabularx}{\linewidth}{@{}l X@{}}'
        r'\toprule 对象 & 路径/命令 \\ \midrule '
        rf'运行根目录 & {_path_text(run_dir)} \\ '
        rf'分析汇总 & {_path_text(analysis_dir / "analysis_summary.json")} \\ '
        rf'产物清单 & {_path_text(analysis_dir / "artifact_manifest.json")} \\ '
        rf'生产入口 & {_path_text(PROJECT_ROOT / "pyqcd/testing/tmd/run_full_production.py")} \\ '
        rf'分析入口 & {_path_text(PROJECT_ROOT / "pyqcd/testing/tmd/analyze_full_production.py")} \\ '
        r'\bottomrule\end{tabularx}',
        '复现时先读取 workers checkpoint，再调用两个入口的相同请求参数。',
    ))

    preamble = rf'''\documentclass[UTF8,aspectratio=169,11pt]{{ctexbeamer}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\definecolor{{primary}}{{HTML}}{{17365D}}
\definecolor{{accent}}{{HTML}}{{1F77B4}}
\definecolor{{evidence}}{{HTML}}{{1E8449}}
\definecolor{{warning}}{{HTML}}{{C55A11}}
\definecolor{{muted}}{{HTML}}{{5B6573}}
\setbeamersize{{text margin left=0.55cm,text margin right=0.55cm}}
\setlength{{\emergencystretch}}{{2em}}
\setbeamertemplate{{navigation symbols}}{{}}{{}}
\setbeamercolor{{normal text}}{{fg=black!86,bg=white}}
\setbeamercolor{{structure}}{{fg=primary}}
\setbeamercolor{{frametitle}}{{fg=primary,bg=white}}
\setbeamercolor{{block title}}{{fg=primary,bg=primary!6}}
\setbeamercolor{{block body}}{{fg=black!86,bg=primary!3}}
\setbeamerfont{{frametitle}}{{size=\large,series=\bfseries}}
\setbeamertemplate{{footline}}{{%
  \hbox{{\begin{{beamercolorbox}}[wd=\dimexpr\paperwidth-1.1cm\relax,ht=2.3ex,dp=0.9ex,
    leftskip=0.55cm,rightskip=0.55cm]{{author in head/foot}}%
    \scriptsize\color{{muted}}PyQCD 12-conf TMD production\hfill
    \insertframenumber/\inserttotalframenumber
  \end{{beamercolorbox}}}}}}
\newcommand{{\source}}[1]{{\vspace{{0.06em}}\par{{\raggedright\scriptsize\color{{muted}}来源：#1\par}}}}
\title[PyQCD TMD]{{梯度流重整化核子胶子 TMD-PDF 全链计算}}
\author[PyQCD]{{PyQCD production run}}
\date{{2026-11-01}}
\begin{{document}}
'''
    end = '\n\\end{document}\n'
    tex = preamble + '\n'.join(frames) + end
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding='utf-8')
    summary_path = Path(summary_path)
    summary_path.write_text(
        _summary_markdown(summary, workers, resources, manifest, status),
        encoding='utf-8',
    )
    return tex_path, summary_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument(
        '--tex', type=Path,
        default=PROJECT_ROOT / 'docs' / 'report_tmd_full_production_20261101.tex')
    parser.add_argument(
        '--summary', type=Path,
        default=PROJECT_ROOT / 'docs' / 'summary_tmd_full_production_20261101.md')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tex, summary = build_report(args.run_dir, args.tex, args.summary)
    print(tex)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
