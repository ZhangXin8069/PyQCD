# import os
# import sys

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, '/public/home/sush/distillation/')

# from function_contraction import *

# def Jackknife(data, Nconf_axes=0, only_sample:bool=False, cov_axes = None):
#     """
#     Jackknife 重采样函数，支持沿指定轴构建协方差矩阵。

#     参数
#     ----------
#     data : array
#         输入数据数组，至少包含组态轴。
#     Nconf_axes : int
#         组态所在的轴编号（默认为 0）。
#     only_sample : bool
#         若为 True，仅返回 Jackknife 样本，不计算均值、误差和协方差。
#     cov_axes : int 或 tuple 或 None
#         指定构建协方差矩阵的轴。对该轴求外积，其他轴保持不变。
#         若为 None，则不计算协方差矩阵。

#     返回
#     ----------
#     dict:
#         'data_sample'  — Jackknife 样本，形状与输入相同
#         'data_mean'    — 数据均值（去掉组态轴）
#         'data_err'     — 标准误差 sqrt(Nconf-1) * std(samples)
#         'data_cov'     — 协方差矩阵（仅当 cov_axes 不为 None 时返回）
#     """
#     backend = get_backend()
#     ndim = data.ndim
#     Nconf_axes = Nconf_axes % ndim
#     Nconf = data.shape[Nconf_axes]

#     # 1. 所有组态的 Jackknife 样本（去掉第 k 个组态的均值）
#     data_sum = backend.sum(data, axis=Nconf_axes, keepdims=True)
#     data_sample = -(data - data_sum) / (Nconf - 1)

#     if only_sample:
#         return {'data_sample': data_sample}

#     # 2. 均值与标准误差
#     data_mean = backend.mean(data_sample, axis=Nconf_axes)
#     data_err = backend.sqrt(Nconf - 1) * backend.std(data_sample, axis=Nconf_axes)

#     result = {'data_sample': data_sample,
#               'data_mean': data_mean,
#               'data_err': data_err}

#     # 3. 协方差矩阵（均值的协方差）
#     if cov_axes is not None:
#         if isinstance(cov_axes, int):
#             cov_axes = (cov_axes % ndim,)
#         else:
#             cov_axes = tuple(ax % ndim for ax in cov_axes)

#         # 残差：Jackknife 样本 - 均值（自动广播）
#         residual = data_sample - backend.mean(data_sample, axis=Nconf_axes, keepdims = True)

#         # 确定轴顺序：组态轴 -> 其他轴 -> 协方差轴
#         all_axes = list(range(ndim))
#         other_axes = [ax for ax in all_axes if ax != Nconf_axes and ax not in cov_axes]
#         new_order = [Nconf_axes] + other_axes + list(cov_axes)
#         r = backend.transpose(residual, new_order)

#         # 展平协方差轴以便使用 einsum
#         shape_other = [residual.shape[ax] for ax in other_axes]
#         shape_cov = [residual.shape[ax] for ax in cov_axes]
#         N_cov = 1
#         for s in shape_cov:
#             N_cov *= s
#         r_flat = r.reshape([Nconf] + shape_other + [N_cov])

#         # 对每个组态计算外积，再求和；可以启用优化路径加速
#         cov_sum = backend.einsum('n...i,n...j->...ij', r_flat, r_flat, optimize=True)

#         # 均值协方差矩阵 = (N-1)/N * 外积和
#         cov = cov_sum * (Nconf - 1) / Nconf
#         result['data_cov'] = cov.reshape(shape_other + shape_cov + shape_cov)

#     return result


# def Bootstrap(data, Nconf_axes=0, only_sample=False, cov_axes=None, M=0, N=0):
#     """
#     Bootstrap 重采样函数，支持沿指定轴构建协方差矩阵。

#     参数
#     ----------
#     data : array
#         输入数据数组，至少包含组态轴。
#     Nconf_axes : int
#         组态所在的轴编号（默认为 0）。
#     only_sample : bool
#         若为 True，仅返回 Bootstrap 样本，不计算均值、误差和协方差。
#     cov_axes : int 或 tuple 或 None
#         指定构建协方差矩阵的轴。对该轴求外积，其他轴保持不变。
#         若为 None，则不计算协方差矩阵。
#     M : int
#         每个 Bootstrap 样本抽取的组态数。默认为 Nconf - 5。
#     N : int
#         Bootstrap 样本总数。默认为 Nconf * 4。

#     返回
#     ----------
#     dict:
#         'data_sample'  — Bootstrap 样本，形状 (N, dim1, dim2, ...)
#         'data_mean'    — Bootstrap 样本均值
#         'data_err'     — Bootstrap 样本标准差
#         'data_cov'     — 协方差矩阵（仅当 cov_axes 不为 None 时返回）
#     """
#     backend = get_backend()
#     ndim = data.ndim
#     Nconf_axes = Nconf_axes % ndim

#     datashape = list(data.shape)
#     Nconf = datashape[Nconf_axes]

#     # 默认参数
#     if M == 0:
#         M = max(Nconf - 5, 1)
#     if N == 0:
#         N = Nconf * 4

#     # 参数校验
#     if M > Nconf:
#         raise ValueError(f'M ({M}) 不能大于 Nconf ({Nconf})')

#     # 生成 bootstrap 样本：每个样本有放回地随机抽取 M 个组态并求均值
#     sample_shape = tuple(datashape[:Nconf_axes] + datashape[Nconf_axes+1:])
#     data_sample = backend.zeros((N,) + sample_shape)
#     for i in range(N):
#         indices = backend.random.choice(Nconf, size=M, replace=True)
#         selected = backend.take(data, indices, axis=Nconf_axes)
#         data_sample[i] = backend.mean(selected, axis=Nconf_axes)

#     if only_sample:
#         return {'data_sample': data_sample}

#     # 计算 bootstrap 样本的均值和标准差
#     data_mean = backend.mean(data_sample, axis=0)
#     data_err = backend.std(data_sample, axis=0)

#     result = {
#         'data_sample': data_sample,
#         'data_mean': data_mean,
#         'data_err': data_err,
#     }

#     # 沿用户指定的轴构建协方差矩阵（参照 Jackknife 模式）
#     if cov_axes is not None:
#         if isinstance(cov_axes, int):
#             cov_axes = (cov_axes % ndim,)
#         else:
#             cov_axes = tuple(ax % ndim for ax in cov_axes)

#         # cov_axes 相对于原始数据轴编号
#         # data_sample 形状为 (N, *sample_shape)，Nconf_axes 已移除，N 轴插入到位置 0
#         # 映射关系：原始轴 < Nconf_axes -> 新轴 = 原始轴 + 1；原始轴 > Nconf_axes -> 新轴 = 原始轴
#         cov_axes_r = tuple(ax + 1 if ax < Nconf_axes else ax for ax in cov_axes)

#         # 计算残差：bootstrap 样本 - 样本均值
#         residual = data_sample - backend.mean(data_sample, axis=Nconf_axes, keepdims = True)  # 形状: (N, dim1, dim2, ...)

#         # 确定轴顺序：N 轴 -> 其他轴 -> 协方差轴
#         ndim_sample = residual.ndim  # 包含 N 轴的总维度
#         all_axes = list(range(ndim_sample))
#         other_axes = [ax for ax in all_axes if ax != 0 and ax not in cov_axes_r]
#         new_order = [0] + other_axes + list(cov_axes_r)
#         r = backend.transpose(residual, new_order)

#         # 记录各维度大小
#         shape_other = [residual.shape[ax] for ax in other_axes]
#         shape_cov = [residual.shape[ax] for ax in cov_axes_r]
#         N_cov = 1
#         for s in shape_cov:
#             N_cov *= s

#         # 展平：(N, *shape_other, N_cov)
#         r_flat = r.reshape((N,) + tuple(shape_other) + (N_cov,))

#         # 通过外积计算协方差矩阵，对 bootstrap 样本轴取均值
#         cov = backend.einsum('n...i,n...j->...ij', r_flat, r_flat) / N

#         # 将协方差轴恢复为原始维度
#         result['data_cov'] = cov.reshape(tuple(shape_other) + tuple(shape_cov) + tuple(shape_cov))

#     return result


