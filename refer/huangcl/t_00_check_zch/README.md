ratio 的计算代码作者是曾春华师兄, 路径: /public/group/imp/zengch/LQCD/renorma/
画图代码的作者是本人.
3 个核心文件, 均读取 input_file_L24x72_dhxmean.py:
1.  C_pt_load.py: 先跑完各个方向的 ratio, 再对方向平均. 数据保存在 ./result_s
    原文件有 bug, ope 只有 3 个方向, 但 2pt 有 6 个方向, 已修改.(别的格子不清楚用了几个方向, 可能 2pt 也是 3 个方向!!!)
    253 行:
    '''
    c2pt_data = (c2pt_data[:3]+c2pt_data[3:])/2
    '''
    另外, 加入了自动生成保存路径的代码.
    

2.  C_pt_load_L24x72_dhxmean.py: 现在组态层次分别对 2pt, ope 的方向平均, 再做 ratio. 数据保存在 ./result_l
    源文件有 bug, 133 行已对 2pt 的方向进行平均, 根据数组形状, 2pt 的真空期望是对 axis=0 的轴进行平均, 但 216 行是对 axis=1 的轴平均, 已注释, 并在 217 行修正.
    另外, 加入了自动生成保存路径的代码.

3.  code_zch_ratio.py: 将 ratio 的数据重构成作者常用的形式, 并画出 z=0 的 ratio 图.
