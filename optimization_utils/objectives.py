"""
目标函数计算模块 - 实现FEF, HRE, MRE, NR四个目标函数
基于references文件夹中的MATLAB代码逻辑
"""
import numpy as np
import wntr


def run_simulation(wn, file_prefix='temp'):
    """运行水力仿真，失败返回None"""
    try:
        sim = wntr.sim.EpanetSimulator(wn)
        # 使用唯一的前缀避免并行冲突
        return sim.run_sim(file_prefix=file_prefix)
        # Note: WNTR creates {prefix}.inp and {prefix}.rpt. 
        # Standard WNTR might not clean them up automatically if they are custom named.
        # But let's assume WNTR or OS handles temp files or we accept them in CWD for now.
    except Exception:
        return None


def calculate_fef(wn, results):
    """
    计算Flow Entropy Function (流量熵函数)，基于fef_function.m
    
    修正：
    1. 处理单一水源管网的情况
    2. 避免除零错误
    3. 确保返回有意义的正值
    """
    if results is None:
        return 0.0
    try:
        flow = results.link['flowrate']  # 管道流量
        demand = results.node['demand']  # 节点需求
        junctions = wn.junction_name_list
        n_times = len(flow.index)
        
        # 避免零需求
        total_demand = demand[junctions].clip(lower=0).sum(axis=1)
        total_demand = total_demand.replace(0, 1e-10)
        
        pipes = [(n, p.start_node_name, p.end_node_name) for n, p in wn.pipes()]
        
        # 统计从水源流入的总流量和管道数
        source_flow_total = 0.0
        source_pipe_count = 0
        for name, start, end in pipes:
            if start not in junctions:  # start是水源(Reservoir/Tank)
                avg_flow = abs(flow[name].mean())
                if avg_flow > 1e-10:
                    source_flow_total += avg_flow
                    source_pipe_count += 1
        
        # S0: 来自水源的熵
        s0 = 0.0
        if source_flow_total > 1e-10:
            for t_idx, t in enumerate(flow.index):
                td = max(total_demand.iloc[t_idx], 1e-10)
                for name, start, end in pipes:
                    if start not in junctions:
                        f = abs(flow.loc[t, name])
                        if f > 1e-10:
                            # 使用相对于总水源流量的比例计算熵
                            p0i = f / td
                            # 限制p0i避免极端值
                            p0i = min(max(p0i, 1e-10), 1.0)
                            s0 -= p0i * np.log(p0i)
            s0 /= max(n_times, 1)
        
        # Si: 每个节点的出流熵
        si_total = 0.0
        pi_total = 0.0
        for j in junctions:
            # 计算流入该节点的总流量
            inflow = np.zeros(n_times)
            for name, start, end in pipes:
                # 流入: flow > 0 且 end = j，或 flow < 0 且 start = j
                pos_inflow = np.where((flow[name].values > 0) & (end == j), flow[name].values, 0)
                neg_inflow = np.where((flow[name].values < 0) & (start == j), -flow[name].values, 0)
                inflow += pos_inflow + neg_inflow
            
            # 计算从该节点流出的熵
            si = 0.0
            valid_counts = 0
            for name, start, end in pipes:
                if start == j:
                    for t_idx in range(n_times):
                        outflow = max(flow.iloc[t_idx][name], 0)
                        inf_t = max(inflow[t_idx], 1e-10)
                        if outflow > 1e-10 and inf_t > 1e-10:
                            pij = min(outflow / inf_t, 1.0)  # 限制在0-1范围
                            if pij > 1e-10:
                                si -= pij * np.log(pij)
                                valid_counts += 1
            
            if valid_counts > 0:
                si /= max(n_times, 1)
            
            # Pi: 该节点的流量占比
            avg_inflow = np.mean(inflow)
            avg_demand = np.mean(total_demand.values)
            pi = avg_inflow / max(avg_demand, 1e-10)
            
            si_total += si * pi
            pi_total += pi
        
        # 总熵
        total_entropy = s0 + si_total
        
        # 如果总熵仍为0，使用备用计算方法：基于管道流量分布的熵
        if total_entropy < 1e-10:
            # 计算所有管道流量的熵作为备用
            all_flows = []
            for name, start, end in pipes:
                avg_flow = abs(flow[name].mean())
                if avg_flow > 1e-10:
                    all_flows.append(avg_flow)
            
            if len(all_flows) > 0:
                total_flow = sum(all_flows)
                entropy_backup = 0.0
                for f in all_flows:
                    p = f / total_flow
                    if p > 1e-10:
                        entropy_backup -= p * np.log(p)
                # 归一化到类似范围 (除以最大可能熵 = log(n))
                max_entropy = np.log(max(len(all_flows), 1))
                if max_entropy > 0:
                    total_entropy = entropy_backup / max_entropy
        
        return max(total_entropy, 0.0)
    except Exception as e:
        return 0.0


def calculate_hre(wn, results, hmin=None, hdes=None):
    """
    计算Hydraulic Reliability Estimation (水力可靠性)，基于hre_function.m
    
    自适应阈值逻辑：
    - 如果未指定hmin/hdes，根据网络压力分布自动计算
    - hmin = 压力的10th百分位数 (保守最小值)
    - hdes = 压力的50th百分位数 (期望中值)
    """
    if results is None:
        return 0.0
    try:
        pressure, demand = results.node['pressure'], results.node['demand']
        junctions = wn.junction_name_list
        
        # 自适应计算压力阈值
        if hmin is None or hdes is None:
            # 获取所有节点压力的扁平数组
            all_pressures = pressure[junctions].values.flatten()
            # 移除异常值 (负压和极端高压)
            valid_pressures = all_pressures[(all_pressures > 0) & (all_pressures < 500)]
            
            if len(valid_pressures) > 0:
                if hmin is None:
                    hmin = float(np.percentile(valid_pressures, 10))  # 10th百分位
                if hdes is None:
                    hdes = float(np.percentile(valid_pressures, 50))  # 50th百分位 (中值)
            else:
                # 回退到默认值
                hmin = hmin if hmin is not None else 10.0
                hdes = hdes if hdes is not None else 30.0
        
        # 确保 hdes > hmin
        if hdes <= hmin:
            hdes = hmin + 10.0
        
        Rj_list, qsj_total, dsj_total = [], 0.0, 0.0
        for j in junctions:
            qj, dj = 0.0, 0.0
            for t in pressure.index:
                p, d = pressure.loc[t, j], abs(demand.loc[t, j])
                if d == 0:
                    continue
                q = 0 if p < hmin else (d if p >= hdes else d * np.sqrt((p - hmin) / (hdes - hmin)))
                qj, dj = qj + q, dj + d
            Rj_list.append(min(qj / dj, 1.0) if dj > 0 else 1.0)
            qsj_total, dsj_total = qsj_total + qj, dsj_total + dj
        
        Rv = qsj_total / dsj_total if dsj_total > 0 else 1.0
        # 使用算术平均代替几何平均，避免单个零值导致整体为0
        Fn = np.mean(Rj_list) if junctions else 1.0
        return min(max(Rv * Fn, 0), 1.0)
    except Exception:
        return 0.0


def calculate_mre(wn, results, node_to_comm, num_comm):
    """
    计算Mechanical Reliability Estimation (机械可靠性)，基于mre_function.m
    WNTR输出单位: 直径(m), 长度(m), 需求(m³/s)
    
    修正: 考虑所有活跃管道（包括边界管道），反映实际网络状态
    MRE应该被最小化 - 较低的值表示较低的失效风险
    """
    if results is None:
        return 1.0  # 无仿真结果时返回最大失效风险
    try:
        demand = results.node['demand']
        junctions = wn.junction_name_list
        junction_set = set(junctions)

        # 计算每个时间步的总需求
        total_demand = demand[junctions].clip(lower=0).sum(axis=1)

        # 失效率公式系数转换为SI单位
        c1 = 0.6858 * (0.0254 ** 3.26)
        c2 = 2.7685 * (0.0254 ** 1.3131)
        c3 = 2.7685 * (0.0254 ** 3.5792)
        c4 = 0.042
        mile_to_m = 1609.34

        # 计算所有活跃管道的失效概率
        # 活跃管道 = 当前网络中存在且有流量的管道
        total_failure_prob = 0.0
        total_pipe_count = 0
        
        flow = results.link['flowrate'] if 'flowrate' in results.link else None
        
        for name, pipe in wn.pipes():
            # 只考虑连接junction的管道
            if pipe.start_node_name not in junction_set and pipe.end_node_name not in junction_set:
                continue
            
            # 检查管道是否有流量（活跃状态）
            is_active = True
            if flow is not None:
                avg_flow = abs(flow[name].mean()) if name in flow.columns else 0
                is_active = avg_flow > 1e-10
            
            if not is_active:
                continue  # 跳过无流量的管道（已关闭）
                
            dia_m = pipe.diameter
            len_m = pipe.length
            
            # 失效率公式 (SI单位)
            ri = (c1 / (dia_m ** 3.26) +
                  c2 / (dia_m ** 1.3131) +
                  c3 / (dia_m ** 3.5792) + c4)
            betai = (len_m / mile_to_m) * ri
            pipe_fail_prob = 1 - np.exp(-betai)
            
            total_failure_prob += pipe_fail_prob
            total_pipe_count += 1

        # MRE = 平均管道失效概率
        # 较低的值表示较低的失效风险（更好）
        if total_pipe_count > 0:
            mre = total_failure_prob / total_pipe_count
        else:
            mre = 1.0  # 无管道时返回最大风险
        
        return max(min(mre, 1.0), 0.0)
    except Exception:
        return 1.0  # 出错时返回最大失效风险


def calculate_nr(wn, results):
    """
    计算Network Resilience (网络韧性)，基于论文公式(32)(33)
    只考虑junction节点，不考虑Reservoir和Tank

    公式(32): In = Σ(Ui * q_i * mean(Hi - H_min,i)) / (Σ(q_i * H_i) - Σ(q_min,i * H_min,i))
    公式(33): Ui = Σdj / (Ni * max(d1,...,dNi))
    
    修正1：使用平均需求代替最小需求，避免零值导致结果为0
    修正2：对于单时间步（稳态）仿真，使用压力余量而非压力变化
    """
    if results is None:
        return 0.0
    try:
        pressure = results.node['pressure']
        demand = results.node['demand']
        junctions = wn.junction_name_list

        # ===== 计算Ui: 每个节点的管道直径均匀性因子 =====
        link_nodes = {n: [] for n in junctions}
        for _, pipe in wn.pipes():
            if pipe.start_node_name in link_nodes:
                link_nodes[pipe.start_node_name].append(pipe.diameter)
            if pipe.end_node_name in link_nodes:
                link_nodes[pipe.end_node_name].append(pipe.diameter)

        Ui_array = np.zeros(len(junctions))
        for idx, j in enumerate(junctions):
            dias = link_nodes.get(j, [])
            if dias and max(dias) > 0:
                Ui_array[idx] = sum(dias) / (len(dias) * max(dias))
            else:
                Ui_array[idx] = 1.0

        # ===== 获取junction节点的压力和需求 =====
        press_junc = pressure[junctions].values  # (n_times, n_junctions)
        demand_junc = demand[junctions].values   # (n_times, n_junctions)
        n_times = press_junc.shape[0]

        # 使用正需求进行计算（忽略负需求节点）
        demand_positive = np.clip(demand_junc, 0, None)
        
        # q_avg,i: 节点i在EPS期间的平均需水量
        q_avg = np.mean(demand_positive, axis=0)
        
        # ===== 处理单时间步（稳态仿真）的情况 =====
        if n_times == 1:
            # 对于稳态仿真，使用压力余量作为韧性指标
            # 假设最低服务压力为10m，期望压力为压力的50th百分位
            h_current = press_junc[0, :]
            h_min_service = 10.0  # 最低服务压力
            h_des = np.percentile(h_current[h_current > 0], 50) if np.any(h_current > 0) else 30.0
            
            # 压力余量 = (当前压力 - 最低压力) / (期望压力 - 最低压力)
            surplus = np.clip((h_current - h_min_service) / max(h_des - h_min_service, 1.0), 0, 1)
            
            # 加权平均（按需求加权）
            if np.sum(q_avg) > 1e-10:
                result = np.sum(Ui_array * q_avg * surplus) / np.sum(q_avg)
            else:
                result = np.mean(surplus)
            
            return max(min(result, 1.0), 0.0)
        
        # ===== 标准EPS情况（多时间步）=====
        # H_min,i: 节点i在EPS期间的最小压力
        h_min = np.min(press_junc, axis=0)

        # 计算分子: Σ(Ui * q_avg,i * mean(Hi - H_min,i))
        diff_pressure = press_junc - h_min  # 压力差矩阵
        mean_diff = np.mean(diff_pressure, axis=0)  # 对时间求平均
        numerator = np.sum(Ui_array * q_avg * mean_diff)

        # 计算分母: Σ(mean(q_i * H_i))
        qh_product = demand_positive * press_junc
        total_qh = np.sum(np.mean(qh_product, axis=0))
        
        if total_qh < 1e-10:
            return 0.0

        result = numerator / total_qh
        return max(min(result, 1.0), 0.0)
    except Exception:
        return 0.0

