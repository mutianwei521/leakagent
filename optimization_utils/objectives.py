"""
Objective function calculation module - Implements FEF, HRE, MRE, NR objective functions.
Based on logic from MATLAB code in references folder.
"""
import numpy as np
import wntr


def run_simulation(wn, file_prefix='temp'):
    """Run hydraulic simulation, return None on failure"""
    try:
        sim = wntr.sim.EpanetSimulator(wn)
        # Use unique prefix to avoid parallel conflicts
        return sim.run_sim(file_prefix=file_prefix)
        # Note: WNTR creates {prefix}.inp and {prefix}.rpt. 
        # Standard WNTR might not clean them up automatically if they are custom named.
        # But let's assume WNTR or OS handles temp files or we accept them in CWD for now.
    except Exception:
        return None


def calculate_fef(wn, results):
    """
    Calculate Flow Entropy Function (FEF), based on fef_function.m
    
    Corrections:
    1. Handle single source networks
    2. Avoid division by zero
    3. Ensure meaningful positive values are returned
    """
    if results is None:
        return 0.0
    try:
        flow = results.link['flowrate']  # Pipe flowrate
        demand = results.node['demand']  # Node demand
        junctions = wn.junction_name_list
        n_times = len(flow.index)
        
        # Avoid zero demand
        total_demand = demand[junctions].clip(lower=0).sum(axis=1)
        total_demand = total_demand.replace(0, 1e-10)
        
        pipes = [(n, p.start_node_name, p.end_node_name) for n, p in wn.pipes()]
        
        # Calculate total flow and pipe count from sources
        source_flow_total = 0.0
        source_pipe_count = 0
        for name, start, end in pipes:
            if start not in junctions:  # start is a source (Reservoir/Tank)
                avg_flow = abs(flow[name].mean())
                if avg_flow > 1e-10:
                    source_flow_total += avg_flow
                    source_pipe_count += 1
        
        # S0: Entropy from sources
        s0 = 0.0
        if source_flow_total > 1e-10:
            for t_idx, t in enumerate(flow.index):
                td = max(total_demand.iloc[t_idx], 1e-10)
                for name, start, end in pipes:
                    if start not in junctions:
                        f = abs(flow.loc[t, name])
                        if f > 1e-10:
                            # Calculate entropy using proportion relative to total source flow
                            p0i = f / td
                            # Clip p0i to avoid extreme values
                            p0i = min(max(p0i, 1e-10), 1.0)
                            s0 -= p0i * np.log(p0i)
            s0 /= max(n_times, 1)
        
        # Si: Outflow entropy for each node
        si_total = 0.0
        pi_total = 0.0
        for j in junctions:
            # Calculate total inflow to the node
            inflow = np.zeros(n_times)
            for name, start, end in pipes:
                # Inflow: flow > 0 and end = j, or flow < 0 and start = j
                pos_inflow = np.where((flow[name].values > 0) & (end == j), flow[name].values, 0)
                neg_inflow = np.where((flow[name].values < 0) & (start == j), -flow[name].values, 0)
                inflow += pos_inflow + neg_inflow
            
            # Calculate entropy of outflow from the node
            si = 0.0
            valid_counts = 0
            for name, start, end in pipes:
                if start == j:
                    for t_idx in range(n_times):
                        outflow = max(flow.iloc[t_idx][name], 0)
                        inf_t = max(inflow[t_idx], 1e-10)
                        if outflow > 1e-10 and inf_t > 1e-10:
                            pij = min(outflow / inf_t, 1.0)  # Limit to 0-1 range
                            if pij > 1e-10:
                                si -= pij * np.log(pij)
                                valid_counts += 1
            
            if valid_counts > 0:
                si /= max(n_times, 1)
            
            # Pi: Flow proportion of the node
            avg_inflow = np.mean(inflow)
            avg_demand = np.mean(total_demand.values)
            pi = avg_inflow / max(avg_demand, 1e-10)
            
            si_total += si * pi
            pi_total += pi
        
        # Total entropy
        total_entropy = s0 + si_total
        
        # If total entropy is still 0, use backup calculation: entropy based on pipe flow distribution
        if total_entropy < 1e-10:
            # Calculate entropy of all pipe flows as backup
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
                # Normalize to similar range (divide by max possible entropy = log(n))
                max_entropy = np.log(max(len(all_flows), 1))
                if max_entropy > 0:
                    total_entropy = entropy_backup / max_entropy
        
        return max(total_entropy, 0.0)
    except Exception as e:
        return 0.0


def calculate_hre(wn, results, hmin=None, hdes=None):
    """
    Calculate Hydraulic Reliability Estimation (HRE), based on hre_function.m
    
    Adaptive threshold logic:
    - If hmin/hdes are not specified, automatically calculate based on network pressure distribution
    - hmin = 10th percentile of pressure (conservative minimum)
    - hdes = 50th percentile of pressure (expected median)
    """
    if results is None:
        return 0.0
    try:
        pressure, demand = results.node['pressure'], results.node['demand']
        junctions = wn.junction_name_list
        
        # Adaptive calculation of pressure thresholds
        if hmin is None or hdes is None:
            # Get flattened array of all node pressures
            all_pressures = pressure[junctions].values.flatten()
            # Remove outliers (negative pressures and extremely high pressures)
            valid_pressures = all_pressures[(all_pressures > 0) & (all_pressures < 500)]
            
            if len(valid_pressures) > 0:
                if hmin is None:
                    hmin = float(np.percentile(valid_pressures, 10))  # 10th percentile
                if hdes is None:
                    hdes = float(np.percentile(valid_pressures, 50))  # 50th percentile (median)
            else:
                # Fallback to default values
                hmin = hmin if hmin is not None else 10.0
                hdes = hdes if hdes is not None else 30.0
        
        # Ensure hdes > hmin
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
        # Use arithmetic mean instead of geometric mean to avoid a single zero value causing overall 0
        Fn = np.mean(Rj_list) if junctions else 1.0
        return min(max(Rv * Fn, 0), 1.0)
    except Exception:
        return 0.0


def calculate_mre(wn, results, node_to_comm, num_comm):
    """
    Calculate Mechanical Reliability Estimation (MRE), based on mre_function.m
    WNTR output units: Diameter (m), Length (m), Demand (m³/s)
    
    Correction: Consider all active pipes (including boundary pipes), reflecting actual network state
    MRE should be minimized - lower values indicate lower failure risk
    """
    if results is None:
        return 1.0  # Return max failure risk when no simulation results
    try:
        demand = results.node['demand']
        junctions = wn.junction_name_list
        junction_set = set(junctions)

        # Calculate total demand for each time step
        total_demand = demand[junctions].clip(lower=0).sum(axis=1)

        # Convert failure rate formula coefficients to SI units
        c1 = 0.6858 * (0.0254 ** 3.26)
        c2 = 2.7685 * (0.0254 ** 1.3131)
        c3 = 2.7685 * (0.0254 ** 3.5792)
        c4 = 0.042
        mile_to_m = 1609.34

        # Calculate failure probability for all active pipes
        # Active pipe = Pipe existing in current network and having flow
        total_failure_prob = 0.0
        total_pipe_count = 0
        
        flow = results.link['flowrate'] if 'flowrate' in results.link else None
        
        for name, pipe in wn.pipes():
            # Only consider pipes connecting to junctions
            if pipe.start_node_name not in junction_set and pipe.end_node_name not in junction_set:
                continue
            
            # Check if pipe has flow (active state)
            is_active = True
            if flow is not None:
                avg_flow = abs(flow[name].mean()) if name in flow.columns else 0
                is_active = avg_flow > 1e-10
            
            if not is_active:
                continue  # Skip pipes with no flow (closed)
                
            dia_m = pipe.diameter
            len_m = pipe.length
            
            # Failure rate formula (SI units)
            ri = (c1 / (dia_m ** 3.26) +
                  c2 / (dia_m ** 1.3131) +
                  c3 / (dia_m ** 3.5792) + c4)
            betai = (len_m / mile_to_m) * ri
            pipe_fail_prob = 1 - np.exp(-betai)
            
            total_failure_prob += pipe_fail_prob
            total_pipe_count += 1

        # MRE = Average pipe failure probability
        # Lower values indicate lower failure risk (better)
        if total_pipe_count > 0:
            mre = total_failure_prob / total_pipe_count
        else:
            mre = 1.0  # Return max risk when no pipes
        
        return max(min(mre, 1.0), 0.0)
    except Exception:
        return 1.0  # Return max failure risk on error


def calculate_nr(wn, results):
    """
    Calculate Network Resilience (NR), based on paper formulas (32)(33)
    Only consider junction nodes, excluding Reservoirs and Tanks

    Formula (32): In = Σ(Ui * q_i * mean(Hi - H_min,i)) / (Σ(q_i * H_i) - Σ(q_min,i * H_min,i))
    Formula (33): Ui = Σdj / (Ni * max(d1,...,dNi))
    
    Correction 1: Use average demand instead of minimum demand to avoid zero values causing 0 result
    Correction 2: For single time step (steady state) simulation, use pressure surplus instead of pressure variation
    """
    if results is None:
        return 0.0
    try:
        pressure = results.node['pressure']
        demand = results.node['demand']
        junctions = wn.junction_name_list

        # ===== Calculate Ui: Pipe diameter uniformity factor for each node =====
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

        # ===== Get pressure and demand for junction nodes =====
        press_junc = pressure[junctions].values  # (n_times, n_junctions)
        demand_junc = demand[junctions].values   # (n_times, n_junctions)
        n_times = press_junc.shape[0]

        # Use positive demand for calculation (ignore nodes with negative demand)
        demand_positive = np.clip(demand_junc, 0, None)
        
        # q_avg,i: Average water demand of node i during EPS
        q_avg = np.mean(demand_positive, axis=0)
        
        # ===== Handle single time step (steady state simulation) case =====
        if n_times == 1:
            # For steady state simulation, use pressure surplus as resilience indicator
            # Assume minimum service pressure is 10m, expected pressure is 50th percentile of pressure
            h_current = press_junc[0, :]
            h_min_service = 10.0  # Minimum service pressure
            h_des = np.percentile(h_current[h_current > 0], 50) if np.any(h_current > 0) else 30.0
            
            # Pressure surplus = (Current Pressure - Min Pressure) / (Expected Pressure - Min Pressure)
            surplus = np.clip((h_current - h_min_service) / max(h_des - h_min_service, 1.0), 0, 1)
            
            # Weighted average (weighted by demand)
            if np.sum(q_avg) > 1e-10:
                result = np.sum(Ui_array * q_avg * surplus) / np.sum(q_avg)
            else:
                result = np.mean(surplus)
            
            return max(min(result, 1.0), 0.0)
        
        # ===== Standard EPS case (multi-time step) =====
        # H_min,i: Minimum pressure of node i during EPS
        h_min = np.min(press_junc, axis=0)

        # Calculate numerator: Σ(Ui * q_avg,i * mean(Hi - H_min,i))
        diff_pressure = press_junc - h_min  # Pressure difference matrix
        mean_diff = np.mean(diff_pressure, axis=0)  # Average over time
        numerator = np.sum(Ui_array * q_avg * mean_diff)

        # Calculate denominator: Σ(mean(q_i * H_i))
        qh_product = demand_positive * press_junc
        total_qh = np.sum(np.mean(qh_product, axis=0))
        
        if total_qh < 1e-10:
            return 0.0

        result = numerator / total_qh
        return max(min(result, 1.0), 0.0)
    except Exception:
        return 0.0

