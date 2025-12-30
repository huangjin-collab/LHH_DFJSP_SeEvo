#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

# ---------------- 如果有新版表达式优先用 ----------------
try:
    from gpt import get_combined_expression_v2 as get_combined_expression
except ImportError:                       # fallback
    from gpt import get_combined_expression


# ======================================================================
#                           1.  Job
# ======================================================================
class Job:
    def __init__(
        self,
        job_index: int,
        num_op: int,
        plan_pts: np.ndarray,     # shape [op, fle]
        eachope_fle: np.ndarray,
        job_machs: np.ndarray,
        kappa: float = 0.20,
    ):
        self.job_index   = job_index
        self.num_op      = num_op
        self.plan_pts    = plan_pts
        self.eachope_fle = eachope_fle
        self.job_machs   = job_machs

        # -------- 名义 / 实际 pt --------
        self.nominal_pts = np.array(
            [plan_pts[i].sum() / eachope_fle[i] for i in range(num_op)],
            dtype=float,
        )
        self.job_actual_pts  = self.nominal_pts.copy()
        self.real_pts        = np.zeros_like(plan_pts)

        self.job_actual_machs = np.zeros(num_op, dtype=int)
        self.sel_cand_idx     = np.full(num_op, -1, dtype=int)

        # -------- 状态 --------
        self.op = 0
        self.ope_machproc = np.zeros((4, num_op))   # 0就绪1开始2结束3已加工
        self.ope_machproc[0][0] = 1                 # 首工序就绪
        self.done = False

        # -------- EMA --------
        self.kappa = kappa
        self.pt_dev_ema: float = 0.0

        self._update()

    # -------- 统计量刷新 --------
    def _update(self):
        self.jtwk = self.job_actual_pts.sum()
        if self.done:
            self.cur_op_pt = self.jwkr = self.jrm = self.jso = 0.0
            return
        self.cur_op_pt = self.job_actual_pts[self.op]
        self.jwkr = self.job_actual_pts[self.op:].sum()
        if self.op < self.num_op - 1:
            self.jrm = self.job_actual_pts[self.op + 1:].sum()
            self.jso = self.job_actual_pts[self.op + 1]
        else:
            self.jrm = self.jso = 0.0

    # -------- EMA 更新 --------
    def update_ema(self, realised: float, nominal: float):
        dev = realised / nominal - 1.0
        self.pt_dev_ema = self.kappa * dev + (1 - self.kappa) * self.pt_dev_ema


# ======================================================================
#                           2.  Machine
# ======================================================================
class Machine:
    def __init__(self, idx: int):
        self.idx = idx                             # 0‑based
        self.buffer: List[Job] = []
        self.processing = np.zeros(3)              # [job+1, op+1, processed]
        self.proc_time = 0.0
        self.available = True
        self.uti = 0.0
        self.buffer_completion_time = 0.0

    # ---------- 利用率 ----------
    def util(self, now: float) -> float:
        self.uti = 0.0 if now == 0 else round(self.proc_time / now, 3)
        return self.uti

    # ---------- 缓冲区 + 正在加工 剩余 ----------
    def get_buffer_completion_time(self, Jobs: List[Job]) -> float:
        t = 0.0
        if self.processing[0] != 0:
            ji = Jobs[int(self.processing[0]) - 1]
            t += ji.job_actual_pts[int(self.processing[1]) - 1] - self.processing[2]
        for ji in self.buffer:
            t += ji.job_actual_pts[ji.op]
        self.buffer_completion_time = t
        return t

# ======================================================================
#                     3.  主环境  RealTimeFJSPDual
# ======================================================================
class RealTimeFJSPDual:
    def __init__(self, plan: dict, real: dict, kappa: float = 0.20):
        # ---------- 静态 ----------
        self._plan_orig = plan
        self._real_orig = real
        self.plan = copy.deepcopy(plan)
        self.real = copy.deepcopy(real)
        self.kappa = kappa

        self.mach_num  = plan["mach_num"]
        self.order_num = plan["order_num"]
        self.fault_num = plan["mach_fault"]["fault_num"]

        # ---------- 订单 1 ----------
        self._load_order(plan["order_1"], real["order_1"], first=True)

        # ---------- 故障 ----------
        self.fault_idx  = plan["mach_fault"]["fault_index"].copy()
        self.fault_t0   = plan["mach_fault"]["fault_start_time"].copy()
        self.fault_t1   = plan["mach_fault"]["fault_end_time"].copy()
        self.fault_t0 = np.asarray(plan["mach_fault"]["fault_start_time"], dtype=float)
        self.fault_t1 = np.asarray(plan["mach_fault"]["fault_end_time"],   dtype=float)
        self.true_fault_t0 = self.fault_t0.copy()
        self.true_fault_t1 = np.full_like(self.fault_t1, np.nan, dtype=float)
        self.over_fault_num = 0

        # ---------- 运行时 ----------
        self.Machs = [Machine(i) for i in range(self.mach_num)]
        self.current_t = 0.0
        self.over_order_num = 1     # 已进入系统的订单数

        self.makespan = self.get_makespan()

    # ------------------------------------------------------------------
    #  把一个订单数据拼到全局数组
    # ------------------------------------------------------------------
    def _load_order(self, plan_ord: dict, real_ord: dict, *, first=False):
        if first:
            self.eachjob_openum = plan_ord["eachjob_openum"].copy()
            self.eachope_mach   = plan_ord["eachope_mach"].copy()
            self.eachope_fle    = plan_ord["eachope_fle"].copy()
            self.plan_time      = plan_ord["eachope_time"].copy()
            self.real_time      = real_ord["eachope_time"].copy()
            self.Jobs, self.pool = [], []
            self.job_num = 0
        else:
            self.eachjob_openum = np.concatenate((self.eachjob_openum, plan_ord["eachjob_openum"]), axis=0)
            self.eachope_mach   = np.concatenate((self.eachope_mach,   plan_ord["eachope_mach"]),  axis=0)
            self.eachope_fle    = np.concatenate((self.eachope_fle,    plan_ord["eachope_fle"]),   axis=0)
            self.plan_time      = np.concatenate((self.plan_time,     plan_ord["eachope_time"]),   axis=0)
            self.real_time      = np.concatenate((self.real_time,     real_ord["eachope_time"]),   axis=0)

        n_new = plan_ord["job_num"]
        for i in range(n_new):
            idx = self.job_num + i
            ji = Job(
                job_index   = idx,
                num_op      = self.eachjob_openum[idx],
                plan_pts    = self.plan_time[idx],
                eachope_fle = self.eachope_fle[idx],
                job_machs   = self.eachope_mach[idx],
                kappa       = self.kappa,
            )
            self.Jobs.append(ji)
            self.pool.append(ji)
        self.job_num = len(self.Jobs)

    # ------------------------------------------------------------------
    #  ========== 特征构造 ==========
    # ------------------------------------------------------------------
    def _gen_job_feats(self):
        cur = np.array([j.cur_op_pt  for j in self.pool], dtype=np.float32)
        wkr = np.array([j.jwkr       for j in self.pool], dtype=np.float32)
        rm  = np.array([j.jrm        for j in self.pool], dtype=np.float32)
        so  = np.array([j.jso        for j in self.pool], dtype=np.float32)
        twk = np.array([j.jtwk       for j in self.pool], dtype=np.float32)
        ema = np.array([j.pt_dev_ema for j in self.pool], dtype=np.float32)
        return cur, wkr, rm, so, twk, ema

    def _gen_mach_feats(self, Ji: Job):
        j, op = Ji.job_index, Ji.op
        cand  = int(self.eachope_fle[j][op])
        avail = [m.idx + 1 for m in self.Machs]

        idxs = [k for k in range(cand)
                if self.eachope_mach[j][op][k] in avail]

        mach_ids = np.array([self.eachope_mach[j][op][k] for k in idxs],
                            dtype=np.int16)
        nom_pts  = np.array([self.plan_time[j][op][k] for k in idxs],
                            dtype=np.float32)
        uti      = np.array([self.Machs[m-1].util(self.current_t)
                             for m in mach_ids],
                            dtype=np.float32)
        return mach_ids, nom_pts, uti

    # ------------------------------------------------------------------
    #  ========== 外层一步 ==========
    # ------------------------------------------------------------------
    def step(self) -> bool:
        # ---------- 1) 选 Job ----------
        cur, wkr, rm, so, twk, ema = self._gen_job_feats()
        # print(ema)
        score = get_combined_expression(cur, wkr, rm, so, twk, ema)
        # print(score)
        job_idx_pool = int(np.argmin(score))
        Ji = self.pool.pop(job_idx_pool)

        # ---------- 2) 选 Machine ----------
        mach_ids, nom_vec, uti_vec = self._gen_mach_feats(Ji)
        sel_mach = int(mach_ids[np.argmin(nom_vec)])     # 示例：名义 pt 最小

        # 记录所选柔性下标
        Ji.sel_cand_idx[Ji.op] = np.where(
            self.eachope_mach[Ji.job_index][Ji.op] == sel_mach)[0][0]
        Ji.job_actual_machs[Ji.op] = sel_mach

        # 放入缓冲
        self.Machs[sel_mach-1].buffer.append(Ji)

        # ---------- 3) 时间推进 ----------
        done = self._forward()
        self.makespan = self.get_makespan()
        return done

    # ------------------------------------------------------------------
    #  ========== 时间推进核心 ==========
    # ------------------------------------------------------------------
    def _forward(self) -> bool:
        if self.pool:
            return False

        while True:
            statetran = np.full(self.mach_num, np.inf)

            # ---- 遍历机器 ----
            for mi in self.Machs:
                if not mi.available:
                    continue

                if mi.processing[0] != 0:     # 正在加工
                    ji = self.Jobs[int(mi.processing[0]) - 1]
                    statetran[mi.idx] = (
                        ji.job_actual_pts[int(mi.processing[1]) - 1] - mi.processing[2]
                    )
                elif mi.buffer:              # 空闲但有缓冲
                    ji = mi.buffer.pop(0)
                    ji.ope_machproc[1][ji.op] = self.current_t

                    cand = ji.sel_cand_idx[ji.op]
                    real_pt = self.real_time[ji.job_index][ji.op][cand]
                    nominal = ji.nominal_pts[ji.op]

                    ji.job_actual_pts[ji.op] = real_pt
                    ji.update_ema(real_pt, nominal)

                    statetran[mi.idx] = real_pt
                    mi.processing[:] = [ji.job_index + 1, ji.op + 1, 0.0]

            # ---- 动态事件 & 全局最短 ----
            dt = np.min(statetran)
            dt = self._dynamic(dt)
            self.current_t += dt

            # ---- 同步推进 ----
            for mi in self.Machs:
                if not mi.available or mi.processing[0] == 0:
                    continue
                ji = self.Jobs[int(mi.processing[0]) - 1]
                op = int(mi.processing[1]) - 1

                ji.ope_machproc[3][op] += dt
                mi.processing[2]       += dt
                mi.proc_time           += dt

                # 工序完成？
                if ji.ope_machproc[3][op] >= ji.job_actual_pts[op]:
                    ji.ope_machproc[2][op] = ji.ope_machproc[1][op] + ji.job_actual_pts[op]
                    mi.processing[:] = 0.0
                    if op < ji.num_op - 1:
                        ji.op += 1
                        ji.ope_machproc[0][ji.op] = 1
                        ji._update()
                        self.pool.append(ji)
                    else:                      # job 完成
                        ji.done = True
                        ji._update()

            # ---- 结束判断 ----
            if all(ji.done for ji in self.Jobs):
                return True
            if self.pool:
                return False

    # ------------------------------------------------------------------
    #  ========== 动态事件（订单 & 故障） ==========
    # ------------------------------------------------------------------
    def _dynamic(self, dt: float) -> float:
        inf = np.inf
        next_t = inf

        # ------ 订单 ------
        if (self.order_num - self.over_order_num > 0 and
            self.current_t + dt >
            self.plan[f"order_{self.over_order_num+1}"]["arr_time"]):
            order_t = self.plan[f"order_{self.over_order_num+1}"]["arr_time"]
            next_t = min(next_t, order_t)
        else:
            order_t = inf

        # ------ 故障开始 ------
        if (self.fault_num - self.over_fault_num > 0 and
            not np.isnan(self.true_fault_t0).all() and
            np.nanmin(self.true_fault_t0) < self.current_t + dt):
            fs_t = np.nanmin(self.true_fault_t0)
            next_t = min(next_t, fs_t)
        else:
            fs_t = inf

        # ------ 故障结束 ------
        if (not np.isnan(self.true_fault_t1).all() and
            np.nanmin(self.true_fault_t1) < self.current_t + dt):
            fe_t = np.nanmin(self.true_fault_t1)
            next_t = min(next_t, fe_t)
        else:
            fe_t = inf

        # ========== 无事件 ==========
        if next_t is inf:
            return dt

        # ========== 订单到达 ==========
        if next_t == order_t:
            n = self.over_order_num + 1
            self.over_order_num = n
            self._load_order(self.plan[f"order_{n}"], self.real[f"order_{n}"], first=False)
            return order_t - self.current_t

        # ========== 故障开始 ==========
        if next_t == fs_t:
            idx = int(np.nanargmin(self.true_fault_t0))
            m_idx = int(self.fault_idx[idx]-1)
            self.Machs[m_idx].available = False
            self.true_fault_t0[idx] = np.nan
            self.true_fault_t1[idx] = self.fault_t1[idx]
            self.over_fault_num += 1

            # 撤掉正在加工工序 & 缓冲
            if self.Machs[m_idx].processing[0]:
                ji = self.Jobs[int(self.Machs[m_idx].processing[0]) - 1]
                op = ji.op
                ji.ope_machproc[1][op] = ji.ope_machproc[3][op] = 0.0
                ji.job_actual_pts[op]  = ji.nominal_pts[op]
                ji._update()
                self.pool.append(ji)
                self.Machs[m_idx].processing[:] = 0.0
            self.pool.extend(self.Machs[m_idx].buffer)
            self.Machs[m_idx].buffer.clear()
            return fs_t - self.current_t

        # ========== 故障结束 ==========
        if next_t == fe_t:
            idx = int(np.nanargmin(self.true_fault_t1))
            m_idx = int(self.fault_idx[idx]-1)
            self.Machs[m_idx].available = True
            self.true_fault_t1[idx] = np.nan
            return fe_t - self.current_t

        return dt   # 理论不会到这

    # ------------------------------------------------------------------
    #  ========== 计算 makespan ==========
    # ------------------------------------------------------------------
    def get_makespan(self) -> float:
        ms = 0.0
        for mi in self.Machs:
            est = mi.get_buffer_completion_time(self.Jobs) + self.current_t
            ms = max(ms, est)
        return ms

    # ------------------------------------------------------------------
    #  ========== 重置 ==========
    # ------------------------------------------------------------------
    def reset(self):
        self.__init__(self._plan_orig, self._real_orig, self.kappa)

