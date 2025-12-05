import time
import cvxpy as cp
import numpy as np
import os
from copy import copy

import torch
from cvxtorch import TorchExpression
from cvxpylayers.torch import CvxpyLayer
from cvxpy.constraints.second_order import SOC
import wandb
from utils import to_numpy, to_torch, _dump_cvxpy, n_threads, slice_params_for_batch
from multiprocessing.pool import ThreadPool
from threadpoolctl import threadpool_limits


def _safe_copy(x):
    """Best-effort copy for CVXPY values/duals."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_safe_copy(y) for y in x]
    if hasattr(x, "copy"):
        try:
            return x.copy()
        except Exception:
            pass
    return copy(x)


@torch.no_grad()
def _compare_grads(params_req, grads, ground_truth_grads):
    est_chunks, gt_chunks = [], []
    for p, ge, gg in zip(params_req, grads, ground_truth_grads):
        ge = torch.zeros_like(p) if ge is None else ge.detach()
        gg = torch.zeros_like(p) if gg is None else gg.detach()
        est_chunks.append(ge.reshape(-1))
        gt_chunks.append(gg.reshape(-1))

    est = torch.cat(est_chunks)
    gt  = torch.cat(gt_chunks)

    eps = 1e-12
    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom
    l2_diff = (est - gt).norm()
    return cos_sim, l2_diff


def BLOLayer(
    problem_list,
    parameters_list,
    variables_list,
    alpha: float = 100.0,
    dual_cutoff: float = 1e-3,
    slack_tol: float = 1e-8,
    eps: float = 1e-7,
    compute_cos_sim: bool = False,
):
    """
    MT-capable BLOLayer (same MT form as your attached file):

    - If `problem` is a single cvxpy.Problem (and `parameters`,`variables` are the single lists):
      works for unbatched, or batched *serially*.
    - If `problem` is a list/tuple of Problems (len >= runtime batch size B), and
      `parameters` / `variables` are lists-of-lists with the same length:
      solves each batch element in parallel via ThreadPool.

    For MT usage:
        problem    = [problem_0, ..., problem_{B-1}]
        parameters = [params_0,  ..., params_{B-1}]   # each params_i is list[cp.Parameter]
        variables  = [vars_0,    ..., vars_{B-1}]     # each vars_i   is list[cp.Variable]
    """

    assert len(problem_list) == len(parameters_list) == len(variables_list), \
        "problem/parameters/variables must either be singletons or same-length lists."

    objective_list = []
    eq_funcs_list = []
    scalar_ineq_funcs_list = []
    cone_constraints_list = []
    cone_exprs_list = []

    for prob in problem_list:
        obj = prob.objective
        if isinstance(obj, cp.Minimize):
            objective_expr = obj.expr
        elif isinstance(obj, cp.Maximize):
            objective_expr = -obj.expr
        else:
            objective_expr = getattr(obj, "expr", None)
            if objective_expr is None:
                raise ValueError("Unsupported objective type; expected Minimize/Maximize.")

        eq_funcs = []
        scalar_ineq_funcs = []
        cone_constraints = []
        cone_exprs = []

        for c in prob.constraints:
            if isinstance(c, cp.constraints.zero.Equality):
                eq_funcs.append(c.expr)
            elif isinstance(c, cp.constraints.nonpos.Inequality):
                scalar_ineq_funcs.append(c.expr)
            else:
                cone_constraints.append(c)
                flat_blocks = []
                for arg in c.args:
                    if arg.ndim == 1:
                        flat_blocks.append(arg)
                    else:
                        flat_blocks.append(cp.vec(arg))
                cone_exprs.append(cp.hstack(flat_blocks))

        objective_list.append(objective_expr)
        eq_funcs_list.append(eq_funcs)
        scalar_ineq_funcs_list.append(scalar_ineq_funcs)
        cone_constraints_list.append(cone_constraints)
        cone_exprs_list.append(cone_exprs)

    return _BLOLayer(
        objective_list=objective_list,
        eq_functions_list=eq_funcs_list,
        scalar_ineq_functions_list=scalar_ineq_funcs_list,
        cone_constraints_list=cone_constraints_list,
        cone_exprs_list=cone_exprs_list,
        parameters_list=parameters_list,
        variables_list=variables_list,
        alpha=alpha,
        dual_cutoff=dual_cutoff,
        slack_tol=slack_tol,
        eps=eps,
        _compute_cos_sim=compute_cos_sim,
    )


class _BLOLayer(torch.nn.Module):
    def __init__(
        self,
        objective_list,
        eq_functions_list,
        scalar_ineq_functions_list,
        cone_constraints_list,
        cone_exprs_list,
        parameters_list,
        variables_list,
        alpha,
        dual_cutoff,
        slack_tol,
        eps,
        _compute_cos_sim=False,
    ):
        super().__init__()

        self.num_copies = len(objective_list)
        assert self.num_copies == len(eq_functions_list) == len(scalar_ineq_functions_list) == len(cone_constraints_list) \
               == len(cone_exprs_list) == len(parameters_list) == len(variables_list)

        self.objective_list = objective_list
        self.eq_functions_list = eq_functions_list
        self.scalar_ineq_functions_list = scalar_ineq_functions_list
        self.cone_constraints_list = cone_constraints_list
        self.cone_exprs_list = cone_exprs_list

        self.param_order_list = parameters_list
        self.variables_list = variables_list

        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self._compute_cos_sim = _compute_cos_sim

        # base problems
        self.eq_constraints_list = [[f == 0 for f in eqs] for eqs in self.eq_functions_list]
        self.scalar_ineq_constraints_list = [[g <= 0 for g in gs] for gs in self.scalar_ineq_functions_list]
        self.problem_list = [
            cp.Problem(
                cp.Minimize(self.objective_list[i]),
                self.eq_constraints_list[i]
                + self.scalar_ineq_constraints_list[i]
                + self.cone_constraints_list[i],
            )
            for i in range(self.num_copies)
        ]

        # parameters for perturbation + KKT selection
        self.dvar_params_list = [[cp.Parameter(shape=v.shape) for v in self.variables_list[i]] for i in range(self.num_copies)]
        self.eq_dual_params_list = [[cp.Parameter(shape=f.shape) for f in self.eq_functions_list[i]] for i in range(self.num_copies)]
        self.scalar_ineq_dual_params_list = [
            [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions_list[i]]
            for i in range(self.num_copies)
        ]
        self.scalar_active_mask_params_list = [
            [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions_list[i]]
            for i in range(self.num_copies)
        ]

        self.cone_dual_params_list = [
            [cp.Parameter(shape=g.shape) for g in self.cone_exprs_list[i]]
            for i in range(self.num_copies)
        ]
        self.cone_primal_star_list = [
            [cp.Parameter(shape=g.shape) for g in self.cone_exprs_list[i]]
            for i in range(self.num_copies)
        ]
        self.cone_active_mask_list = [
            [cp.Parameter(nonneg=True) for _ in self.cone_exprs_list[i]]
            for i in range(self.num_copies)
        ]

        # new objective (per copy)
        self.new_objective_list = []
        self.active_eq_constraints_list = []
        self.perturbed_problem_list = []

        for i in range(self.num_copies):
            vars_dvars_product = cp.sum([
                cp.sum(cp.multiply(dv, v))
                for dv, v in zip(self.dvar_params_list[i], self.variables_list[i])
            ])
            scalar_ineq_dual_product = cp.sum([
                cp.sum(cp.multiply(lm, g))
                for lm, g in zip(self.scalar_ineq_dual_params_list[i], self.scalar_ineq_functions_list[i])
            ])
            cone_dual_product = cp.sum([
                cp.sum(cp.multiply(y, g))
                for y, g in zip(self.cone_dual_params_list[i], self.cone_exprs_list[i])
            ])

            new_obj = (1.0 / self.alpha) * vars_dvars_product + self.objective_list[i] + scalar_ineq_dual_product + cone_dual_product
            self.new_objective_list.append(new_obj)

            active_eqs = []
            # scalar: mask * g(x) == 0
            for j, g in enumerate(self.scalar_ineq_functions_list[i]):
                active_eqs.append(cp.multiply(self.scalar_active_mask_params_list[i][j], g) == 0)

            # cone: mask * <y*, g(x) - g*> == 0
            for j, g in enumerate(self.cone_exprs_list[i]):
                dual_param = self.cone_dual_params_list[i][j]
                g_star_param = self.cone_primal_star_list[i][j]
                lin = cp.sum(cp.multiply(dual_param, g - g_star_param))
                active_eqs.append(self.cone_active_mask_list[i][j] * lin == 0)

            self.active_eq_constraints_list.append(active_eqs)

            self.perturbed_problem_list.append(
                cp.Problem(
                    cp.Minimize(self.new_objective_list[i]),
                    self.eq_constraints_list[i] + self.active_eq_constraints_list[i],
                )
            )

        # phi torch expression (compile once; shapes match across copies)
        self.phi_torch = TorchExpression(
            self.objective_list[0]
            + cp.sum([cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params_list[0], self.eq_functions_list[0])])
            + cp.sum([cp.sum(cp.multiply(du, g)) for du, g in zip(self.scalar_ineq_dual_params_list[0], self.scalar_ineq_functions_list[0])])
            + cp.sum([cp.sum(cp.multiply(du, g)) for du, g in zip(self.cone_dual_params_list[0], self.cone_exprs_list[0])]),
            provided_vars_list=[
                *self.variables_list[0],
                *self.param_order_list[0],
                *self.eq_dual_params_list[0],
                *self.scalar_ineq_dual_params_list[0],
                *self.cone_dual_params_list[0],
            ],
        ).torch_expression

        print("new_objective DPP? (copy0)", self.new_objective_list[0].is_dcp(dpp=True))
        print("perturbed_problem is_dcp (copy0):", self.perturbed_problem_list[0].is_dcp())
        print("perturbed_problem is_dpp (copy0):", self.perturbed_problem_list[0].is_dpp())

        self.forward_setup_time = 0
        self.forward_solve_time = 0
        self.backward_setup_time = 0
        self.backward_solve_time = 0

    def forward(self, *params, solver_args={}):
        if solver_args is None:
            solver_args = {}
        default_solver_args = {"ignore_dpp": True}
        if solver_args.get("solver") == cp.SCS:
            default_solver_args = dict(
                solver=cp.SCS,
                warm_start=False,
                ignore_dpp=True,
                max_iters=2500,
                eps=self.eps,
            )
        solver_args = {**default_solver_args, **solver_args}

        info = {}
        f = _BLOLayerFn(self, solver_args=solver_args, _compute_cos_sim=self._compute_cos_sim, info=info)
        sol = f(*params)
        self.info = info
        return sol


def _BLOLayerFn(blolayer, solver_args, _compute_cos_sim, info):
    class _BLOLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.solver_args = solver_args

            # infer batch sizes from copy0 parameter shapes
            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, blolayer.param_order_list[0])):
                if p.dtype != ctx.dtype:
                    raise ValueError(
                        "Two or more parameters have different dtypes. Expected parameter %d to have dtype %s but got %s."
                        % (i, str(ctx.dtype), str(p.dtype))
                    )
                if p.device != ctx.device:
                    raise ValueError(
                        "Two or more parameters are on different devices. Expected parameter %d to be on %s but got %s."
                        % (i, str(ctx.device), str(p.device))
                    )

                if p.ndimension() == q.ndim:
                    batch_size = 0
                elif p.ndimension() == q.ndim + 1:
                    batch_size = p.size(0)
                    if batch_size == 0:
                        raise ValueError(f"The batch dimension for parameter {i} is zero but should be non-zero.")
                else:
                    raise ValueError(
                        f"Invalid parameter size passed in. Expected parameter {i} to have {q.ndim} or {q.ndim+1} dims but got {p.ndimension()}."
                    )

                ctx.batch_sizes.append(batch_size)
                p_shape = p.shape if batch_size == 0 else p.shape[1:]
                if not np.all(p_shape == blolayer.param_order_list[0][i].shape):
                    raise ValueError(f"Inconsistent parameter shapes passed in for param {i}.")

            ctx.batch_sizes = np.array(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)
            if ctx.batch:
                nonzero = ctx.batch_sizes[ctx.batch_sizes > 0]
                ctx.batch_size = int(nonzero[0])
                if np.any(nonzero != ctx.batch_size):
                    raise ValueError(f"Inconsistent batch sizes: {ctx.batch_sizes}.")
            else:
                ctx.batch_size = 1

            B = ctx.batch_size
            ctx.params = params

            params_numpy = [to_numpy(p) for p in params]
            ctx.params_numpy = params_numpy

            # allocate buffers off copy0 shapes
            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables_list[0]]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in blolayer.eq_functions_list[0]]
            scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions_list[0]]
            scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions_list[0]]
            cone_primal_vals = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.cone_exprs_list[0]]
            cone_dual_vals = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.cone_exprs_list[0]]

            def _flatten_cone_dual(dv_raw, g_shape):
                if isinstance(dv_raw, (list, tuple)):
                    flat_chunks = []
                    for part in dv_raw:
                        part_arr = np.asarray(part, dtype=float).reshape(-1)
                        flat_chunks.append(part_arr)
                    dv_flat = np.concatenate(flat_chunks, axis=0)
                else:
                    dv_flat = np.asarray(dv_raw, dtype=float).reshape(-1)

                n_expected = int(np.prod(g_shape))
                if dv_flat.size != n_expected:
                    raise RuntimeError(f"cone dual size mismatch: got {dv_flat.size}, expected {n_expected}")
                return dv_flat.reshape(g_shape)

            def _solve_one_forward(i):
                slot = i  # MT form: one copy per batch element
                if ctx.batch:
                    params_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_i = params_numpy

                for p_val, param_obj in zip(params_i, blolayer.param_order_list[slot]):
                    param_obj.value = p_val

                try:
                    blolayer.problem_list[slot].solve(**ctx.solver_args)
                except Exception:
                    print("Forward pass SCS failed, using OSQP")
                    exit()
                    blolayer.problem_list[slot].solve(solver=cp.OSQP, warm_start=False, verbose=False)

                st = blolayer.problem_list[slot].status
                if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"Forward problem status = {st}")

                sol_vals = [v.value for v in blolayer.variables_list[slot]]
                eq_vals = [c.dual_value for c in blolayer.eq_constraints_list[slot]]
                sinq_vals = [c.dual_value for c in blolayer.scalar_ineq_constraints_list[slot]]
                slack_vals = [np.maximum(-expr.value, 0.0) for expr in blolayer.scalar_ineq_functions_list[slot]]
                cone_g_vals = [np.asarray(expr.value, dtype=float) for expr in blolayer.cone_exprs_list[slot]]
                cone_y_vals = [
                    _flatten_cone_dual(c.dual_value, blolayer.cone_exprs_list[slot][j].shape)
                    for j, c in enumerate(blolayer.cone_constraints_list[slot])
                ]

                setup_time = getattr(blolayer.problem_list[slot], "compilation_time", 0.0) or 0.0
                solve_time = getattr(getattr(blolayer.problem_list[slot], "solver_stats", None), "solve_time", 0.0) or 0.0
                return sol_vals, eq_vals, sinq_vals, slack_vals, cone_g_vals, cone_y_vals, setup_time, solve_time

            use_mt = (ThreadPool is not None) and (B > 1) and (blolayer.num_copies >= B)
            if use_mt:
                with threadpool_limits(limits=1):
                    pool = ThreadPool(processes=min(B, n_threads))
                    try:
                        results = pool.map(_solve_one_forward, range(B))
                    finally:
                        pool.close()
            else:
                # serial fallback (still supports B>1 if you only provided 1 copy)
                results = []
                for i in range(B):
                    slot = 0 if blolayer.num_copies == 1 else i
                    if ctx.batch:
                        params_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                    else:
                        params_i = params_numpy
                    for p_val, param_obj in zip(params_i, blolayer.param_order_list[slot]):
                        param_obj.value = p_val
                    try:
                        blolayer.problem_list[slot].solve(**ctx.solver_args)
                    except Exception:
                        blolayer.problem_list[slot].solve(solver=cp.OSQP, warm_start=False, verbose=False)

                    st = blolayer.problem_list[slot].status
                    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        raise RuntimeError(f"Forward problem status = {st}")

                    sol_vals = [v.value for v in blolayer.variables_list[slot]]
                    eq_vals = [c.dual_value for c in blolayer.eq_constraints_list[slot]]
                    sinq_vals = [c.dual_value for c in blolayer.scalar_ineq_constraints_list[slot]]
                    slack_vals = [np.maximum(-expr.value, 0.0) for expr in blolayer.scalar_ineq_functions_list[slot]]
                    cone_g_vals = [np.asarray(expr.value, dtype=float) for expr in blolayer.cone_exprs_list[slot]]
                    cone_y_vals = [
                        _flatten_cone_dual(c.dual_value, blolayer.cone_exprs_list[slot][j].shape)
                        for j, c in enumerate(blolayer.cone_constraints_list[slot])
                    ]
                    setup_time = getattr(blolayer.problem_list[slot], "compilation_time", 0.0) or 0.0
                    solve_time = getattr(getattr(blolayer.problem_list[slot], "solver_stats", None), "solve_time", 0.0) or 0.0
                    results.append((sol_vals, eq_vals, sinq_vals, slack_vals, cone_g_vals, cone_y_vals, setup_time, solve_time))

            avg_setup_time, avg_solve_time = 0.0, 0.0
            for i, (sol_i, eq_i, sinq_i, slack_i, cone_g_i, cone_y_i, setup_t, solve_t) in enumerate(results):
                for v_id in range(len(sol_numpy)):
                    sol_numpy[v_id][i, ...] = sol_i[v_id]
                for c_id in range(len(eq_dual)):
                    eq_dual[c_id][i, ...] = eq_i[c_id]
                for j in range(len(scalar_ineq_dual)):
                    scalar_ineq_dual[j][i, ...] = sinq_i[j]
                    scalar_ineq_slack[j][i, ...] = slack_i[j]
                for j in range(len(cone_primal_vals)):
                    cone_primal_vals[j][i, ...] = cone_g_i[j]
                    cone_dual_vals[j][i, ...] = cone_y_i[j]
                avg_setup_time += setup_t
                avg_solve_time += solve_t

            info["forward_setup_time"] = avg_setup_time / max(1, B)
            info["forward_solve_time"] = avg_solve_time / max(1, B)

            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.cone_primal_vals = cone_primal_vals
            ctx.cone_dual_vals = cone_dual_vals
            ctx.blolayer = blolayer

            # warm starts (values only)
            ctx._warm_vars_list = [[_safe_copy(sol_numpy[k][i, ...]) for k in range(len(sol_numpy))] for i in range(B)]

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]
            return tuple(sol_torch)

        @staticmethod
        def backward(ctx, *dvars):
            blolayer = ctx.blolayer
            B = ctx.batch_size

            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            sol_numpy = ctx.sol_numpy
            eq_dual = ctx.eq_dual
            scalar_ineq_dual = ctx.scalar_ineq_dual
            scalar_ineq_slack = ctx.scalar_ineq_slack
            cone_primal_vals = ctx.cone_primal_vals
            cone_dual_vals = ctx.cone_dual_vals

            num_scalar_ineq = len(blolayer.scalar_ineq_functions_list[0])
            num_cones = len(blolayer.cone_exprs_list[0])

            # prepare grad inputs
            params_req, req_grad_mask = [], []
            for p in ctx.params:
                q = p.detach().clone()
                req_grad = bool(p.requires_grad)
                q.requires_grad_(req_grad)
                params_req.append(q)
                req_grad_mask.append(req_grad)

            if _compute_cos_sim:
                cvxpylayer_problem = cp.Problem(
                    cp.Minimize(blolayer.objective_list[0]),
                    constraints=(
                        blolayer.eq_constraints_list[0]
                        + blolayer.scalar_ineq_constraints_list[0]
                        + blolayer.cone_constraints_list[0]
                    ),
                )
                _cvx_layer = CvxpyLayer(
                    cvxpylayer_problem,
                    parameters=blolayer.param_order_list[0],
                    variables=blolayer.variables_list[0],
                )
                with torch.enable_grad():
                    sol_tensors = _cvx_layer(*params_req)
                if not isinstance(sol_tensors, (tuple, list)):
                    sol_tensors = (sol_tensors,)
                grad_outputs = [torch.zeros_like(out) if gv is None else gv for out, gv in zip(sol_tensors, dvars)]
                inputs_for_grad = tuple(q for q, need in zip(params_req, req_grad_mask) if need)
                ground_truth_grads = torch.autograd.grad(
                    outputs=tuple(sol_tensors),
                    inputs=inputs_for_grad,
                    grad_outputs=tuple(grad_outputs),
                    allow_unused=True,
                    retain_graph=False,
                )

            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(sol_numpy))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(eq_dual))]
            new_scalar_ineq_dual = [np.empty_like(scalar_ineq_dual[j]) for j in range(num_scalar_ineq)]
            new_cone_dual_vals = [np.empty_like(cone_dual_vals[j]) for j in range(num_cones)]

            def _solve_one_backward(i):
                slot = i if blolayer.num_copies >= B else 0

                # params for this batch element
                if ctx.batch:
                    params_i = [p[i] if bs > 0 else p for p, bs in zip(ctx.params_numpy, ctx.batch_sizes)]
                else:
                    params_i = ctx.params_numpy

                for p_val, param_obj in zip(params_i, blolayer.param_order_list[slot]):
                    param_obj.value = p_val

                # dL/dv parameters + warm-start primal vars
                for j, v in enumerate(blolayer.variables_list[slot]):
                    blolayer.dvar_params_list[slot][j].value = dvars_numpy[j][i]
                    v.value = ctx._warm_vars_list[i][j]

                # forward eq dual as params
                for j in range(len(blolayer.eq_functions_list[slot])):
                    blolayer.eq_dual_params_list[slot][j].value = eq_dual[j][i]

                # scalar ineq duals + active set masks
                y_dim = dvars_numpy[0].shape[1] if dvars_numpy[0].ndim > 1 else dvars_numpy[0].shape[0]
                num_eq = 0 if len(eq_dual) == 0 else (eq_dual[0].shape[1] if eq_dual[0].ndim > 1 else eq_dual[0].shape[0])

                for j in range(num_scalar_ineq):
                    lam = scalar_ineq_dual[j][i]
                    lam = np.where(lam < -1e-6, lam, np.maximum(lam, 0.0))
                    blolayer.scalar_ineq_dual_params_list[slot][j].value = lam

                    sl = scalar_ineq_slack[j][i]
                    mask = (sl <= blolayer.slack_tol).astype(np.float64)
                    if mask.sum() > max(1, y_dim - num_eq):
                        k = int(max(1, y_dim - num_eq))
                        lam_flat = lam.reshape(-1)
                        idx = np.argpartition(lam_flat, -k)[-k:]
                        mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                        mask_flat[idx] = 1.0
                        mask = mask_flat.reshape(lam.shape)
                    blolayer.scalar_active_mask_params_list[slot][j].value = mask

                # cone params + active mask
                for j in range(num_cones):
                    g_star = cone_primal_vals[j][i, ...]
                    y_star = cone_dual_vals[j][i, ...]
                    blolayer.cone_primal_star_list[slot][j].value = g_star
                    blolayer.cone_dual_params_list[slot][j].value = y_star

                    y_norm = float(np.linalg.norm(y_star.reshape(-1), ord=np.inf))
                    blolayer.cone_active_mask_list[slot][j].value = 1.0 if y_norm > blolayer.dual_cutoff else 0.0

                backward_solver_args = dict(ctx.solver_args)
                backward_solver_args["warm_start"] = False
                backward_solver_args["verbose"] = False
                # if ctx.solver_args.get("solver") != cp.MOSEK:
                #     backward_solver_args["warm_start"] = True

                try:
                    blolayer.perturbed_problem_list[slot].solve(**backward_solver_args)
                except Exception:
                    print("Backward pass SCS failed, using OSQP")
                    exit()
                    blolayer.perturbed_problem_list[slot].solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4,
                                                                warm_start=True, verbose=False)

                st = blolayer.perturbed_problem_list[slot].status
                if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"New bilevel problem status = {st}")

                new_sol_vals = [v.value for v in blolayer.variables_list[slot]]
                new_eq_vals = [c.dual_value for c in blolayer.eq_constraints_list[slot]]
                new_active_vals = [c.dual_value for c in blolayer.active_eq_constraints_list[slot]]

                setup_time = getattr(blolayer.perturbed_problem_list[slot], "compilation_time", 0.0) or 0.0
                solve_time = getattr(getattr(blolayer.perturbed_problem_list[slot], "solver_stats", None), "solve_time", 0.0) or 0.0
                return new_sol_vals, new_eq_vals, new_active_vals, setup_time, solve_time

            use_mt = (ThreadPool is not None) and (B > 1) and (blolayer.num_copies >= B)
            if use_mt:
                with threadpool_limits(limits=1):
                    pool = ThreadPool(processes=min(B, n_threads))
                    try:
                        results = pool.map(_solve_one_backward, range(B))
                    finally:
                        pool.close()
            else:
                results = [_solve_one_backward(i) for i in range(B)]

            avg_setup, avg_solve = 0.0, 0.0
            for i, (new_sol_vals, new_eq_vals, new_active_vals, setup_t, solve_t) in enumerate(results):
                for j in range(len(new_sol_lagrangian)):
                    new_sol_lagrangian[j][i, ...] = new_sol_vals[j]
                for j in range(len(new_eq_dual)):
                    new_eq_dual[j][i, ...] = new_eq_vals[j]

                # active duals: first scalar ineq, then cones
                for j in range(num_scalar_ineq):
                    new_scalar_ineq_dual[j][i, ...] = new_active_vals[j]

                for j in range(num_cones):
                    idx = num_scalar_ineq + j
                    lam_cone = float(np.asarray(new_active_vals[idx]).reshape(()))
                    y_old = cone_dual_vals[j][i, ...]
                    new_cone_dual_vals[j][i, ...] = (1.0 + lam_cone) * y_old

                avg_setup += setup_t
                avg_solve += solve_t

            info["backward_setup_time"] = avg_setup / max(1, B)
            info["backward_solve_time"] = avg_solve / max(1, B)

            # torch-side phi evaluation
            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
            new_eq_dual_torch  = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_torch  = [to_torch(v, ctx.dtype, ctx.device) for v in eq_dual]

            old_scalar_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in scalar_ineq_dual]
            new_scalar_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_scalar_ineq_dual]

            old_cone_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in cone_dual_vals]
            new_cone_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_cone_dual_vals]

            # rebuild leaf params for autograd
            params_req2 = []
            for p, need in zip(ctx.params, req_grad_mask):
                q = p.detach().clone()
                if need:
                    q.requires_grad_(True)
                params_req2.append(q)

            if ctx.device != torch.device("cpu"):
                torch.set_default_device(torch.device(ctx.device))

            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(sol_numpy))]
                    params_i = slice_params_for_batch(params_req2, ctx.batch_sizes, i)

                    new_eq_dual_i  = [d[i] for d in new_eq_dual_torch]
                    old_eq_dual_i  = [d[i] for d in old_eq_dual_torch]

                    new_scalar_dual_i = [d[i] for d in new_scalar_ineq_dual_torch]
                    old_scalar_dual_i = [d[i] for d in old_scalar_ineq_dual_torch]

                    new_cone_dual_i = [d[i] for d in new_cone_dual_torch]
                    old_cone_dual_i = [d[i] for d in old_cone_dual_torch]

                    phi_new_i = blolayer.phi_torch(*vars_new_i, *params_i, *new_eq_dual_i, *new_scalar_dual_i, *new_cone_dual_i)
                    phi_old_i = blolayer.phi_torch(*vars_old_i, *params_i, *old_eq_dual_i, *old_scalar_dual_i, *old_cone_dual_i)
                    loss = loss + (phi_new_i - phi_old_i)

                loss = blolayer.alpha * loss

            grads_req = torch.autograd.grad(
                outputs=loss,
                inputs=[q for q, need in zip(params_req2, req_grad_mask) if need],
                allow_unused=True,
                retain_graph=False,
            )

            grads = []
            it = iter(grads_req)
            for need in req_grad_mask:
                grads.append(next(it) if need else None)

            if _compute_cos_sim:
                cos_sim, l2_norm = _compare_grads(
                    [q for q, need in zip(params_req2, req_grad_mask) if need],
                    list(grads_req),
                    list(ground_truth_grads),
                )
                wandb.log({"cos_sim": cos_sim, "l2_norm": l2_norm})

            return tuple(grads)

    return _BLOLayerFnFn.apply
