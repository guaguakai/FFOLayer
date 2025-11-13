import torch
import cvxpy as cp
import numpy as np

from cvxpylayers.torch import CvxpyLayer

from ffocp_eq_cone import BLOLayer

def test_soc_blolayer_vs_cvxpy():
    torch.manual_seed(0)

    n = 5
    z = cp.Variable(n)
    q = cp.Parameter(n)

    objective = 0.5 * cp.sum_squares(z) + q.T @ z
    constraints = [cp.SOC(1.0, z)]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dpp()

    cvx_layer = CvxpyLayer(problem, parameters=[q], variables=[z])
    blolayer = BLOLayer(problem, parameters=[q], variables=[z], compute_cos_sim=False)

    q_torch = torch.randn(n, requires_grad=True)

    q_gt = q_torch.clone().detach().requires_grad_(True)
    sol_cvx, = cvx_layer(q_gt)
    loss_cvx = sol_cvx.sum()
    loss_cvx.backward()
    grad_cvx = q_gt.grad.detach().clone()

    print("CvxpyLayer gradient:", grad_cvx)

    batch_size = 1
    q_bl = q_torch.unsqueeze(0).expand(batch_size, n)
    q_bl = q_bl.clone().detach().requires_grad_(True)
    sol_blo, = blolayer(q_bl)
    loss_blo = sol_blo.sum()
    loss_blo.backward()
    grad_blo = q_bl.grad.detach().clone()

    print("BLOLayer gradient:", grad_blo)

    est = grad_blo.reshape(-1)
    gt  = grad_cvx.reshape(-1)

    eps = 1e-12
    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom
    l2_diff = (est - gt).norm()

    print(f"cosine similarity: {cos_sim.item():.6f}")
    print(f"L2 difference:     {l2_diff.item():.6e}")


if __name__ == "__main__":
    test_soc_blolayer_vs_cvxpy()
