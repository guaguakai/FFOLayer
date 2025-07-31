import unittest

import cvxpy as cp
import numpy as np
import numpy.random as npr
import torch
from torch.autograd import grad

from bilevel_layer import BilevelLayer
from utils import forward_numpy, backward_numpy
import diffcp

torch.set_default_dtype(torch.double)


def set_seed(x):
    npr.seed(x)
    torch.manual_seed(x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class TestCvxpyLayer(unittest.TestCase):

    def test_example(self):
        set_seed(243)
        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        bilevel_layer = BilevelLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        solution, = bilevel_layer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()

        print(solution)

    def test_qp(self):
        # torch.manual_seed(SEED)
        # Create a sample optimization problem
        n = 100
        n_eq_constraints   = 0
        n_ineq_constraints = 50
        Q = torch.eye(n)
        q = torch.rand(n)
        q.requires_grad_(True)
        A = torch.empty(n_eq_constraints, n)
        b = torch.empty(n_eq_constraints)
        G = torch.randn(n_ineq_constraints, n)
        h = torch.randn(n_ineq_constraints)

        # Create an instance of the CVXPY layer
        Q_cp = cp.Parameter((n, n), PSD=True)
        q_cp = cp.Parameter(n)
        # A_cp = cp.Parameter((n_eq_constraints, n))
        # b_cp = cp.Parameter(n_eq_constraints)
        G_cp = cp.Parameter((n_ineq_constraints, n))
        h_cp = cp.Parameter(n_ineq_constraints)
        z_cp = cp.Variable(n)

        # Define the objective and constraints for CVXPY
        objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
        constraints = [G_cp @ z_cp <= h_cp]
        # if n_eq_constraints > 0:
        #     constraints.append(A_cp @ z_cp == b_cp)

        # Create the CVXPY problem
        print('calling cvxpylaer...')
        problem = cp.Problem(cp.Minimize(objective_fn), constraints)
        # print("problem is dpp:", problem.is_dpp())
        # assert(problem.is_dpp())

        # Cvxpylayer
        layer = BilevelLayer(problem, parameters=[Q_cp, q_cp, G_cp, h_cp], variables=[z_cp])
        sol = layer(Q, q, G, h)
        z_sol = sol[0]

        loss = torch.sum(z_sol)
        loss.backward()

        print(z_sol)
        
        # print('ffoqp_grad', ffoqp_grad)
        # print('cvxpylayer_grad', cvxpylayer_grad)
        # print('gradient difference', torch.norm(ffoqp_grad - cvxpylayer_grad, p=1))
        # print('cosine similarity', torch.nn.functional.cosine_similarity(ffoqp_grad, cvxpylayer_grad, dim=0))

    @unittest.skip
    def test_simple_batch_socp(self):
        set_seed(243)
        n = 5
        m = 1
        batch_size = 4

        P_sqrt = cp.Parameter((n, n), name='P_sqrt')
        q = cp.Parameter((n, 1), name='q')
        A = cp.Parameter((m, n), name='A')
        b = cp.Parameter((m, 1), name='b')

        x = cp.Variable((n, 1), name='x')

        objective = 0.5 * cp.sum_squares(P_sqrt @ x) + q.T @ x
        constraints = [A@x == b, cp.norm(x) <= 1]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        prob_tch = BilevelLayer(prob, [P_sqrt, q, A, b], [x])

        P_sqrt_tch = torch.randn(batch_size, n, n, requires_grad=True)
        q_tch = torch.randn(batch_size, n, 1, requires_grad=True)
        A_tch = torch.randn(batch_size, m, n, requires_grad=True)
        b_tch = torch.randn(batch_size, m, 1, requires_grad=True)

        torch.autograd.gradcheck(prob_tch, (P_sqrt_tch, q_tch, A_tch, b_tch))

    @unittest.skip
    def test_least_squares(self):
        set_seed(243)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = BilevelLayer(prob, [A, b], [x])

        A_th = torch.randn(m, n).double().requires_grad_()
        b_th = torch.randn(m).double().requires_grad_()

        x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

        def lstsq(
            A,
            b): return torch.linalg.solve(
            A.t() @ A + torch.eye(n, dtype=torch.float64),
                (A.t() @ b).unsqueeze(1))
        x_lstsq = lstsq(A_th, b_th)

        grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])
        grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th])

        self.assertAlmostEqual(
            torch.norm(
                grad_A_cvxpy -
                grad_A_lstsq).item(),
            0.0)
        self.assertAlmostEqual(
            torch.norm(
                grad_b_cvxpy -
                grad_b_lstsq).item(),
            0.0)
        
    @unittest.skip
    def test_least_squares_custom_method(self):
        set_seed(243)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = BilevelLayer(prob, [A, b], [x], custom_method=(forward_numpy, backward_numpy))

        A_th = torch.randn(m, n).double().requires_grad_()
        b_th = torch.randn(m).double().requires_grad_()

        x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

        def lstsq(
            A,
            b): return torch.linalg.solve(
            A.t() @ A + torch.eye(n, dtype=torch.float64),
                (A.t() @ b).unsqueeze(1))
        x_lstsq = lstsq(A_th, b_th)

        grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])
        grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th])

        self.assertAlmostEqual(
            torch.norm(
                grad_A_cvxpy -
                grad_A_lstsq).item(),
            0.0)
        self.assertAlmostEqual(
            torch.norm(
                grad_b_cvxpy -
                grad_b_lstsq).item(),
            0.0)

    @unittest.skip
    def test_logistic_regression(self):
        set_seed(243)
        N, n = 10, 2
        X_np = np.random.randn(N, n)
        a_true = np.random.randn(n, 1)
        y_np = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))

        X_tch = torch.from_numpy(X_np)
        X_tch.requires_grad_(True)
        lam_tch = 0.1 * torch.ones(1, requires_grad=True, dtype=torch.double)

        a = cp.Variable((n, 1))
        X = cp.Parameter((N, n))
        lam = cp.Parameter(1, nonneg=True)
        y = y_np

        log_likelihood = cp.sum(
            cp.multiply(y, X @ a) -
            cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ a]).T, axis=0,
                           keepdims=True).T
        )
        prob = cp.Problem(
            cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))

        fit_logreg = BilevelLayer(prob, [X, lam], [a])

        def layer_eps(*x):
            return fit_logreg(*x, solver_args={"eps": 1e-12})

        torch.autograd.gradcheck(layer_eps,
                                 (X_tch,
                                  lam_tch),
                                 eps=1e-4,
                                 atol=1e-3,
                                 rtol=1e-3)

    @unittest.skip
    def test_entropy_maximization(self):
        set_seed(243)
        n, m, p = 5, 3, 2

        tmp = np.random.rand(n)
        A_np = np.random.randn(m, n)
        b_np = A_np.dot(tmp)
        F_np = np.random.randn(p, n)
        g_np = F_np.dot(tmp) + np.random.rand(p)

        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        F = cp.Parameter((p, n))
        g = cp.Parameter(p)
        obj = cp.Maximize(cp.sum(cp.entr(x)) - .01 * cp.sum_squares(x))
        constraints = [A * x == b,
                       F * x <= g]
        prob = cp.Problem(obj, constraints)
        layer = BilevelLayer(prob, [A, b, F, g], [x])

        A_tch, b_tch, F_tch, g_tch = map(
            lambda x: torch.from_numpy(x).requires_grad_(True), [
                A_np, b_np, F_np, g_np])
        torch.autograd.gradcheck(
            lambda *x: layer(*x, solver_args={"eps": 1e-12,
                                              "max_iters": 10000}),
            (A_tch,
             b_tch,
             F_tch,
             g_tch),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3)

    @unittest.skip
    def test_lml(self):
        set_seed(1)
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
        cons = [cp.sum(y) == k]
        prob = cp.Problem(cp.Minimize(obj), cons)
        lml = BilevelLayer(prob, [x], [y])

        x_th = torch.DoubleTensor([1., -1., -1., -1.])
        x_th.requires_grad_(True)
        torch.autograd.gradcheck(lml, x_th, eps=1e-5, atol=1e-4, rtol=1e-4)

    @unittest.skip
    def test_sdp(self):
        set_seed(2)

        n = 3
        p = 3
        C = cp.Parameter((n, n))
        A = [cp.Parameter((n, n)) for _ in range(p)]
        b = [cp.Parameter((1, 1)) for _ in range(p)]

        C_tch = torch.randn(n, n, requires_grad=True).double()
        A_tch = [torch.randn(n, n, requires_grad=True).double()
                 for _ in range(p)]
        b_tch = [torch.randn(1, 1, requires_grad=True).double()
                 for _ in range(p)]

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(A[i]@X) == b[i] for i in range(p)
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(C@X) + cp.sum_squares(X)),
                          constraints)
        layer = BilevelLayer(prob, [C] + A + b, [X])
        torch.autograd.gradcheck(lambda *x: layer(*x,
                                                  solver_args={'eps': 1e-12}),
                                 [C_tch] + A_tch + b_tch,
                                 eps=1e-6,
                                 atol=1e-3,
                                 rtol=1e-3)

    @unittest.skip
    def test_not_enough_parameters(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaises(ValueError):
            layer = BilevelLayer(prob, [lam], [x])  # noqa: F841

    @unittest.skip
    def test_not_enough_parameters_at_call_time(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        layer = BilevelLayer(prob, [lam, lam2], [x])  # noqa: F841
        with self.assertRaisesRegex(
                ValueError,
                'A tensor must be provided for each CVXPY parameter.*'):
            layer(lam)

    @unittest.skip
    def test_too_many_variables(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaises(ValueError):
            layer = BilevelLayer(prob, [lam], [x, y])  # noqa: F841

    def test_infeasible(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        layer = BilevelLayer(prob, [param], [x])
        param_tch = torch.ones(1)
        with self.assertRaises(diffcp.SolverError):
            layer(param_tch)

    @unittest.skip
    def test_unbounded(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(x), [x <= param])
        layer = BilevelLayer(prob, [param], [x])
        param_tch = torch.ones(1)
        with self.assertRaises(diffcp.SolverError):
            layer(param_tch)

    @unittest.skip
    def test_incorrect_parameter_shape(self):
        set_seed(243)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = BilevelLayer(prob, [A, b], [x])

        A_th = torch.randn(32, m, n).double().requires_grad_()
        b_th = torch.randn(20, m).double().requires_grad_()

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        A_th = torch.randn(32, m, n).double().requires_grad_()
        b_th = torch.randn(32, 2 * m).double().requires_grad_()

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        A_th = torch.randn(m, n).double().requires_grad_()
        b_th = torch.randn(2 * m).double().requires_grad_()

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        A_th = torch.randn(32, m, n).double().requires_grad_()
        b_th = torch.randn(32, 32, m).double().requires_grad_()

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

    @unittest.skip
    def test_broadcasting(self):
        set_seed(243)
        n_batch, m, n = 2, 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = BilevelLayer(prob, [A, b], [x])

        A_th = torch.randn(m, n).double().requires_grad_()
        b_th = torch.randn(m).double().unsqueeze(0).repeat(n_batch, 1) \
            .requires_grad_()
        b_th_0 = b_th[0]

        x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

        def lstsq(
            A,
            b): return torch.linalg.solve(
            A.t() @ A + torch.eye(n).double(),
                (A.t() @ b).unsqueeze(1))
        x_lstsq = lstsq(A_th, b_th_0)

        grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])
        grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th_0])

        self.assertAlmostEqual(
            torch.norm(
                grad_A_cvxpy / n_batch -
                grad_A_lstsq).item(),
            0.0)
        self.assertAlmostEqual(
            torch.norm(
                grad_b_cvxpy[0] -
                grad_b_lstsq).item(),
            0.0)

    @unittest.skip
    def test_shared_parameter(self):
        set_seed(243)
        m, n = 10, 5

        A = cp.Parameter((m, n))
        x = cp.Variable(n)
        b1 = np.random.randn(m)
        b2 = np.random.randn(m)
        prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b1)))
        layer1 = BilevelLayer(prob1, parameters=[A], variables=[x])
        prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b2)))
        layer2 = BilevelLayer(prob2, parameters=[A], variables=[x])

        A_tch = torch.randn(m, n, requires_grad=True)
        solver_args = {
            "eps": 1e-10,
            "acceleration_lookback": 0,
            "max_iters": 10000
        }

        torch.autograd.gradcheck(lambda A: torch.cat(
            [layer1(A, solver_args=solver_args)[0],
             layer2(A, solver_args=solver_args)[0]]), (A_tch,))

    @unittest.skip
    def test_equality(self):
        set_seed(243)
        n = 10
        A = np.eye(n)
        x = cp.Variable(n)
        b = cp.Parameter(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A@x == b])
        layer = BilevelLayer(prob, parameters=[b], variables=[x])
        b_tch = torch.randn(n, requires_grad=True)
        torch.autograd.gradcheck(lambda b: layer(
            b, solver_args={"eps": 1e-10,
                            "acceleration_lookback": 0})[0].sum(),
            (b_tch,))

    @unittest.skip
    def test_basic_gp(self):
        set_seed(243)

        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable(pos=True)

        a = cp.Parameter(pos=True, value=2.0)
        b = cp.Parameter(pos=True, value=1.0)
        c = cp.Parameter(value=0.5)

        objective_fn = 1/(x*y*z)
        constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
        problem = cp.Problem(cp.Minimize(objective_fn), constraints)
        problem.solve(cp.SCS, gp=True, eps=1e-12)

        layer = BilevelLayer(
            problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
        a_tch = torch.tensor(2.0, requires_grad=True)
        b_tch = torch.tensor(1.0, requires_grad=True)
        c_tch = torch.tensor(0.5, requires_grad=True)
        with torch.no_grad():
            x_tch, y_tch, z_tch = layer(a_tch, b_tch, c_tch)

        self.assertAlmostEqual(x.value, x_tch.detach().numpy(), places=5)
        self.assertAlmostEqual(y.value, y_tch.detach().numpy(), places=5)
        self.assertAlmostEqual(z.value, z_tch.detach().numpy(), places=5)

        torch.autograd.gradcheck(lambda a, b, c: layer(
            a, b, c, solver_args={
                "eps": 1e-12, "acceleration_lookback": 0})[0].sum(),
                (a_tch, b_tch, c_tch), atol=1e-3, rtol=1e-3)

    @unittest.skip
    def test_no_grad_context(self):
        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        bilevel_layer = BilevelLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        with torch.no_grad():
            solution, = bilevel_layer(A_tch, b_tch)

        self.assertFalse(solution.requires_grad)

    @unittest.skip
    def test_requires_grad_false(self):
        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        bilevel_layer = BilevelLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=False)
        b_tch = torch.randn(m, requires_grad=False)

        solution, = bilevel_layer(A_tch, b_tch)

        self.assertFalse(solution.requires_grad)


if __name__ == '__main__':
    unittest.main()