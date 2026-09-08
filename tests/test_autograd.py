import unittest
import math
from autograd import Value, MLP, SGD, mse_loss

class TestAutogradEngine(unittest.TestCase):
    def test_scalar_arithmetic(self):
        a = Value(3.0)
        b = Value(-4.0)
        c = a * b + Value(10.0)
        d = c / a - Value(2.0)
        d.backward()
        self.assertAlmostEqual(d.data, (-12.0 + 10.0)/3.0 - 2.0, places=5)
        self.assertNotEqual(a.grad, 0.0)

    def test_finite_difference_grad_check(self):
        def f(x_val):
            x = Value(x_val)
            y = (x ** 2) + 3*x.exp() + (2*x).sin_approx() if hasattr(x, 'sin_approx') else (x ** 2) + 3*x.exp()
            y.backward()
            return y.data, x.grad

        x0 = 1.5
        eps = 1e-6
        y0, analytical_grad = f(x0)
        x_plus = Value(x0 + eps)
        y_plus = (x_plus ** 2) + 3*x_plus.exp()
        x_minus = Value(x0 - eps)
        y_minus = (x_minus ** 2) + 3*x_minus.exp()
        numerical_grad = (y_plus.data - y_minus.data) / (2 * eps)
        self.assertAlmostEqual(analytical_grad, numerical_grad, places=4)

    def test_activations(self):
        x = Value(0.0)
        t = x.tanh()
        t.backward()
        self.assertAlmostEqual(t.data, 0.0, places=5)
        self.assertAlmostEqual(x.grad, 1.0, places=5)

        x2 = Value(2.0)
        r = x2.relu()
        r.backward()
        self.assertAlmostEqual(x2.grad, 1.0, places=5)

        x3 = Value(-2.0)
        r2 = x3.relu()
        r2.backward()
        self.assertAlmostEqual(x3.grad, 0.0, places=5)

    def test_mlp_xor_convergence(self):
        model = MLP(2, [4, 1])
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        xs = [[2.0, 3.0], [3.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]

        initial_loss = None
        final_loss = None
        for k in range(50):
            ypred = [model(x) for x in xs]
            loss = mse_loss(ypred, [Value(y) for y in ys])
            if k == 0:
                initial_loss = loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.data

        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
