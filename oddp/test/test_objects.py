import unittest

from oddp.objects import *
from oddp._utils import *


class TestParametricCounter(unittest.TestCase):

    def setUp(self):
        self.x1 = ParametricCounter({'a': 1, 'b': -2})
        self.x2 = ParametricCounter({'a': 1, 'b': -2}, p=3, q=4)
        self.x3 = ParametricCounter({'a': 1, 'b': -2, 'c': 0})
        self.y1 = ParametricCounter({'a': -1, 'b': -2})
        self.y2 = ParametricCounter({'a': -1, 'b': -2}, p=3, q=4)
        self.y3 = {'a': -1, 'b': -2}
        self.z1 = ParametricCounter({'b': -4})
        self.z2 = ParametricCounter({'b': -4}, p=3, q=4)
        self.a1 = ParametricCounter({'a': 2})
        self.a2 = ParametricCounter({'a': 2}, p=3, q=4)
        self.b1 = ParametricCounter({'a': -2, 'b': -4})
        self.b2 = ParametricCounter({'a': -2, 'b': -4}, p=3, q=4)

    def test_init(self):
        self.assertNotIn('c', self.x3.keys(), f'__init__ does not delete'
                         + f'keys with 0 value')

    def test_add(self):
        self.assertEqual(self.x2 + self.y2, self.z2)
        self.assertEqual(self.x1 + self.y1, self.z1)
        self.assertEqual(self.x3 + self.y1, self.z1)

    def test_sub(self):
        self.assertEqual(self.x2 - self.y2, self.a2)
        self.assertEqual(self.x1 - self.y1, self.a2)

    def test_mul(self):
        self.assertEqual(2 * self.y2, self.b2)
        self.assertEqual(2 * self.y1, self.b1)
        self.assertEqual(2 * self.y2, self.y2 + self.y2)
        self.assertEqual(2 * self.y1, self.y1 + self.y1)

    def test_neg(self):
        self.assertEqual(-self.x1, (-1) * self.x1)

    def test_iadd(self):
        w = self.x1.copy()
        w += self.y1
        self.assertEqual(w, self.x1 + self.y1)
        u = self.x2.copy()
        u += self.y2
        self.assertEqual(u, self.x2 + self.y2)

    def test_isub(self):
        w = self.x1.copy()
        w -= self.y1
        self.assertEqual(w, self.x1 - self.y1)
        u = self.x2.copy()
        u -= self.y2
        self.assertEqual(u, self.x2 - self.y2)

    def test_torsion(self):
        x = ParametricCounter({'a': 10, 'b': 7, 'c': 0, 'd':-3}, torsion = 5)
        y = ParametricCounter({'b': 2, 'd': 2}, torsion=5)
        self.assertEqual(y, x)
        z = ParametricCounter({'b': 2, 'd': 3}, torsion=5)
        u = ParametricCounter({'b': 4}, torsion=5)
        self.assertEqual(u, y + z)
        v = ParametricCounter({'d': 4}, torsion=5)
        self.assertEqual(v, y - z)
        w = y.copy()
        w += z
        self.assertEqual(u, w)
        w = y.copy()
        w -= z
        self.assertEqual(v, w)
        w = ParametricCounter({'b':1}, torsion=5)
        self.assertEqual(w, -u)

    def test_copy(self):
        w = self.x2.copy()
        self.assertEqual(self.x2, w)

class TestMinimalCochain(unittest.TestCase):

    def setUp(self):
        self.name = 'MinimalCochain'

    def test_action(self):
        for y in MinimalCochain._examples():
            y1 = y
            for i in range(y.order):
                y1 = y1.action()
            self.assertEqual(y, y1, f'Failure at {y = }, {y.order = }')

    def test_partial(self):
        for y in MinimalCochain._examples():
            x = y.create()
            x.degree = x.degree - 2
            self.assertEqual(y.partial().partial(), x, f'element tested: {y}')

    def test_partial_action(self):
        for y in MinimalCochain._examples():
            self.assertEqual(y.partial().action(),
                             y.action().partial(),
                             f'element tested: {y}')




class TestSimp(unittest.TestCase):

    def test_action(self):
        for face, n, dual in Simp._examples():
            y = Simp(face, n, dual)
            x = y
            a = 1
            for _ in range(n + 1):
                new_face, b = x.action()
                x = Simp(new_face, n, dual)
                a = a * b
            self.assertEqual(y.face, x.face,
                             f'Failure at {face, n, dual}')
            self.assertEqual(1, a)

    def test_partial(self):
        for face, n, dual in Simp._examples():
            double_coboundary = []
            for sigma, _ in Simp(face, n, dual).partial():
                for tau, _ in Simp(sigma, n, dual).partial():
                    if tau in double_coboundary:
                        double_coboundary.remove(tau)
                    else:
                        double_coboundary.append(tau)
            self.assertEqual(double_coboundary, [],
                             f'Error data: {face}, {n}, {dual}')

    def test_alex(self):
        for face, n, dual in Simp._examples():
            dual_face, a = Simp(face, n, dual).alex()
            dual_dual_face, b = Simp(dual_face, n, not dual).alex()
            self.assertEqual(face, dual_dual_face)
            self.assertEqual(1, a * b, f'Failure at {face, n, dual}')


class TestPerCochain(unittest.TestCase):

    def test_action(self):
        for y in PerCochain._examples():
            y1 = y
            for i in range(y.order):
                y1 = y1.action()
            self.assertEqual(y, y1,
                             f'Failure at {y=}')

    def test_partial(self):
        for y in PerCochain._examples():
            self.assertEqual(y.partial().partial(),
                             y.create(degree=y.degree - 2),
                             f'Failure at: {y=}')

    def test_partial_action(self):
        for y in PerCochain._examples():
            if len(y) > 0:
                self.assertEqual(y.partial().action(), y.action().partial(),
                             msg=f'Failure at: {y=}')


class TestMilnorCochain(unittest.TestCase):

    def test_action(self):
        for y in MilnorCochain._examples():
            y1 = y
            for i in range(y.order):
                y1 = y1.action()
            self.assertEqual(y, y1, f'Failure at {y=}')

    def test_partial(self):
        for y in MilnorCochain._examples():
            self.assertEqual(y.partial().partial(),
                             y.create(degree=y.degree - 2),
                             f'Failure at: {y=}')

    def test_partial_action(self):
        for y in MilnorCochain._examples():
            self.assertEqual(y.partial().action(), y.action().partial(),
                             msg=f'Failure at: {y=}')

    def test_evaluate(self):
        for r in [3, 5]:
            for q1 in range(r):
                for factor1 in combinations(range(r), q1):
                    for q2 in range(r):
                        for factor2 in combinations(range(r), q2):
                            tensor = (factor1, factor2)
                            q = q1 + q2
                            x12 = MilnorCochain({tensor: 1}, -q, r, 1,
                                                True)
                            x1 = MilnorCochain({(factor1,): 1}, q1, r, 0,
                                               False)
                            x2 = MilnorCochain({(factor2,): 1}, q2, r, 0,
                                               False)
                            x1d = MilnorCochain({(factor1,): 1}, -q1, r, 0,
                                                True)
                            x2d = MilnorCochain({(factor2,): 1}, -q2, r, 0,
                                                True)
                            self.assertEqual(x12.evaluate(x2), x1d,
                                             f'Failure at: {x12}')
                            self.assertEqual(x12.evaluate(x1, left=True), x2d,
                                             f'Failure at: {x12}')
                            if factor1 != factor2:
                                self.assertEqual(x12.evaluate(x2, left=True),
                                                 x1d.create(),
                                                 f'Failure at: {x12}')
                                self.assertEqual(x12.evaluate(x1),
                                                 x2d.create(),
                                                 f'Failure at: {x12}')


class TestTensorCochain(unittest.TestCase):

    def test_action(self):
        for y in TensorCochain._examples():
            y1 = y
            for i in range(y.order):
                y1 = y1.action()
            self.assertEqual(y, y1, f'Failure at {y=}')

    def test_partial(self):
        for y in TensorCochain._examples():
            self.assertEqual(y.partial().partial(),
                             y.create(degree=y.degree + 2), f'Failure at {y=}')

    def test_partial_action(self):
        for y in TensorCochain._examples():
            self.assertEqual(y.partial().action(), y.action().partial(),
                             msg=f'Failure at {y=}')
            
    def test_alex(self):
        for y in TensorCochain._examples():
            self.assertEqual(y.alex().action(),y.action().alex(), y)
            self.assertEqual(y.alex().partial(), (-1)**((y.n+1) * y.order) * y.partial().alex(), y)
            self.assertEqual(y, (-1)** ((y.n+1) * binom(y.order)) * y.alex().alex(), y)

# if __name__ == '__main__':
#     unittest.main()
