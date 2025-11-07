from unittest import TestCase
from itertools import combinations

from ..tate_product import straightening_builder, codegeneracy, \
    tate_product_dual, tate_product
from ..objects import PerCochain, MilnorCochain


class TestStraightening(TestCase):
    def test_straightening_builder(self):
        """tests whether straightening satisfies the following:
        (1) The length of the complete dictionary E is r times the length of
        the concise dictionary D.
        (2) that straightening is defined for all non-empty simplices
        (3) that the value of the straightening of A belongs to A
        (4) that the predecessor of the value of the straightening of A does
        not belong to A
        (5) that the straightening is cyclically equivariant
        """
        for r in [3, 5, 7, 11]:
            short_straightening, straightening = straightening_builder(r)
            self.assertEqual(len(short_straightening) * r,
                             len(straightening))  # (1)
            for k in range(1, r):
                for face in combinations(range(r), k):
                    self.assertIn(face, straightening.keys())  # (2)
                    self.assertIn(straightening[face], face)  # (3)
                    self.assertNotIn((straightening[face] - 1) % r,
                                     face)  # (4)
                    for j in range(r):
                        face_1 = tuple((a + j for a in face if a + j < r))
                        face_2 = tuple((a + j - r for a in face if a + j >= r))
                        self.assertEqual(
                            (straightening[face_2 + face_1] - j) % r,
                            straightening[face])  # (5)


class TestTateProductDual(TestCase):
    def setUp(self):
        codegeneracy_dict = {}
        for r in [3, 5, 7]:
            _, straightening = straightening_builder(r)
            codegeneracy_dict[r] = codegeneracy(straightening, r)
        self.codegeneracy_dict = codegeneracy_dict

    def test_tate_product_dual(self):
        """Tests whether codegeneracy_map is
        (0) non-trivial
        (1) a chain map
        (2) equivariant
        (3) satisfies the codegeneracy conditions
        """
        codeg_dict = self.codegeneracy_dict
        for y in PerCochain._examples(order_tuple=(3, 5)):
            if y.dual:
                r, q, t = y.order, -y.degree, y.torsion
                if q < 2 * (r - 1):
                    x = tate_product_dual(y, codeg_dict[r])
                    self.assertNotEqual(x,
                                        MilnorCochain({}, q, r, n=1, dual=True,
                                                      torsion=t),
                                        msg=f'y = {y}, r={r}')
                    self.assertEqual(x.partial(),
                                     tate_product_dual(y.partial(),
                                                       codeg_dict[r]),
                                     f'Failure at y = {y}, r = {r}, '
                                     f'q = {y.degree}, n = {x.n}')
                    self.assertEqual(x.action(),
                                     tate_product_dual(y.action(),
                                                       codeg_dict[r]))
                    if y.degree > r - 1:
                        z1 = x.evaluate(x.p_sigma())
                        z2 = x.evaluate(x.p_sigma(), left=True)
                        aux = {(per,): c for per, c in y.items()}
                        x2 = MilnorCochain(aux, q - r + 1, r, 0, True, t)
                        self.assertEqual(z1, x2)
                        self.assertEqual(z2, x2.action(), f'{y}')

class TestTateProduct(TestCase):
    def setUp(self):
        codegeneracy_dict = {}
        for r in [3, 5, 7]:
            _, straightening = straightening_builder(r)
            codegeneracy_dict[r] = codegeneracy(straightening, r, dual=False)
        self.codegeneracy_dict = codegeneracy_dict

    def test_tate_product(self):
        """Tests whether codegeneracy_map is
        (0) non-trivial
        (1) a chain map
        (2) equivariant
        (3) satisfies the codegeneracy conditions
        """
        codeg_dict = self.codegeneracy_dict
        def sample_generator():
            for r in [3,5]:
                for m1 in range(r):
                    for m2 in range(r):
                        for element1 in combinations(range(r), m1):
                            for element2 in combinations(range(r), m2):
                                per_chain_1 = PerCochain({element1:1}, degree = m1, order=r, dual=False)
                                per_chain_2 = PerCochain({element2: 1}, degree=m2,
                                                        order=r, dual=False)
                                yield per_chain_1, per_chain_2

        for per_chain_1, per_chain_2 in sample_generator():
            x = tate_product(per_chain_1, per_chain_2, codeg_dict[per_chain_1.order])
            y1 = tate_product(per_chain_1.partial(), per_chain_2, codeg_dict[per_chain_1.order])
            y2 = tate_product(per_chain_1, per_chain_2.partial(), codeg_dict[per_chain_1.order])

            self.assertEqual(x.partial(), y1 + (-1)**per_chain_1.degree * y2)

            z = tate_product(per_chain_1.action(), per_chain_2.action(), codeg_dict[per_chain_1.order])
            self.assertEqual(x.action(), z)

            if per_chain_1.degree != 0:

                a = tate_product(per_chain_1, per_chain_1.p_sigma(x.order-1), codeg_dict[per_chain_1.order])
                b = per_chain_1.copy()
                b.degree += b.order-1
                self.assertEqual(a, b, f'{per_chain_1}')


                a = tate_product(per_chain_1.p_sigma(x.order - 1), per_chain_1,
                                 codeg_dict[per_chain_1.order])
                b = per_chain_1.copy()
                b.degree += b.order - 1

                self.assertEqual(a, b.action(), f'{per_chain_1}')
