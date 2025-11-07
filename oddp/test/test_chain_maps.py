from unittest import TestCase

from ..objects import MinimalCochain, PerCochain, MilnorCochain, TensorCochain
from ..tate_product import straightening_builder, codegeneracy
from ..chain_maps import phi, phi_dual, psi, psi_dual, abc, abc_dual

class TestPhi(TestCase):
    def test_phi_chain_map(self):
        for y in PerCochain._examples():
            if y.dual: continue
            if y.order > 3: y.torsion = y.order
            self.assertEqual(phi(y).partial(), phi(y.partial()),
            f'Failure at element y = {y}, phi(y) = {phi(y)}')

    def test_phi_dual_equivariant_map(self):
        for y in PerCochain._examples():
            if y.dual: continue
            self.assertEqual(phi(y).action(), phi(y.action()),
            f'Failure at element y = {y}, phi(y) = {phi(y)}')


class TestPhiDual(TestCase):
    def test_phi_dual_chain_map(self):
        for y in MinimalCochain._examples():
            if not y.dual: continue
            if y.order > 3: y.torsion = y.order
            self.assertEqual(phi_dual(y).partial(), phi_dual(y.partial()),
            f'Failure at element y = {y}, phi(y) = {phi_dual(y)}')

    def test_phi_dual_equivariant_map(self):
        for y in MinimalCochain._examples():
            if not y.dual: continue
            self.assertEqual(phi_dual(y).action(), phi_dual(y.action()),
            f'Failure at element y = {y}, phi(y) = {phi_dual(y)}')

class TestPsiDual(TestCase):
    def setUp(self):
        codegeneracy_dict = {}
        for r in [3, 5, 7]:
            _, straightening = straightening_builder(r)
            codegeneracy_dict[r] = codegeneracy(straightening, r)
        self.codegeneracy_dict = codegeneracy_dict

    def test_psi_dual(self):
        """Tests whether psi is
        (0) is non-trivial
        (1) a chain map
        (2) equivariant
        """
        codeg_dict = self.codegeneracy_dict
        for y in PerCochain._examples(order_tuple=(3, 5)):
            for n in range(6):
                if y.dual:
                    q, r = y.degree, y.order
                    # self.assertNotEqual(psi(y, F[r]), MilnorElement({},r,q,
                    # y.torsion, n),
                    # f'Failure at element y = {y}, r={r}, q={q} n = {n}')
                    self.assertEqual(psi_dual(y, n, codeg_dict[r]).partial(),
                                     psi_dual(y.partial(), n, codeg_dict[r]),
                                     f'Failure at element y = {y} \
                        r={r}, q={q}, n = {n}')
                    self.assertEqual(psi_dual(y, n, codeg_dict[r]).action(),
                                     psi_dual(y.action(), n, codeg_dict[r]),
                                     f'Failure at element y = {y},  \
                        r={r}, n = {n}')

class TestPsi(TestCase):
    def setUp(self):
        codegeneracy_dict = {}
        for r in [3, 5, 7]:
            _, straightening = straightening_builder(r)
            codegeneracy_dict[r] = codegeneracy(straightening, r, dual=False)
        self.codegeneracy_dict = codegeneracy_dict

    def test_psi(self):
        """Tests whether psi is
        (0) is non-trivial
        (1) a chain map
        (2) equivariant
        """
        codeg_dict = self.codegeneracy_dict
        for y in MilnorCochain._examples(order_tuple=(3, 5)):
            for n in range(6):
                if not y.dual:
                    q, r = y.degree, y.order
                    # self.assertNotEqual(psi(y, F[r]), MilnorElement({},r,q,
                    # y.torsion, n),
                    # f'Failure at element y = {y}, r={r}, q={q} n = {n}')
                    self.assertEqual(psi(y).partial(),
                                     psi(y.partial()),
                                     f'Failure at element y = {y} \
                        r={r}, q={q}, n = {n}')
                    self.assertEqual(psi(y).action(),
                                     psi(y.action()),
                                     f'Failure at element y = {y},  \
                        r={r}, n = {n}')


class TestAbcDual(TestCase):

    def test_abc_dual_chainmap(self):
        for y in MilnorCochain._examples():
            if y.dual:
                self.assertEqual(abc_dual(y).partial(), abc_dual(y.partial()),
                f'Failure at element y = {y}, abc(y) = {abc_dual(y)} \
                r={y.order}, n = {y.n}')

    def test_abc_dual_equivariant_map(self):
        for y in MilnorCochain._examples():
            if y.dual:
                self.assertEqual(abc_dual(y).action(), abc_dual(y.action()),
                f'Failure at element y = {y}, abc(y) = {abc_dual(y)} \
                r={y.order}, n = {y.n}')


class TestAbc(TestCase):

    def test_abc_chainmap(self):
        for y in TensorCochain._examples():
            if not y.dual and y.sph_aug is True:
                self.assertEqual(abc(y).partial(), abc(y.partial()),
                f'Failure at element y = {y}, abc(y) = {abc(y)} \
                r={y.order}, n = {y.n}')

    def test_abc_equivariant_map(self):
        for y in TensorCochain._examples():
            if not y.dual and y.sph_aug is True:
                self.assertEqual(abc(y).action(), abc(y.action()),
                f'Failure at element y = {y}, abc(y) = {abc(y)} \
                r={y.order}, n = {y.n}')
