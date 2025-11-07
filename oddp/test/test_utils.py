from unittest import TestCase
from itertools import combinations

from .._utils import partitions, sign_complement


class TestUtils(TestCase):

    def test_sign_reorder(self):
        for n in range(8):
            for q in range(n):
                for a in combinations(range(n), q):
                    sign1 = (-1) ** (sum(a) + (q // 2) % 2 + (n + 1) * q)
                    sign2 = sign_complement(range(n), a)
                    sign3 = (-1) ** (sum(a) + (q // 2) % 2)
                    sign4 = sign_complement(range(n), a, last=False)
                    self.assertEqual(sign1, sign2, f'A={a}, n={n} ')
                    self.assertEqual(sign3, sign4, f'A={a}, n={n} ')

    def test_partitions(self):
        '''Tests whether partitions((0,1)) and partitions((0,1,2)) are
        correct'''
        self.assertEqual(({((0, 1),): 1}, {((0,), (1,)): 1, ((1,), (0,)): -1}),
                         partitions((0, 1)))
        self.assertEqual(
            ({((0, 1, 2),): 1},
             {((0, 1), (2,)): 1, ((0, 2), (1,)): -1, ((1, 2), (0,)): 1,
              ((0,), (1, 2)): 1,
              ((1,), (0, 2)): -1, ((2,), (0, 1)): 1},
             {((0,), (1,), (2,)): 1, ((0,), (2,), (1,)): -1,
              ((1,), (0,), (2,)): -1,
              ((1,), (2,), (0,)): 1, ((2,), (0,), (1,)): 1,
              ((2,), (1,), (0,)): -1}),
            partitions((0, 1, 2)),
        )
