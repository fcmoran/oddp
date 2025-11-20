r"""
This module uses the chain maps ``phi_dual``, ``psi_dual`` and ``abc_dual``
to implement higher cup products and chain level power operations as in [2].

Higher cup products
-------------------

The higher cup product of degree `i` for the prime `p` on the simplex of
dimension `n` is a chain map

.. math::
    \smile_{p,i}\colon \mathrm{N}(\Delta^{n}) \to
    \mathrm{N}(\Delta^{n})^{\otimes p}

of degree `i` computed as follows: First, create an instance of the generator
:math:`e_{i-(p-1)n}^\vee` of the dual of the shifted augmented minimal
resolution:

.. code-block:: python

   minimal_cochain = MinimalCochain({0:1}, degree=i-(p-1)*n, order=p, torsion=0)

Then, if ``torsion=p`` or ``p=3``, apply the following composition:

.. code-block:: python

   from math import factorial
   factorial((p - 1) // 2)**(n+1) * abc_dual(psi_dual(phi_dual(minimal_cochain),n)).alex()

For example,

>>> p, n, i =3, 1, 0
>>> minimal_cochain = MinimalCochain({0:1}, degree=i-(p-1)*n , order=p, torsion=0)
>>> abc_dual(psi_dual(phi_dual(minimal_cochain),n)).alex()
TensorCochain({((0, 1), (1,), (1,)): 1, ((0,), (0,), (0, 1)): 1, ((0,), (0, 1), (1,)): -1, ((0, 1), (), (0, 1)): -1}, degree=4, order=3, n=1, dual=False, torsion=0, sph_aug=False)

If ``torsion=0`` and ``p!=3``, then the map ``phi_dual`` is only defined on the
rational numbers (although after multiplying by the factorial number, the
coefficients will be actually integers), thus we have to write instead:

.. code-block:: python

    from fractions import Fraction
    minimal_cochain = MinimalCochain({0:Fraction(1)}, degree=i-p*(n+1), order=p, torsion=0)
    factorial((p - 1) // 2)**(n+1) * abc_dual(psi_dual(phi_dual(minimal_cochain),n)).alex()

This can be compared to the method ``Surjection.may_steenrod_structure`` of
the comch package that also computes higher cup products:

.. code-block:: python

    from comch import Surjection.may_steenrod_structure
    Surjection.may_steenrod_structure(p, i, torsion=p)

The higher cup products obtained here and comch are in general different,
but for ``p = 3`` they agree up to sign.

Power operations
----------------

If `X` is a simplicial set and :math:`s \leq 0`, there are power operations

.. math::
    P_s \colon \mathrm{H}(\mathrm{N}^{\vee}(X)) \to \mathrm{H}(\mathrm{N}^{\vee}(X))

.. math::
    \beta P_s \colon \mathrm{H}(\mathrm{N}^{\vee}(X)) \to \mathrm{H}(\mathrm{N}^{\vee}(X))

of degree :math:`2s(p-1)` and :math:`2s(p-1) - 1` respectively.

If :math:`x` is a representative of a cohomology class of degree `q`, then
a cochain representative `y` of :math:`P_s([x])` or :math:`\beta P_s([x])`
is obtained as the only cochain whose evaluation on
a chain :math:`a` is the evaluation of :math:`x^{\otimes p}` on the
`p`-fold tensor :math:`\smile_{p, (2s-q)(p-1)}`
(or :math:`\smile_{p, (2s-q)(p-1)+p}`):

.. code-block:: python

    from math import factorial
    degree = (2 * (p-1) * s - bockstein) * p
    n = -q -2*s*(p-1) + bockstein
    minimal_cochain = MinimalCochain({0:1}, degree=degree, order=p, torsion=p)
    factorial((p - 1) // 2)**(n+1) * abc_dual(psi_dual(phi_dual(minimal_cochain),n,homogeneous=True)).alex()

This can be compared to the method ``Surjection.steenrod_chain`` of comch:

.. code-block:: python

    from comch import Surjection.steenrod_chain
    Surjection.steenrod_chain(p,s,q,bockstein)


Power operations for simplicial complexes
-----------------------------------------

The functions below allow to compute power operations on simplicial sets. Here
are some applications: First, we bring the following simplicial complex on 19 vertices found
by Bagchi and Datta [1] that is a model of :math:`\mathbb{CP}^3`. We also bring in
a representative of a generator of the second cohomology group, found using
``sage.topology``.

>>> from oddp._samples import CP3, cochain_CP3

Now, we may apply either of the implementations below to find that the image
of the class of this cochain under the power operation :math:`P_{-1}` is
non-trivial:

>>> power_operation_0(CP3, cochain_CP3, 3, -1, -2)
ParametricCounter({(2, 8, 10, 11, 13, 14, 18): 2}, torsion=3)

>>> power_operation_1(CP3, cochain_CP3, 3, -1, -2)
ParametricCounter({(2, 8, 10, 11, 13, 14, 18): 2}, torsion=3)

As a second example, we bring in the simplicial model of the Moore space
for the group :math:`\mathbb{Z}/3 \mathbb{Z}` implemented in ``sage.topology``. A cochain
generating the first cohomology group was found using ``sage.topology`` too.

>>> from oddp._samples import Moore3, cochain_Moore3

Now, we may apply either of the implementations below to find that the image
of the class of this cochain under the power operation :math:`\beta P_{0}` is
non-trivial:

>>> power_operation_0(Moore3, cochain_Moore3, 3, 0, -1, bockstein=True)
ParametricCounter({(4, 5, 6): 1}, torsion=3)

>>> power_operation_1(Moore3, cochain_Moore3, 3, 0, -1, bockstein=True)
ParametricCounter({(4, 5, 6): 1}, torsion=3)

References
----------
[1] Bagchi, B., Datta, B. *A Triangulation of CP^3 as Symmetric Cube of S^2*.
Discrete Comput Geom 48, 310–329 (2012).
https://doi.org/10.1007/s00454-012-9436-2

[2] Cantero-Morán, F., Medina-Mardones, A., Steenrod operations and Tate
resolutions.
"""


from itertools import permutations, combinations

from oddp.chain_maps import phi_dual, psi_aux, psi_dual, abc, abc_dual
from oddp.objects import ParametricCounter, MinimalCochain, TensorCochain
from oddp._utils import admissible_iterator, admissible_iterator_set


def _conversion_comch_to_oddp(p, s, q, bockstein):
    degree = s*2*(p-1)*p - bockstein*p
    order = p
    n = -q -s*2*(p-1) + bockstein
    return order, degree, n

def _conversion_oddp_to_comch(order, degree, n):
    p = order
    if degree % p == 0 and ((degree // p) - 1) % (2 * (p - 1)) == 0:
        bockstein = True
        s = - ((degree // p) - bockstein) // (2 * (p - 1))
    elif degree % p == 0 and (degree // p) % (2 * (p - 1)) == 0:
        bockstein = False
        s = - ((degree // p) - bockstein) // (2 * (p - 1))
    else:
        return None
    q = -n - s * 2 * (p - 1) + bockstein
    return p, s, q, bockstein






def power_operation_0(simplicial_complex, cochain, p, s, q, bockstein = False, comch=True):
    r"""
    Computes a representative of the power operation :math:`P_s` applied to the cochain **cochain**.

    Parameters
    ----------
    simplicial_complex: dict[int, set]
        A dictionary with the faces of the simplicial complex.
    cochain: dict[tuple, int]
        A dictionary representing a cochain in the simplicial complex.
    p: int
        The order of the power operation.
    s: int
        The index of the power operation.
    q: int
        the degree of the cochain **cochain**.
    bockstein: bool
        Whether the power operation includes the Bockstein or not.
    comch: TO BE REMOVED.

    Returns
    -------
    ParametricCounter

    Notes
    -----
    This implementation requires N times S simple operations, where N is the
    number of summands in the corresponding cup-i product and S is the number
    of faces of dimension -q - 2(p-1)s + Bockstein.

    """

    if comch:
        order, degree, n = _conversion_comch_to_oddp(p, s, q, bockstein)
    else:
        order, degree, n = p, p*(s-bockstein), -(q+s-bockstein)
    minimal_cochain = MinimalCochain({0: 1}, degree, order, torsion=order)
    tensor_chain = abc_dual(psi_dual(phi_dual(minimal_cochain), n, homogeneous=True)).alex()
    sol = ParametricCounter(torsion=order)
    for facet in simplicial_complex[n]:
        for tensor, a in tensor_chain.items():
            new_a = a
            for factor in tensor:
                bla = cochain.get(tuple((facet[j] for j in factor)))
                if bla is None:
                    break
                else:
                    new_a *= bla
            else:
                sol += ParametricCounter({facet:new_a}, torsion=order)
    return sol


def power_operation_1(simplicial_complex, cochain, p, s, q, bockstein = False):
    r"""
    Computes a representative of the power operation :math:`P_s` applied to the cochain **cochain**.

    Parameters
    ----------
    simplicial_complex: dict[int, set]
        A dictionary with the faces of the simplicial complex.
    cochain: dict[tuple, int]
        A dictionary representing a cochain in the simplicial complex.
    p: int
        The order of the power operation.
    s: int
        The index of the power operation.
    q: int
        the degree of the cochain **cochain**.
    bockstein: bool
        Whether the power operation includes the Bockstein or not.

    Returns
    -------
    ParametricCounter

    Notes
    -----
    This implementation requires binom(c,p) simple operations, where c is the
    number of summands of the cochain, p is the order of the power operation
    and binom is the binomial coefficient. This is a generalization of the
    implementation ``steenrod_diagonal`` from the package steenroder for `p=2`
    to odd primes.

    """

    order, degree, n = _conversion_comch_to_oddp(p, s, q, bockstein)
    signatures = psi_aux(phi_dual(MinimalCochain({0:1}, degree=degree, order=order, torsion=order)), n)
    test = TensorCochain(degree=-degree, order=order, n=n, dual=False, torsion=order, sph_aug=True)
    sol = ParametricCounter(torsion=order)
    # facets = tuple(sorted(simplicial_complex[n]))
    counter = 0
    doit = True
    for cochain_fold in combinations(cochain.items(), order):

        top = set()
        b = 1
        for cochain_element, a in cochain_fold:
            top.update(cochain_element)
            b *= a
        top = tuple(sorted(top))

        if top in simplicial_complex[n]:
            counter += 1
            u_tensor = []
            for cochain_element, a in cochain_fold:
                diff = [j for j, v in enumerate(top) if v not in cochain_element]
                if len(diff) == -2 * (p - 1) * s + bockstein:
                    u_tensor.append(tuple(diff))
                else:
                    break
            else:
                counter += 1
                for tensor in permutations(u_tensor, order):

                    tensor_cochain = test.create({tensor: b},
                                                 degree=degree, dual=True)

                    sign = next(iter(tensor_cochain.alex().values()))

                    tensor_chain = test.create({tensor:b})
                    milnor_chain = abc(tensor_chain)
                    shape, c = next(iter(milnor_chain.items()))
                    shape = tuple((a for a in shape if len(a)>0))
                    c *= signatures.get(shape, 0) * b * sign

                    if c:
                        sol += sol.create({top: c})
    return sol


def power_operation_three(simplicial_complex, cochain, p, s, q, bockstein = False):
    r"""
    Computes a representative of :math:`P_s` for the prime 3 applied to the cochain **cochain**.

    Parameters
    ----------
    simplicial_complex: dict[int, set]
        A dictionary with the faces of the simplicial complex.
    cochain: dict[tuple, int]
        A dictionary representing a cochain in the simplicial complex.
    p: 3
        The order of the power operation.
    s: int
        The index of the power operation.
    q: int
        the degree of the cochain **cochain**.
    bockstein: bool
        Whether the power operation includes the Bockstein or not.

    Returns
    -------
    ParametricCounter

    Notes
    -----
    This implementation is similar to power_operation_1, but a bit faster.

    """
    if p != 3:
        return NotImplemented

    order, degree, n = _conversion_comch_to_oddp(p, s, q, bockstein)
    signatures = psi_aux(phi_dual(MinimalCochain({0:1}, degree=degree, order=order, torsion=order)), n)
    test = TensorCochain(degree=-degree, order=order, n=n, dual=False, torsion=order, sph_aug=True)
    sol = ParametricCounter(torsion=order)
    # facets = tuple(sorted(simplicial_complex[n]))

    ord_cochain_list = sorted(cochain.items())

    L = {}
    for i, (key, value) in enumerate(ord_cochain_list):
        new = []
        top1=set(key)
        for k,v in ord_cochain_list[i+1:]:
            if len(top1.difference(k)) <= -2 * (3 - 1) * s:
                new.append((k,v))
        L[key] = new



    for cochain_1, a1 in ord_cochain_list:
        top1 = set(cochain_1)
        cochain_fold = [cochain_1,(),()]
        for cochain_2, a2 in L[cochain_1]:
            top2 = top1.union(cochain_2)
            cochain_fold[1] = cochain_2
            for cochain_3, a3 in L[cochain_2]:
                top3 = top2.union(cochain_3)
                cochain_fold[2] = cochain_3
                ttop = tuple(sorted(top3))
                b = a1*a2*a3
                if ttop in simplicial_complex[n]:
                    u_tensor = []
                    for cochain_element in cochain_fold:
                        diff = [j for j, v in enumerate(ttop) if v not in cochain_element]
                        if len(diff) == -2 * (p - 1) * s + bockstein:
                            u_tensor.append(tuple(diff))
                        else:
                            break
                    else:
                        for tensor in permutations(u_tensor, order):

                            tensor_cochain = test.create({tensor: b},
                                                         degree=degree, dual=True)

                            sign = next(iter(tensor_cochain.alex().values()))

                            tensor_chain = test.create({tensor:b})
                            milnor_chain = abc(tensor_chain)
                            shape, c = next(iter(milnor_chain.items()))
                            shape = tuple((a for a in shape if len(a)>0))
                            c *= signatures.get(shape, 0) * b * sign

                            if c:
                                sol += sol.create({ttop: c})
    return sol