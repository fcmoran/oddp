r"""
This module implements the following chain maps and their duals
(see ``oddp.objects``):

.. math::
   \mathrm{W}_+(r) \overset{\phi}{\longleftarrow}
   \Lambda_+(r) \overset{\psi_n}{\longleftarrow}
   \mathrm{N}(\partial \Delta_+^{r-1})^{\otimes n+1}
   \overset{\alpha\circ \beta}{\longleftarrow}
   I \subset \mathrm{N}(\Delta_+^{n})^{\otimes r}

"""

from math import factorial
from itertools import combinations

from oddp.objects import (MinimalCochain, Simp, PerCochain, MilnorCochain,
                          TensorCochain)
from oddp.tate_product import straightening_builder, codegeneracy
from oddp._utils import partitions, sign_reorder




def phi_dual(minimal_cochain):
    """
    The dual of the chain map :math:`\phi`.

    Parameters
    ----------
    minimal_cochain : MinimalCochain
        A cochain in the Minimal resolution of the cyclic group. Only defined
        if minimal_cochain has the following coefficients:

            - Rational coefficients and odd order,
            - integer coefficients and ``order=3``,
            - modular integer coefficients with ``torsion=order`` and either
             odd order or ``order=2``.

    Returns
    ----------
    PerCochain

    Examples
    --------
    >>> minimal_cochain = MinimalCochain({0: 1}, degree=-2, order=3)
    >>> phi_dual(minimal_cochain)
    PerCochain({(0, 1): 1}, degree=-2, order=3, dual=True, torsion=0)

    >>> minimal_cochain = MinimalCochain({0:1}, degree=-2, order=5, torsion=5)
    >>> phi_dual(minimal_cochain)
    PerCochain({(0, 1): 3, (0, 3): 3, (2, 3): 3}, degree=-2, order=5, dual=True, torsion=5)

    """
    if not minimal_cochain.dual:
        raise ValueError('Only defined for cochains')
    q, r = -minimal_cochain.degree, minimal_cochain.order
    if q != 0 and q % (r-1) == 0:
        q_mod = r-1
    else:
        q_mod = q % (r-1)

    sol = PerCochain({}, **minimal_cochain.__dict__)
    for j, a in minimal_cochain.items():
        new = tuple(range(q_mod))
        sol += sol.create({new: a}).action(-j)
        # the max in the following line are there in order to work for order=2.
        last = tuple(range(max(r - 1 - (q_mod % 2) - q_mod, 0),
                           max(r - 1 - (q_mod % 2), 1)))
        while new != last:
            for i, v in enumerate(reversed(new)):
                #ii = q_mod - i - 1
                if v != last[-i-1]:
                    new = new[0:-i-1] + tuple(range(v + 2, v + 3 + i))
                    sol += sol.create({new: a}).action(-j)
                    break
    # the coefficient varphi(q).
    if minimal_cochain.order == 3:
        return sol
    else:
        q_mod = q % (r - 1)
        q_div = q // (r - 1)
        r_tilde = (r - 1) // 2
        r_tilde_fact = factorial(r_tilde)
        denominator = r_tilde_fact ** (q_div + 1)
        numerator = factorial((r - 1 - q_mod) // 2)
        return numerator * sol / denominator



def phi(per_chain):
    r"""
    The chain map :math:`\phi`.

    Parameters
    ----------
    per_chain: PerCochain(dual=False)
        A chain in the periodic resolution of the cyclic group. Only defined
        if per_chain has the following coefficients:

            - Rational coefficients and odd order,
            - integer coefficients and ``order=3``,
            - modular integer coefficients with ``torsion=order`` and either
            odd order or ``order=2``.
    Returns
    -------
    MinimalCochain(dual=False)

    Examples
    --------
    >>> per_chain = PerCochain({(0,1):1}, degree=2, order=3, dual=False)
    >>> phi(per_chain)
    MinimalCochain({0: 1}, degree=2, order=3, dual=False, torsion=0)

    If ``torsion!=order>3``, then ``phi`` is only defined with rational
    coefficients:

    >>> from fractions import Fraction
    >>> per_chain = PerCochain({(0,1):Fraction(1)}, degree=2, order=5, dual=False)
    >>> phi(per_chain)
    MinimalCochain({3: Fraction(1, 2), 0: Fraction(1, 2)}, degree=2, order=5, dual=False, torsion=0)

    but if ``torsion=order``, fractions are not needed:

    >>> per_chain = PerCochain({(0,1):1}, degree=2, order=5, dual=False, torsion=5)
    >>> phi(per_chain)
    MinimalCochain({3: 3, 0: 3}, degree=2, order=5, dual=False, torsion=5)

    """

    q, r = per_chain.degree, per_chain.order
    sol = MinimalCochain({}, **per_chain.__dict__)

    if q == 0:
        a = next(iter(per_chain.values()))
        sol += sol.create({0: a})
        return sol

    for face, c in per_chain.items():

        if (r + face[0] - face[-1]) % 2 == 0:
            even = (face[-1], face[0] + r)
            sign = (-1) ** len(face)
        else:
            even = None
            sign = 1

        for j in range(len(face) - 1):
            if (face[(j+1)] - face[j]) % 2 == 0:
                if even is None:
                    even = (face[j], face[j+1])
                    sign = (-1) ** (j + 1)
                else:
                    break
        else:
            if q % 2 == 0:
                for a in range(even[0] + 2, even[1] + 1, 2):
                    sol += sol.create({a % r: sign*c})
            else:
                for a in range(face[-1] + 3, face[0]+r+1, 2):
                    sol += sol.create({a % r: c})
                for j in range(len(face) - 1):
                    for a in range(face[j] + 3, face[j+1]+1, 2):
                        sol += sol.create({a:c})
    if r == 3:
        return sol
    else:
        q_mod = q % (r - 1)
        q_div = q // (r - 1)
        r_tilde = (r - 1) // 2
        r_tilde_fact = factorial(r_tilde)
        denominator = r_tilde_fact ** (q_div + 1)
        numerator = factorial((r - 1 - q_mod) // 2)
        return numerator * sol / denominator


def psi_aux(per_cochain, n, codegeneracy_dict=None):
    """The chain map psi_n without empty factors.

    :meta private:
    """

    q, r = -per_cochain.degree, per_cochain.order
    if codegeneracy_dict is None:
        _, straightening = straightening_builder(r)
        codegeneracy_dict = codegeneracy(straightening, r)
    sol_0 = {}
    if q == 0:
        for per_element, a in per_cochain.items():
            sol_0.update({(): a})
    else:
        m = (q - 1) // (r - 1)  # how many codegeneracies must be applied.
        # STEP 1: initialise the dictionary sol_0 with all the splittings.
        for per_element, a in per_cochain.items():
            per_partitions = partitions(per_element)
            for size in range(min(len(per_partitions), n + 1 - m)):
                for partition, b in per_partitions[size].items():
                    sol_0.update({partition: a * b})

        # STEP 2: Apply the codegeneracy map m times, while splitting the new
        # last factor in all possible ways.
        for j in range(m):
            sol_1 = {}
            for element, a in sol_0.items():
                length = len(element)
                initial, end = element[:-1], element[-1]
                for (factor_1, factor_2), b in codegeneracy_dict[end].items():
                    new_partitions = partitions(factor_2)
                    for size in range(min(len(new_partitions),
                                          n + 1 - length - (m - j - 1))):
                        for partition, c in new_partitions[size].items():
                            new_milnor = initial + (factor_1,) + partition
                            sol_1.update({new_milnor: a * b * c})

            sol_0 = sol_1
    return sol_0


def homogeneous_iterator(milnor_cochain, milnor_element, q, r, n, rule=None):
    """
    Builds an iterator used in the function psi_dual to improve performance.

    :meta private:
    """

    length = len(milnor_element)
    if q % r != 0:
        return ()
    if rule is None:
        rule = milnor_cochain.default_rule()
    sol_0 = {(): (0,) * r}
    for j, factor in enumerate(milnor_element):
        sol_1 = {}
        for labels, weights in sol_0.items():
            bottom = 0 if labels == () else labels[-1] + 1
            for new_label in range(bottom, n + 1 - (length - j - 1)):
                new_labels = labels + (new_label,)
                new_weights = list(weights)
                for v in factor:
                    new_weights[(v - rule[new_label]) % r] += 1
                    if new_weights[(v - rule[new_label]) % r] > q // r:
                        break
                else:
                    sol_1.update({new_labels: tuple(new_weights)})
        sol_0 = sol_1
    return tuple(sol_0.keys())


def psi_dual(per_cochain, n, codegeneracy_dict=None, homogeneous=False,
             rule=None):
    r"""The dual of the chain map :math:`\psi_n`.

    Parameters
    ----------
    per_cochain: PerCochain
        A cochain in the periodic resolution of the cyclic group. The order of
        the cyclic group ``per_cochain.order`` must be a prime number.
    n: int
    codegeneracy_dict: dict[tuple, ParametricCounter], default=None
        A dictionary implementing the dual of the weak codegeneracy. If None,
        a codegeneracy_dict is generated using the function
        ``tate_products.codegeneracy``.
    homogeneous: bool, default=False
        If True, returns only milnor cochains that will yield via
        ``chain_maps.abc_dual`` homogeneous
        tensor cochains (i.e., returns only the part needed for computing
        power operations)
    rule: tuple, default=None
        A tuple encoding the endomorphism that will be used in the map
        ``chain_maps.abc_dual``. It is only used if homogeneous is True.
        If None, the default value included as a method of the class
        MilnorCochain will be used.

    Returns
    ----------
    MilnorCochain

    Notes
    ----------
    The map :math:`\psi_n^\vee\colon \Lambda_+(r)^\vee \to
    \left(\mathrm{N}(\partial\Delta_+^{r-1})^{\otimes n+1}\right)^\vee`

    Examples
    --------
    >>> per_cochain = PerCochain({(0,):1}, degree=-3, order=3)
    >>> psi_dual(per_cochain, 1)
    MilnorCochain({((0,), (1, 2)): 1, ((0, 1), (2,)): 1}, degree=-3, order=3, n=1, dual=True, torsion=0)

    >>> per_cochain = PerCochain({(0,1):1}, degree=-4, order=3)
    >>> psi_dual(per_cochain, 1)
    MilnorCochain({((0, 1), (0, 2)): -1}, degree=-4, order=3, n=1, dual=True, torsion=0)

    """

    q, r, t = -per_cochain.degree, per_cochain.order, per_cochain.torsion
    if codegeneracy_dict is None:
        _, straightening = straightening_builder(r)
        codegeneracy_dict = codegeneracy(straightening, r)
    sol_without_gaps = psi_aux(per_cochain, n, codegeneracy_dict)
    sol = MilnorCochain({}, **per_cochain.__dict__)
    sol.n = n
    for element, a in sol_without_gaps.items():
        if not homogeneous:
            iterator = combinations(range(n + 1), len(element))
        else:
            iterator = homogeneous_iterator(sol, element, q, r, n, rule)
        for labels in iterator:
            it = iter(element)
            new_element = tuple(
                (next(iter(it)) if k in labels else () for k in range(n + 1)))
            sol += sol.create({new_element: a})
    return sol


def psi(milnor_chain, codegeneracy_dict=None):
    r"""
    The chain map :math:`\psi_n`.

    Parameters
    ----------
    milnor_chain: MilnorCochain(dual=False)
        A chain in the Milnor resolution of the cyclic group. The order of
        the cyclic group ``milnor_chain.order`` must be a prime number.
    codegeneracy_dict: dict[(tuple, tuple), ParametricCounter], default=None
        A dictionary implementing the weak codegeneracy. If None,
        a codegeneracy_dict is generated using the function codegeneracy.

    Returns
    -------
    PerCochain(dual=False)

    Examples
    --------
    >>> milnor_chain = MilnorCochain({((0,1),(2,),(0,)):1}, degree=4, order=3, n=2, dual=False)
    >>> psi(milnor_chain)
    PerCochain({(0, 1): 1}, degree=4, order=3, dual=False, torsion=0)

    """

    kwargs = milnor_chain.__dict__.copy()
    del kwargs['n']
    sol = PerCochain({}, **kwargs)
    r = sol.order
    if codegeneracy_dict is None:
        _, straightening = straightening_builder(r)
        codegeneracy_dict = codegeneracy(straightening, r, dual=False)

    for tensor, a in milnor_chain.items():
        new_tensor = list(tensor)
        new_sign = a
        while len(new_tensor) > 1:
            second = new_tensor.pop(-1)
            first = new_tensor.pop(-1)
            if len(first) + len(second) < r:
                if set(first).intersection(set(second)) != set():
                    break
                new_tensor.append(tuple(sorted(first + second)))
                new_sign *= sign_reorder(first, second)
            else:
                new_element = codegeneracy_dict.get((first, second))
                if new_element is not None:
                    if len(new_element) > 1:
                        raise ValueError('tate product is not setlike')
                    new_tensor.append(next(iter(new_element)))
                    new_sign *= next(iter(new_element.values()))
                else:
                    break
        else:
            sol += sol.create({new_tensor[0]: new_sign})
    return sol



def abc_dual(milnor_cochain, rule=None):
    r"""
    The dual of the composition of alpha and beta.

    Parameters
    ----------
    milnor_cochain: MilnorCochain
    rule: tuple[int]
        The tuple :math:`(i_0,\ldots,i_n)`, where :math:`\alpha =
        \rho^{i_0}\otimes \ldots \otimes \rho^{i_n}`

    Returns
    -------
    TensorCochain

    Notes
    -------
    This function applies first the endomorphism alpha given by
    :math:`\rho^{-i_0}\otimes \ldots \otimes \rho^{-i_n}`
    and then the chain map beta that takes a cochain in
    :math:`\left(\mathrm{N}(\partial \Delta_+^{r-1})^{\otimes (n+1)}\right)^{\vee}`
    to a cochain in
    :math:`\left(\mathrm{N}(\Delta_+^{n})^{\otimes r}\right)^{\vee}`.
    If **rule** is **None**, then it is set by the method default_rule()
    of the class MilnorCochain.

    [1], Section 4.3

    References
    -------
    [1] Cantero-Moran and Medina-Mardones, Steenrod operations and Tate
    resolutions of the cyclic group, arXiv...

    Examples
    -------
    >>> x = MilnorCochain({((0,),(),(0,2),()):1}, degree=-3, order=3, n=3)
    >>> print(abc_dual(x))
    TensorCochain({((0,), (2,), (2,)): -1}, degree=-3, order=3, n=3, dual=True, torsion=0, sph_aug=True)

    >>> milnor_cochain = MilnorCochain({((0,1),(2,),(0,),(1,)):1}, degree=-5, order=3, n=3)
    >>> abc_dual(milnor_cochain)
    TensorCochain({((0, 1), (0, 3), (2,)): 1}, degree=-5, order=3, n=3, dual=True, torsion=0, sph_aug=True)

    """
    r = milnor_cochain.order
    if rule is None:
        rule = milnor_cochain.default_rule()
    sol = TensorCochain(**milnor_cochain.__dict__)
    for tensor, a in milnor_cochain.items(): #(1,2)()()(0,)

        #Step 1: the endomorphism beta, that applies rho^{-i} to the ith factor
        new_tensor, new_a = [], 1
        for j, factor in enumerate(tensor):
            new_factor, new_b = Simp(factor,r-1).action(rule[j])
            new_tensor.append(new_factor)
            new_a = new_a*new_b
        new_tensor = tuple(new_tensor)

        #Step 2: the map alpha
        new_new_tensor = [[] for _ in range(r)]
        signs = dict({w:0 for w in range(r)})
        par = 0
        for j,new_factor in enumerate(new_tensor):
            for v in new_factor:
                new_new_tensor[v].append(j)
                for w in range(v):
                    signs[w] += 1
                par += signs[v]
        new_new_tensor = tuple((tuple(factor) for factor in new_new_tensor))

        sol += sol.create({new_new_tensor:new_a*a*(-1)**par})
    return sol



def abc(tensor_chain, rule=None):
    """
    The composition of the maps alpha and beta.

    Parameters
    ----------
    tensor_chain: TensorCochain(dual=False)
        A chain in the r-fold tensor product of the chains on the n-simplex.
    rule: tuple[int], default = None
        A tuple encoding the endomorphism alpha. If None, it takes
        the value of MilnorCochain.default_rule().

    Returns
    -------
    MilnorCochain(dual=False)

    Examples
    --------
    >>> tensor_chain = TensorCochain({((0,),(),(1,3)):1}, degree=3, order=3,\
     n=3, dual=False)
    >>> abc(tensor_chain)
    MilnorCochain({((0,), (1,), (), (2,)): 1}, degree=3, order=3, n=3, dual=False, torsion=0)

    """

    if tensor_chain.dual or tensor_chain.sph_aug is None or \
    tensor_chain.sph_aug is False:
        raise ValueError("Only defined for chains with sph_aug=True")

    n = tensor_chain.n
    r = tensor_chain.order
    D = tensor_chain.__dict__.copy()
    del D['sph_aug']
    sol = MilnorCochain(**D)

    if rule is None:
        rule = sol.default_rule()

    for tensor, a in tensor_chain.items():
        new_tensor, new_a = [[] for _ in range(n+1)], a
        signs = dict({w:0 for w in range(n+1)})
        par = 0
        for j, factor in enumerate(tensor):
            for v in factor:
                new_tensor[v].append(j)
                for w in range(v):
                    signs[w] += 1
                par += signs[v]
        new_a *= (-1)**par
        new_tensor = tuple((tuple(factor) for factor in new_tensor))

        #Step 1: the endomorphism beta, that applies rho^{-i} to the i-th factor.
        new_new_tensor = []
        for j, new_factor in enumerate(new_tensor):
            new_new_factor, new_b = Simp(new_factor,r-1,dual=False).action(rule[j])
            new_new_tensor.append(new_new_factor)
            new_a *= new_b
        new_new_tensor = tuple((tuple(factor) for factor in new_new_tensor))
        sol += sol.create({new_new_tensor: new_a})
    return sol


