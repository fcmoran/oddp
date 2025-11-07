"""
One of the ingredients to build the chain map ``chain_maps.psi`` is a product
on the chains of the shifted augmented periodic resolution of the cyclic group.
These chains can be identified with the negative part of the Tate resolution
of the cyclic group associated to the periodic resolution. Thus, we have an
explicit description of a chain refinement of the cup product on the negative
Tate cohomology of the cyclic group.

In order to define this product, an `r`-cyclic straightening with duality must
be chosen. The function ``tate_product.straightening_builder(r)`` builds one
of these straightenings. The functions ``tate_product.tate_product`` and
``tate_product.tate_product_dual`` implement the product on the chains
of the shifted augmented periodic resolution of the cyclic group and its dual.

"""

from itertools import combinations, permutations

from oddp.objects import MilnorCochain, Simp, ParametricCounter, PerCochain
from oddp._utils import sign_perm, sign_reorder


def straightening_builder(r):
    """Builds a cyclic straightening with duality of order r

    Parameters
    ----------
    r: int
        the order of the straightening, must be a prime number.

    Returns
    ----------
    dict
        the collection of faces of the (r-1)-simplex whose straightening is
        the vertex 0 (useful for visualization purposes)

    dict
        a dictionary whose keys are the faces of the (r-1)-simplex and whose
        values are the values of the straightening at those faces.

    Examples
    ----------
    >>> short_straightening, straightening = straightening_builder(3)
    >>> short_straightening
    {(0,): 0, (0, 1): 0}
    >>> straightening
    {(0,): 0, (1,): 1, (2,): 2, (0, 1): 0, (0, 2): 2, (1, 2): 1}
    """

    short_sol, long_sol = {}, {} #{(0,): 0}, {(j,): j for j in range(r)}
    for m in range(1, r):
        for face in combinations(range(r), m):
            # we check whether A is already in short_sol up to rotation
            for i in range(r):
                new_face, a = Simp(face, r - 1).action(i)
                if new_face in short_sol.keys():
                    long_sol[face] = (short_sol[new_face] + i) % r
                    break
            else:  # if it is not in short_sol, we assign it a value.
                new_face, a = Simp(face, r - 1).action(face[0])
                for i in range(len(new_face)):
                    if new_face[i - 1] % r != new_face[i] - 1:
                        short_sol[new_face] = new_face[i]
                        long_sol[face] = (new_face[i] + face[0]) % r
                        break
    return short_sol, long_sol


def codegeneracy(straightening, r, dual=True):
    """The value of the (dual) of the weak codegeneracy on generators.

    Parameters
    ----------
    straightening: dict
        A cyclic straightening with duality of order r.
    r: int
        The order of the cyclic group, must be a prime number.
    dual: bool, default=True

    Returns
    ----------
    dict[tuple, ParametricCounter]
        A dictionary whose keys are the faces of the (r-1)-simplex and whose
        values are their images under the dual of the weak codegeneracy.

    Examples
    ----------
    >>> short_straightening, straightening = straightening_builder(3)
    >>> codegeneracy(straightening, 3)[(0,)]
    ParametricCounter({((0,), (1, 2)): 1, ((0, 1), (2,)): 1})
    >>> codegeneracy(straightening, 3)[(0,2)]
    ParametricCounter({((0, 2), (1, 2)): 1})
    """
    sol_dual = ParametricCounter()
    sol = ParametricCounter()

    # We will find the value of codegeneracy on elements of the pair
    # subdivision

    # PART 1: listing all the generators of the pair subdivision (omega, tau),
    # with omega a tuple that contains the tuple tau
    for k in range(1, r):
        for m in range(1, k + 1):
            for omega in combinations(range(r), k):
                for tau in combinations(omega, m):
                    nu = sorted(tuple(set(omega).difference(set(tau))))
                    s1 = sign_reorder(nu, tau) * (-1) ** m

                    # PART 2: listing all generators of the barycentric
                    # subdivision that are summands in the image by s^P of
                    # the pair subdivision element (omega, tau). They are
                    # indexed by permutations.
                    for pi_nu in permutations(nu):
                        s2 = sign_perm(pi_nu)

                        # bar_subdiv_vertex will hold the different vertices of
                        # the barycentric subdivision generator indexed by
                        # (omega, tau) and the permutation pi_nu.
                        bar_subdivision_vertex = list(tau)

                        # PART 3: simplex_face is the face of the (r-1)-simplex
                        # that is the image under the assemblage map of
                        # the barycentric subdivision element indexed by the
                        # (omega, tau) and the permutation pi_nu.
                        simplex_face = [straightening[
                                            tuple(bar_subdivision_vertex)]]
                        for vertex in pi_nu:
                            bar_subdivision_vertex.append(vertex)
                            simplex_vertex = straightening[
                                tuple(sorted(bar_subdivision_vertex))]
                            if simplex_vertex not in simplex_face:
                                simplex_face.append(simplex_vertex)
                            else:
                                break
                        else:
                            s3 = sign_perm(simplex_face)
                            simplex_face = tuple(sorted(simplex_face))

                            # PART 0: All generators of N(Delta^{r-1})^{ot 2}
                            # that map to (omega, tau) under id ot Lambda.
                            new_tau, s4 = Simp(tuple(tau), r - 1).alex()

                            # the Koszul sign of moving Lambda over omega.
                            s5 = (-1) ** len(omega)
                            if simplex_face in sol_dual.keys():
                                sol_dual[simplex_face] += ParametricCounter(
                                    {(omega, new_tau): s1 * s2 * s3 * s4 * s5}
                                )
                            else:
                                sol_dual[simplex_face] = ParametricCounter(
                                    {(omega, new_tau): s1 * s2 * s3 * s4 * s5}
                                )
                            if (omega, new_tau) in sol.keys():
                                sol[(omega, new_tau)] += ParametricCounter({simplex_face: s1 * s2 * s3 * s4 * s5})
                            else:
                                sol[(omega, new_tau)] = ParametricCounter(
                                    {simplex_face: s1 * s2 * s3 * s4 * s5})

    if dual:
        return sol_dual
    else:
        return sol

def tate_product_dual(per_cochain, codegeneracy_dict):
    """The dual of the weak codegeneracy.

    Parameters
    ----------
    per_cochain: PerCochain
        A cochain of degree q in the Periodic resolution of the cyclic group
        with  2 <= q <= 2(r-1). The order of the cyclic group
        ``per_cochain.order`` must be a prime number.
    codegeneracy_dict: dict[tuple, ParametricCounter]
        A dictionary encoding the values of this map on generators (obtained
        via the function codegeneracy).

    Returns
    ----------
    MilnorCochain
    """
    q, r, t =  -per_cochain.degree,per_cochain.order, per_cochain.torsion
    if q > 2 * r - 2:
        raise ValueError(
            f'codegeneracy is only defined in degrees 2 <= q <= 2(r-1) = '
            f'{2 * r - 2}. In this case, q = {q}')
    sol = MilnorCochain(degree=-q, order=r, torsion=t, n=1)
    if q < r:  # in this case, we apply the dual of the join map
        for per_element, a in per_cochain.items():
            for m in range(q + 1):
                for first_factor in combinations(per_element, m):
                    second_factor = tuple(
                        set(per_element).difference(first_factor))
                    milnor_element = (first_factor, second_factor)
                    s = sign_reorder(first_factor, second_factor)
                    sol += sol.create({milnor_element: s * a})
    else:  # in this case, we apply the dual of the codegeneracy.
        for per_element, a in per_cochain.items():
            for milnor_element, s in codegeneracy_dict[per_element].items():
                sol += sol.create({milnor_element: s * a})
    return sol

def tate_product(per_chain_1, per_chain_2, codegeneracy_dict):
    """The weak codegeneracy.

    Parameters
    ----------
    per_chain_1: PerCochain(dual=False)
        A chain in the shifted augmented periodic resolution of the cyclic
        group. The order of the cyclic group ``per_chain_1.order`` must be a
        prime number.
    per_chain_2: PerCochain(dual=False)
        A chain in the shifted augmented periodic resolution of the cyclic
        group. The order of the cyclic group ``per_chain_2.order`` must be a
        prime number.
    codegeneracy_dict: dict[tuple, ParametricCounter]
        A dictionary encoding the values of this map on generators (obtained
        via the function codegeneracy(dual=False)).

    Returns
    ----------
    PerCochain(dual=False)
    """
    kwargs1 = per_chain_1.__dict__.copy()
    kwargs2 = per_chain_2.__dict__.copy()
    q = per_chain_1.degree + per_chain_2.degree
    kwargs1['degree'] = q
    kwargs2['degree'] = q
    if kwargs1 != kwargs2 or per_chain_1.dual:
        raise ValueError()
    sol = PerCochain({},**kwargs1)
    for per_element_1, a_1 in per_chain_1.items():
        for per_element_2, a_2 in per_chain_2.items():
            if len(per_element_1) + len(per_element_2) < sol.order:  # in this case, we apply the dual of the join map
                per_element = tuple(sorted(per_element_1 + per_element_2))
                sign = sign_reorder(per_element_1, per_element_2)
                if len(per_element_1) + len(per_element_2) == len(set(per_element)):
                    sol += sol.create({per_element:sign * a_1 * a_2})
            else:
                per_chain = codegeneracy_dict.get((per_element_1, per_element_2))
                if per_chain is not None:
                    sol += a_1 * a_2 * sol.create(per_chain)
    return sol
