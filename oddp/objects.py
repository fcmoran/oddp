"""
This module implements chains and cochains in three shifted augmented
resolutions of the cyclic group: The minimal resolution, the periodic
resolution and the Milnor resolution. The latter is isomorphic to a subcomplex
(or a quotient) of an iterated tensor product of the chain complex of the
standard simplex that is also implemented.

Each of them is implemented as a subclass of a class **ParametricCounter**
that inherits from the class dictionary: the keys of
the dictionary are basis elements of these resolutions and the values are their
coefficients. Each of these subclasses has at least the following attributes:

:degree: the degree of the chain or cochain.

:order: the order of the cyclic group.

:dual: whether it is a cochain or a chain. Default, True.

:torsion: if nonzero and the values of the dictionary are integers, then these
    values are treated as modular integers. Default, 0.

Each of these subclasses has at least the following methods:

:action(): the action of the cyclic group on the chain or cochain.

:partial(): the image of the chain or cochain by the differential.

together with a static method ``_examples()`` that provides instances of the
classes useful for testing purposes.
"""

from random import randint, sample
from collections import deque, Counter
from itertools import combinations

from oddp._utils import sign_complement


class ParametricCounter(Counter):
    """
    A linear combination of multigraded elements.

    Parameters
    ----------
    data: Counter
    **kwargs: dict
        The keys are the different gradings, the values are the value of
        these gradings. If the values of the Counter data are integers, then
        an additional argument **torsion** may be included in ****kwargs**
        to make the linear combination over the modular integers of order
        **torsion**.

    Notes
    ----------
    This class implements a multiset (linear combination) with coefficients
    in an arbitrary ring together with parameters. It consists of a dictionary
    with values in some ring, together with a couple of parameters. Two such
    dictionaries with parameters can be added, subtracted and right-multiplied
    by an element of the ring whenever the parameters are equal. Keys with zero
    value are deleted from the dictionary.

    Examples
    ----------
    >>> x = ParametricCounter({'a':1, 'b':2}, q = 2, r = 3)
    >>> y = ParametricCounter({'a':1, 'b':-1, 'c':-1}, q = 2, r = 3)
    >>> print(x + 2*y)
    ParametricCounter({'a': 3, 'c': -2}, q=2, r=3)

    """

    def __init__(self, data=None, **kwargs):
        """Creates a new instance of ParametricCounter.

        :param data: a dictionary or iterable
        :param kwargs: an arbitrary number of keyword arguments
        """
        super(ParametricCounter, self).__init__(data)
        self.__dict__.update(kwargs)
        self._valid()

    # def __str__(self):
    #     sol = dict.__str__(self)[:-1]
    #     for key, value in self.__dict__.items():
    #         sol += ', ' + str(key) + '=' + str(value)
    #     sol += ')'
    #     return sol

    def __repr__(self):
        sol = super().__repr__()[:-1]

        for key, value in self.__dict__.items():
            sol += f', ' + f'{key}' + f'=' + f'{value}'
        sol += f')'
        return sol

    def __add__(self, other):
        """Adds self and other if they share the same parameters. It is also
        implemented if other is a dictionary or a Counter, in which case the
        result is endowed with the parameters of self.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        if self.__dict__ == other.__dict__:
            data = {}
            for key, value in self.items():
                other_value = other.get(key)
                if other_value is None:
                    data[key] = value
                elif torsion and (self[key] + other_value) % torsion:
                    data[key] = (self[key] + other_value) % torsion
                elif not torsion and self[key] + other_value:
                    data[key] = self[key] + other_value
            for key, value in other.items():
                if key not in self.keys():
                    data[key] = value
            kwargs = dict(self.__dict__)
            return type(self)(data, **kwargs)
        else:
            raise ValueError('Cannot add elements with different attributes'
                             f'{self} and {other}')

    def __sub__(self, other):
        """Subtracts other from self if they share the same parameters. It is
        also implemented if other is a dictionary or a Counter, in which case
        the result is endowed with the parameters of self.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        if self.__dict__ == other.__dict__ or isinstance(other, dict):
            data = {}
            for key, value in self.items():
                other_value = other.get(key)
                if other_value is None:
                    data[key] = value
                elif torsion and (self[key] - other_value) % torsion:
                    data[key] = (self[key] - other_value) % torsion
                elif not torsion and self[key] - other_value:
                    data[key] = self[key] - other_value
            for key, value in other.items():
                if key not in self.keys():
                    data[key] = -value
            kwargs = dict(self.__dict__)
            return type(self)(data, **kwargs)
        else:
            raise ValueError('Cannot add elements with different attributes')

    def __iadd__(self, other):
        """
        In place addition of self and other.
        :param other: A ParametricCounter, a Counter or a dictionary
        :return: A ParametricCounter with the same parameters as self.
        if other is another ParametricCounter, then it is only defined
        when self and other have the same parameters.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        if self.__dict__ == other.__dict__ or isinstance(other, dict):
            for key, value in other.items():
                self_value = self.get(key)
                if self_value is None:
                    self[key] = other[key]
                elif torsion and (other[key] + self_value) % torsion:
                    self[key] = (self[key] + other[key]) % torsion
                elif (not torsion) and (other[key] + self_value):
                    self[key] += other[key]
                else:
                    del self[key]
            return self
        else:
            raise ValueError('Cannot add elements with different attributes')

    def __isub__(self, other):
        """
        In place subtraction of self and other.
        :param other: A ParametricCounter, a Counter or a dictionary
        :return: A ParametricCounter with the same parameters as self.
        if other is another ParametricCounter, then it is only defined
        when self and other have the same parameters.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        if self.__dict__ == other.__dict__ or isinstance(other, dict):
            for key, value in other.items():
                self_value = self.get(key)
                if self_value is None:
                    self[key] = other[key]
                elif torsion and (self_value - other[key]) % torsion:
                    self[key] = (self_value - other[key]) % torsion
                elif not torsion and self_value - other[key]:
                    self[key] -= other[key]
                else:
                    del self[key]
            return self
        else:
            raise ValueError('Cannot add elements with different attributes')

    def __neg__(self):
        """The negation of self.
        :return: A ParametricCounter.
        """
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        data = {}
        for key, value in self.items():
            if torsion:
                data[key] = (-value) % torsion
            else:
                data[key] = -value
        kwargs = dict(self.__dict__)
        return type(self)(data, **kwargs)

    def __rmul__(self, other):
        """

        :param other: anything that can be multiplied by the coefficients of
        self.
        :return:
        """
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        data = {}
        if other:
            for key, value in self.items():
                if torsion and (other * value) % torsion:
                    data[key] = (other * value) % torsion
                elif not torsion and other * value:
                    data[key] = other * value
        kwargs = dict(self.__dict__)
        return type(self)(data, **kwargs)

    def __truediv__(self, other):
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        data = {}
        if other:
            for key, value in self.items():
                if torsion:
                    data[key] = (value * pow(other, -1, torsion)) % torsion
                elif not torsion:
                    data[key] = value / other
        else:
            raise ZeroDivisionError('Cannot divide by zero')
        kwargs = dict(self.__dict__)
        return type(self)(data, **kwargs)

    # def update(self, other):
    #     if not isinstance(other, dict):
    #         return NotImplemented
    #     if hasattr(self, 'torsion'):
    #         torsion = self.torsion
    #     else:
    #         torsion = 0
    #     for key, value in other.items():
    #         self_value = self.get(key)
    #         if self_value is None:
    #             self[key] = other[key]
    #         elif torsion and (other[key] + self_value) % torsion:
    #             self[key] = (self[key] + other[key]) % torsion
    #         elif (not torsion) and (other[key] + self_value):
    #             self[key] += other[key]
    #         else:
    #             del self[key]
    #     return self

    def _valid(self):
        if hasattr(self, 'torsion'):
            torsion = self.torsion
        else:
            torsion = 0
        if torsion:
            zeros = []
            for key in self.keys():
                if self[key] % torsion:
                    self[key] = self[key] % torsion
                else:
                    zeros.append(key)
        else:
            zeros = [key for key in self.keys() if self[key] == 0]
        for key in zeros:
            del self[key]

    def create(self, data=None, **kwargs):
        """
        Creates an instance with the same parameters as self.

        Parameters
        ----------
        data: dict, default None
            If None, an empty dictionary is used.
        kwargs: dict
            These attributes are added or overwritten
            over the attributes of self.

        Returns
        -------
        ParametricCounter

        Examples
        --------
        >>> x = ParametricCounter({3: 4},q=3 ,p=5)
        >>> y = x.create(p=7, r=0)
        >>> print(y)
        ParametricCounter(, q=3, p=7, r=0)
        >>> y = x.create({1: -2}, p=7, r=0)
        >>> print(y)
        ParametricCounter({1: -2}, q=3, p=7, r=0)

        :meta private:
        """

        new_args = self.__dict__.copy()
        for key, value in kwargs.items():
            new_args[key] = value
        return type(self)(data, **new_args)

    def copy(self):
        """
        :meta private:
        """
        kwargs = self.__dict__.copy()
        return type(self)(self, **kwargs)

    # def get_ring_type(self):
    #     if self:
    #         return next(iter(self.values()))
    #     else:
    #         return None


class MinimalCochain(ParametricCounter):
    r"""
    A (co)chain of the shifted augmented minimal resolution of the cyclic
    group.

    Attributes
    ----------
    degree: int
        The degree of the cochain (nonpositive if ``dual=True``, nonnegative
        if ``dual=False``).
    order: int
        The order of the cyclic group.
    dual: bool, default=True
        If ``dual=True``, cochains are being implemented, if ``dual=False``,
        chains are being implemented.
    torsion: int, default=0.
        If the values of the dictionary data are integers, then they are
        interpreted as modular integers with modulus=torsion.

    Notes
    -----
    Let :math:`R` be a commutative ring with unit, such as
    :math:`\mathbb Z` or :math:`\mathbb Z/r Z`. The shifted augmented minimal
    resolution of the cyclic group :math:`\mathbb C_r` of order `r` is the
    chain complex

    .. math::
        R \leftarrow R\mathbb C_r\langle e_1\rangle \leftarrow
        R\mathbb C_r\langle e_2\rangle \leftarrow R\mathbb C_r\langle
        e_3\rangle
        \ldots

    concentrated in nonnegative degrees. The differential is given by

    .. math:
        \partial(e_{2k}) = \rho(e_{2k-1})-e_{2k-1}
        \partial(e_{2k+1}) = (1 + \rho + \rho^2 + \ldots + \rho^{r-1})(e_{2k})
        \partial(e_{1}) = 1.

    where :math:`\rho` denotes the standard generator of the cyclic group.
    Instances of this class are elements of this resolution or its dual.
    For example, the cochain :math:`e_3^\vee + 4\cdot (\rho^2e_3)^\vee`
    in the shifted augmented minimal resolution of the cyclic group of order 5
    with integer coefficients is created as

    .. code-block:: python

        MinimalCochain({0:1, 2:4}, degree=-3, order=5)

    while the chain :math:`2\cdot e_3 + \rho^4e_3` in the shifted augmented
    minimal resolution of the cyclic group of order 5 with mod 5 coefficients
    is created as

    .. code-block:: python

        MinimalCochain({0:2, 4:1}, degree=3, order=5, dual=False, torsion=5)

    To create chains or cochains in degree 0 use the key 0:

    .. code-block:: python

        MinimalCochain({0:3}, degree=0, order=5)

    Observe in the first example that
    :math:`\rho(e_k^\vee) = (\rho^{-1}e_k)^\vee`

    References
    ----------
    [1] Cantero-Morán, F., Medina-Mardones, A., Steenrod operations and Tate
    resolutions.

    """

    def __init__(self, data=None, degree=None, order=3, dual=True, torsion=0):
        kwargs = {'degree': degree, 'order': order, 'dual': dual,
                  'torsion': torsion}
        super().__init__(data, **kwargs)
        self.degree = degree
        self.order = order
        self.dual = dual
        self.torsion = torsion

    def action(self, i=1):
        r"""
        The action of a generator of the cyclic group on self.

        Parameters
        ----------
        i: int, default=1
            The generator that is acting.

        Returns
        -------
        MinimalCochain

        Notes
        -----
        The action of the generator :math:`\rho^i` of the cyclic group on
        self.

        Examples
        --------
        >>> cochain = MinimalCochain({0:1}, degree=-2, order=3, dual=True, torsion=0)
        >>> cochain.action()
        MinimalCochain({2: 1}, degree=-2, order=3, dual=True, torsion=0)
        >>> chain = MinimalCochain({1: 1}, degree=2, order=3, dual=False, torsion=0)
        >>> chain.action()
        MinimalCochain({2: 1}, degree=2, order=3, dual=False, torsion=0)

        """

        if self.degree == 0:
            return self.create({0: c for j, c in self.items()})
        else:
            new_i = (-1) ** self.dual * i
            return self.create({(j + new_i) % self.order: c
                                for j, c in self.items()})

    def partial(self):
        """The differential of self.

        Returns
        -------
        MinimalCochain

        Examples
        --------
        >>> cochain = MinimalCochain({0:1}, degree=-2, order=3, dual=True, torsion=0)
        >>> cochain.partial()
        MinimalCochain({0: -1, 1: -1, 2: -1}, degree=-3, order=3, dual=True, torsion=0)
        >>> chain = MinimalCochain({1: 1}, degree=2, order=3, dual=False, torsion=0)
        >>> chain.partial()
        MinimalCochain({2: 1, 1: -1}, degree=1, order=3, dual=False, torsion=0)

        :meta public:
        """

        if self.dual:
            if self.degree > 0:
                sol = self.create(degree=self.degree - 1)
            elif self.degree % 2 == 0:
                a = (-1) ** (-self.degree - 1) * self.total()
                data = {j: a for j in range(self.order)}
                sol = self.create(data, degree=self.degree - 1)
            else:
                sol = (-1) ** (-self.degree - 1) * (self.action() - self)
                sol.degree = self.degree - 1
        else:
            if self.degree <= 0:
                sol = self.create(degree=self.degree - 1)
            elif self.degree % 2 == 0:
                sol = self.action() - self
                sol.degree = sol.degree - 1
            elif self.degree == 1:
                a = self.total()
                data = {0: a}
                sol = self.create(data, degree=0)
            else:
                a = self.total()
                data = {j: a for j in range(self.order)}
                sol = self.create(data, degree=self.degree - 1)
        return sol

    def _valid(self):
        """ Checks whether self is a valid Minimal Cochain.

        Returns
        -------
        None

        """

        ParametricCounter._valid(self)
        if self.degree * (1 - 2 * self.dual) < 0 and len(self) != 0:
            raise ValueError(f'The chain complex is trivial in this degree:'
                             f' {self}')
        elif self.degree == 0 and tuple(self.keys()) not in [(), (0,)]:
            raise ValueError(f'The only generator in degree 0 is 0: {self}')
        else:
            if not set(self.keys()).issubset(set(range(self.order))):
                raise ValueError(f'Not a valid element: {self}')

    @staticmethod
    def _examples(order_tuple=(3, 5, 7)):
        for r in order_tuple:
            yield MinimalCochain({0: 1}, 0, r)
            yield MinimalCochain({0: 1}, 0, r, False)
            for q in range(1, 2 * r):
                for n in range(2 * r):
                    for j in range(r):
                        for dual in [True, False]:
                            yield MinimalCochain({j: 1}, (-1) ** dual * q, r,
                                                 dual)


class Simp:
    """A generator of the group of cochains of the standard simplex of
    dimension n.

    :meta private:
    """

    def __init__(self, face, dim, dual=True, sphere=True, augmented=True):
        self.face = face
        self.dim = dim
        self.dual = dual
        self.sphere = sphere
        self.augmented = augmented

    def action(self, i=1):
        r = self.dim + 1
        i = i % r
        if self.dual:
            first = tuple((a - i for a in self.face if a - i >= 0))
            second = tuple((a - i + r for a in self.face if a - i < 0))
        else:
            second = tuple((a + i for a in self.face if a + i < r))
            first = tuple((a + i - r for a in self.face if a + i >= r))
        return first + second, (-1) ** (len(first) * len(second))

    def partial(self):
        """Dual of the boundary of a coface A in the standard (r-1)-simplex,
        output as an iterable. It does not have the sign (-1)** (deg + 1)

        :meta private:
        """
        face = self.face
        if self.dual:
            if len(face) == 0 and self.augmented:
                for v in range(self.dim + 1):
                    yield (v,), 1
            elif len(face) == self.dim and self.sphere:
                pass
            else:
                for v in range(face[0]):
                    yield (v,) + face, 1
                for i in range(1, len(face)):
                    for v in range(face[i - 1] + 1, face[i]):
                        yield face[: i] + (v,) + face[i:], (-1) ** i
                for v in range(face[-1] + 1, self.dim + 1):
                    yield face + (v,), (-1) ** (len(face))
        else:
            if len(face) > 1 or self.augmented:
                for i, v in enumerate(face):
                    yield face[:i] + face[i + 1:], (-1) ** i

    def alex(self):
        """
        The Alexander dual of tau in Delta n. If ``dual=True``, the source are
        cochains, otherwise chains.

        :meta private:
        """
        new_face = (v for v in range(self.dim + 1) if v not in self.face)
        new_face = tuple(new_face)
        new_a = sign_complement(range(self.dim + 1), self.face, self.dual)
        return new_face, new_a

    def p_sigma(self):
        dim = self.dim
        for i in range(dim + 1):
            yield tuple(range(i)) + tuple(range(i + 1, dim + 1)), (-1) ** i

    @staticmethod
    def _examples():
        for n in range(4):
            for m in range(n + 1):
                for face in combinations(range(n + 1), m):
                    for dual in [True, False]:
                        yield face, n, dual


class PerCochain(ParametricCounter):
    r"""
    A (co)chain of the shifted augmented periodic resolution of the cyclic
    group.

    Attributes
    ----------
    degree: int
        The degree of the cochain (nonpositive if ``dual=True``, nonnegative
        if ``dual=False``).
    order: int
        The order of the cyclic group.
    dual: bool, default=True
        If ``dual=True``, cochains are being implemented, if ``dual=False``,
        chains
        are being implemented.
    torsion: int, default=0.
        If the values of the dictionary data are integers, then they are
        interpreted as modular integers with modulus=torsion. If ``order``
        is the even prime, then modular coefficients with ``torsion=order``
        must be used.

    Notes
    -----
    Let :math:`R` be a commutative ring with unit, such as
    :math:`\mathbb Z` or :math:`\mathbb Z/r \mathbb Z`. Let
    :math:`\mathrm{N}(\partial\Delta^{r-1})` be the
    normalised chain complex of the boundary of the standard simplex of
    dimension `r-1`. Permuting the vertices
    of the simplex defines an action of the cyclic group :math:`\mathbb C_r`
    of order `r` on this chain complex.

    This complex has an augmentation map of degree `0` that sends every vertex
    of the simplex to the unit in the ring `R`. It also has a coaugmentation
    map of degree `r-2` that sends the unit to the boundary of the top face of
    :math:`\Delta^{r-1}` with its standard orientation. The composition of
    these two maps defines a differential of degree `r-2` with which
    :math:`\bigoplus_{j=0}^{\infty} \mathrm{s}^{j(r-1)}\mathrm{N}(
    \partial\Delta^{r-1})`
    becomes a contractible chain complex with an augmentation to `R`.

    The suspension of this augmented chain complex is the shifted augmented
    Periodic resolution of the cyclic group of order `r`.

    Instances of this class are elements of this resolution. For example,
    the generator :math:`2\cdot (1,3) + 3\cdot (0,4)` in degree 6 in the
    shifted augmented periodic resolution of the cyclic group of order 5
    with integer coefficients is created as

    .. code-block:: python

        PerCochain({(1,3):2, (0,4):3}, degree=6, order=5, dual=False)

    .. Warning::
        The faces of the simplex (for example, (1,3) or (0,4)) are
        always assumed to be ordered.

    """

    def __init__(self, data, degree, order, dual=True, torsion=0):
        kwargs = {'degree': degree, 'order': order, 'dual': dual,
                  'torsion': torsion}
        super().__init__(data, **kwargs)
        self.degree = degree
        self.order = order
        self.dual = dual
        self.torsion = torsion

    def action(self, i=1):
        """
        The action of a generator of the cyclic group on self.

        Parameters
        ----------
        i: int, default=1
            The generator of the cyclic group that is acting.

        Returns
        -------
        PerCochain

        Examples
        --------
        >>> cochain = PerCochain({(0, 2): 1}, degree=-2, order=3, dual=True, torsion=0)
        >>> cochain.action()
        PerCochain({(1, 2): -1}, degree=-2, order=3, dual=True, torsion=0)
        >>> chain = PerCochain({(0, 2): 1}, degree=2, order=3, dual=False, torsion=0)
        >>> chain.action()
        PerCochain({(0, 1): -1}, degree=2, order=3, dual=False, torsion=0)

        """
        sol = self.create()
        for face, a in self.items():
            new_face, b = Simp(face, self.order - 1, self.dual).action(i)
            sol += sol.create({new_face: a * b})
        return sol

    def partial(self):
        """
        The differential of self.

        Returns
        -------
        PerCochain

        Examples
        --------
        >>> cochain = PerCochain({(0, 2): 1}, degree=-2, order=3, dual=True, torsion=0)
        >>> cochain.partial()
        PerCochain({(0,): 1, (1,): 1, (2,): 1}, degree=-3, order=3, dual=True, torsion=0)
        >>> chain = PerCochain({(0, 2): 1}, degree=2, order=3, dual=False, torsion=0)
        >>> chain.partial()
        PerCochain({(2,): 1, (0,): -1}, degree=1, order=3, dual=False, torsion=0)

        """
        q, r, dual = self.degree, self.order, self.dual
        sol = self.create(degree=self.degree - 1)
        dual_sign = (-1) ** ((-q + 1) * dual)
        for face, a in self.items():
            if dual and q < 0 and q % (r - 1) == 0:
                j = tuple(filter(lambda k: k not in face, range(r)))
                sol += sol.create(
                    {(v,): a * (-1) ** j[0] * dual_sign for v in range(r)})
            elif not dual and q > 1 and q % (r - 1) == 1:
                sol = self.total() * self.p_sigma(self.degree - 1)
            else:
                for new_face, b in Simp(face, r - 1, dual).partial():
                    sol += sol.create({new_face: a * b * dual_sign})
        return sol

    def _valid(self):
        ParametricCounter._valid(self)
        r, q = self.order, (-1) ** self.dual * self.degree
        qq = q % (r - 1) if (q % (r - 1) != 0 or q == 0) else r - 1
        for face in self.keys():
            if qq != len(face):
                raise ValueError(
                    f'the degree {q} and the length of {face} do not match.')
            if tuple(sorted(face)) != face:
                raise ValueError(f'All faces must be sorted: {self}')
            if not set(face).issubset(set(range(r))):
                raise ValueError(
                    f'All faces must be subfaces of the r-1 simplex: {self}')

    def p_sigma(self, degree):
        """
        The (dual of the) differential of the top face of the (r-1)-simplex.

        Parameters
        ----------
        degree: int
            A multiple of **self.order** - 1.

        Returns
        -------
        PerCochain

        Examples
        --------
        >>> cochain = PerCochain({(0, 2): 1}, degree=-2, order=3, dual=True, torsion=0)
        >>> cochain.p_sigma(-6)
        PerCochain({(1, 2): 1, (0, 1): 1, (0, 2): -1}, degree=-6, order=3, dual=True, torsion=0)
        >>> chain = PerCochain({(0, 2): 1}, degree=2, order=3, dual=False, torsion=0)
        >>> chain.p_sigma(4)
        PerCochain({(1, 2): 1, (0, 1): 1, (0, 2): -1}, degree=4, order=3, dual=False, torsion=0)

        """

        if degree % (self.order - 1) != 0 or degree == 0:
            raise ValueError('Degree must be a multiple of (order - 1)')
        sol = self.create(degree=degree)
        for face, a in Simp((), self.order - 1).p_sigma():
            sol += self.create({face: a}, degree=degree)
        return sol

    @staticmethod
    def _examples(order_tuple=(3, 5, 7)):
        for r in order_tuple:
            for q in range(r + 5):
                for dual in [True, False]:
                    if q == 0:
                        yield PerCochain({(): 1}, (-1) ** dual * q, r, dual)
                    else:
                        m = q % (r - 1)
                        if m == 0: m = r - 1
                        face = tuple(sorted(sample(range(r), m)))
                        yield PerCochain({face: 1}, (-1) ** dual * q, r, dual)


class MilnorCochain(ParametricCounter):
    r"""
    A (co)chain of the shifted augmented Milnor resolution of the cyclic group.

    Attributes
    ----------
    degree: int
        The degree of the cochain (nonpositive if ``dual=True``, nonnegative
        if ``dual=False``).
    order: int
        The order of the cyclic group.
    n: int
        n+1 is the number of tensor factors in the resolution.
    dual: bool, default=True
        If ``dual=True``, cochains are being implemented, if ``dual=False``,
        chains
        are being implemented.
    torsion: int, default=0.
        If the values of the dictionary data are integers, then they are
        interpreted as modular integers with modulus=torsion.

    Notes
    -----
    Let :math:`R` be a commutative ring with unit, such as
    :math:`\mathbb Z` or :math:`\mathbb Z/r \mathbb Z`. Let
    :math:`\mathrm{N}(\partial\Delta^{r-1})` be the
    normalised chain complex of the boundary of the standard simplex of
    dimension `r-1`. Permuting the vertices of the simplex defines an action
    of the cyclic group :math:`\mathbb C_r` of order `r` on this chain complex.

    The Milnor resolution (of size `n`) of the cyclic group of order `r` is
    the tensor product :math:`\mathrm{N}(\partial \Delta_+^{r-1})^{\otimes
    n+1}`,
    with its natural augmentation over the ring `R`, and suspended once.

    Instances of this class are elements of this resolution.

    .. Warning::
        The faces of the simplex (for example, (1,3) or (0,4)) are
        always assumed to be ordered.

    """

    def __init__(self, data=None, degree=0, order=3, n=0, dual=True,
                 torsion=0):
        kwargs = {'degree': degree, 'order': order, 'n': n, 'dual': dual,
                  'torsion': torsion}
        super().__init__(data, **kwargs)
        self.degree = degree
        self.order = order
        self.dual = dual
        self.torsion = torsion
        self.n = n

    def default_rule(self):
        r"""
        A tuple that encodes the default endomorphism of the Milnor resolution.

        This tuple encodes the endomorphism :math:`\alpha = (\rho^{r}\otimes
        \rho^{r-1}\otimes \ldots \otimes \rho^{r-n})` of the Milnor resolution.
        The composition :math:`\alpha\circ\beta` is implemented in the function
        ``chain_maps.abc``.

        Returns
        -------
        tuple

        """
        return tuple((-j % self.order for j in range(self.n + 1)))

    def inner_action(self, rule=None):
        """
        The endomorphism of the Milnor resolution defined by `rule`.

        Parameters
        ----------
        rule: tuple[int]
            A tuple encoding **self.n** + 1 generators of the cyclic group.
            If None, defaults to *default_rule()*.

        Returns
        -------
        MilnorCochain

        """

        r, dual = self.order, self.dual
        if rule is None:
            rule = self.default_rule()
        sol = self.create()
        for tensor, a in self.items():
            new_tensor = []
            new_a = a
            for k, factor in enumerate(tensor):
                new_factor, b = Simp(factor, r - 1, dual).action(rule[k])
                new_tensor.append(new_factor)
                new_a = new_a * b
            sol += self.create({tuple(new_tensor): new_a})
        return sol

    def action(self, i=1):
        """
        The diagonal action of the cyclic group on self.

        Parameters
        ----------
        i: int
            The generator of the cyclic group that is acting on self.

        Returns
        -------
        MilnorCochain

        Examples
        --------
        >>> cochain = MilnorCochain({((0,2),(2,)):1}, degree=-3, order=3, n=1, dual=True, torsion=0)
        >>> cochain.action()
        MilnorCochain({((1, 2), (1,)): -1}, degree=-3, order=3, n=1, dual=True, torsion=0)
        >>> chain = MilnorCochain({((0,2),(2,)):1}, degree=3, order=3, n=1, dual=False, torsion=0)
        >>> chain.action()
        MilnorCochain({((0, 1), (0,)): -1}, degree=3, order=3, n=1, dual=False, torsion=0)

        """

        return self.inner_action((i,) * (self.n + 1))

    def partial(self):
        """
        The differential of self.

        Returns
        -------
        MilnorCochain

        Examples
        --------
        >>> cochain = MilnorCochain({((0,2),(2,)):1}, degree=-3, order=3, n=1, dual=True, torsion=0)
        >>> cochain.partial()
        MilnorCochain({((0, 2), (0, 2)): 1, ((0, 2), (1, 2)): 1}, degree=-4, order=3, n=1, dual=True, torsion=0)
        >>> chain = MilnorCochain({((0,2),(2,)):1}, degree=3, order=3, n=1, dual=False, torsion=0)
        >>> chain.partial()
        MilnorCochain({((2,), (2,)): 1, ((0, 2), ()): 1, ((0,), (2,)): -1}, degree=2, order=3, n=1, dual=False, torsion=0)

        """

        sol = self.create(degree=self.degree - 1)
        dual_sign = (-1) ** ((-self.degree - 1) * self.dual)
        for tensor, a in self.items():
            b = 1
            for i, factor in enumerate(tensor):
                for new_factor, c in Simp(factor, self.order - 1,
                                          self.dual).partial():
                    new_tensor = list(tensor)
                    new_tensor[i] = new_factor
                    new_tensor = tuple(new_tensor)
                    sol += sol.create({new_tensor: a * b * c * dual_sign})
                b = b * (-1) ** len(factor)
        return sol

    def p_sigma(self):
        """
        The (dual of the) differential of the top face of the (
        **self.order** - 1)-simplex.

        Returns
        -------
        MilnorCochain

        Examples
        --------
        >>> cochain = MilnorCochain({((0,2),(2,)):1}, degree=-3, order=3, n=1, dual=True, torsion=0)
        >>> cochain.p_sigma()
        MilnorCochain({((1, 2),): 1, ((0, 1),): 1, ((0, 2),): -1}, degree=2, order=3, n=0, dual=False, torsion=0)
        >>> chain = MilnorCochain({((0,2),(2,)):1}, degree=3, order=3, n=1, dual=False, torsion=0)
        >>> chain.p_sigma()
        MilnorCochain({((1, 2),): 1, ((0, 1),): 1, ((0, 2),): -1}, degree=2, order=3, n=0, dual=False, torsion=0)

        """
        sol = self.create(degree=self.order - 1, n=0, dual=False)
        for face, a in Simp((), self.order - 1).p_sigma():
            sol += sol.create({(face,): a})
        return sol

    def _valid(self):
        ParametricCounter._valid(self)
        for element in self.keys():
            if len(element) != self.n + 1:
                raise ValueError(
                    f'len of MilnorCochain does not match its n: {self}')
            elif sum((len(factor) for factor in element)) != (
            -1) ** self.dual * self.degree:
                raise ValueError(
                    f'MilnorCochain does not match its expected degree:'
                    f'{self}')
            for factor in element:
                if not set(factor).issubset(range(self.order)):
                    raise ValueError(
                        f'Some factors of this MilnorCochain are not faces '
                        f'of the simplex of dimension {self.order - 1}: '
                        f'{self}')
                if tuple(sorted(factor)) != factor:
                    raise ValueError(f'All faces must be sorted: {self}')

    def evaluate(self, other, left=False):
        """
        Evaluates the Milnor cochain **self** on the Milnor chain **other**.

        **self.dual** must be True and **other.dual** must be False.

        Parameters
        ----------
        other: MilnorCochain(dual=False)
            The Milnor chain that is being evaluated.
        left: bool
            To choose whether the Milnor chain is evaluated on the left or on
            the right.

        Returns
        -------
        MilnorCochain

        """

        if self.order != other.order or self.dual == other.dual:
            raise ValueError
        else:
            sol = self.create(degree=self.degree + other.degree,
                              n=self.n - other.n - 1)
            for tensor1, a1 in self.items():
                for tensor2, a2 in other.items():
                    if left and tensor1[:len(tensor2)] == tensor2:
                        sol += sol.create({tensor1[len(tensor2):]: a1 * a2})
                    if not left and tensor1[-len(tensor2):] == tensor2:
                        sol += sol.create({tensor1[:len(tensor2)]: a1 * a2})
            return sol

    @staticmethod
    def _examples(order_tuple=(3, 5, 7)):
        for r in order_tuple:
            for dual in [True, False]:
                for n in range(2 * r):
                    element = ((),) * (n + 1)
                    yield MilnorCochain({element: 1}, 0, r, n, dual)
                for n in range(2 * r):
                    for j in range(n):
                        a = randint(0, r - 1)
                        element = ((),) * j + ((a,),) + ((),) * (n - j)
                        yield MilnorCochain({element: 1}, (-1) ** dual, r, n,
                                            dual)
                for j in range(5):
                    # r = choice(rs)
                    q = 0
                    n = randint(1, r + 1)
                    element = []
                    for l in range(n + 1):  # number of joined simplices
                        factor = tuple(
                            sorted(sample(range(r), randint(0, r - 1))))
                        element.append(factor)
                        q += len(factor)
                    yield MilnorCochain({tuple(element): 1}, (-1) ** dual * q,
                                        r,
                                        n, dual)


class TensorCochain(ParametricCounter):
    r"""
    A cochain (or a chain) in :math:`\mathrm{N}(\Delta_+^{n})^{\otimes r}`.

    Attributes
    ----------
    degree: int
        The degree of the cochain (nonpositive if ``dual=True``, nonnegative
        if ``dual=False``).
    order: int
        The number of tensor factors.
    n: int
        The dimension of the simplex.
    dual: bool, default=True
        If ``dual=True``, cochains are being implemented, if ``dual=False``,
        chains
        are being implemented.
    torsion: int, default=0.
        If the values of the dictionary data are integers, then they are
        interpreted as modular integers with modulus=torsion.
    sph_aug: bool or None, default=True
        See the notes below.

    Notes
    -----
    Let :math:`R` be a commutative ring with unit, such as
    :math:`\mathbb Z` or :math:`\mathbb Z/r Z`. Let
    :math:`\mathrm{N}(\Delta^{n})` be the
    normalised chain complex of the standard simplex of dimension `n`.

    The cyclic group of order `r` acts on the tensor product
    :math:`\mathrm{N}(\Delta_+^{n})^{\otimes r}` by cyclically permuting
    its factors. This tensor product has a subcomplex `I` generated by
    r-tuples of faces that do not have a common vertex and a subcomplex `J`
    generated by r-tuples of faces such that there is a vertex of the standard
    simplex is not contained in any of the faces.

    If `sph_aug=True`, then instances of this class are elements of the
    subcomplex I or its dual. If `sph_aug=False`, then instances of this
    class are elements of the quotient
    :math:`\mathrm{N}(\Delta_+^{n})^{\otimes r} / J`
    or its dual. Finally, if `sph_aug=None`, then they are elements of the
    whole complex
    :math:`\mathrm{N}(\Delta_+^{n})^{\otimes r}`

    The implemented map ``chain_maps.abc`` induces an isomorphism between
    the Milnor
    resolution of size `n` (implemented as MilnorCochain) and the
    subcomplex I. The Alexander (or Poincaré) duality map implemented
    as the method ``alex()`` of this class, induces an isomorphism between the
    subcomplex I and the dual of the quotient
    :math:`\mathrm{N}(\Delta_+^{n})^{\otimes r} / J`.

    .. Warning::
        The faces of the simplex (for example, (1,3) or (0,4)) are
        always assumed to be ordered.

    """

    def __init__(self, data=None, degree=1, order=3, n=1, dual=True, torsion=0,
                 sph_aug=True):

        kwargs = {'degree': degree, 'order': order, 'n': n, 'dual': dual,
                  'torsion': torsion, 'sph_aug': sph_aug}
        super().__init__(data, **kwargs)
        self.degree = degree
        self.order = order
        self.dual = dual
        self.torsion = torsion
        self.n = n
        self.sph_aug = sph_aug

    def action(self, i=1):
        r"""
        The action of a generator of the cyclic group on self.

        Parameters
        ----------
        i: int
            The generator of the cyclic group that is acting

        Returns
        -------
        TensorCochain

        Examples
        --------
        >>> cochain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=-4, order=3, n=2, dual=True, torsion=0)
        >>> cochain.action()
        TensorCochain({((2,), (1,), (0, 2)): 1}, degree=-4, order=3, n=2, dual=True, torsion=0, sph_aug=True)
        >>> chain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=4, order=3, n=2, dual=False, torsion=0)
        >>> chain.action()
        TensorCochain({((1,), (0, 2), (2,)): -1}, degree=4, order=3, n=2, dual=False, torsion=0, sph_aug=True)

        """
        sol = self.create()
        new_i = (-1) ** self.dual * (i % self.order)
        for tensor, a in self.items():
            new_tensor = deque(tensor)
            new_tensor.rotate(new_i)
            new_tensor = tuple(new_tensor)

            par = (sum((len(factor) for factor in tensor[-new_i:])) *
                   sum((len(factor) for factor in tensor[:-new_i])))
            sol += sol.create({new_tensor: a * (-1) ** par})
        return sol

    def partial(self):
        """
        The differential of self.

        Returns
        -------
        TensorCochain

        Examples
        --------
        >>> cochain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=-4, \
        order=3, n=2, dual=True, torsion=0, sph_aug=True)
        >>> cochain.partial()
        TensorCochain({((0, 1, 2), (2,), (1,)): 1, ((0, 2), (2,), (0, 1)): 1, ((0, 2), (0, 2), (1,)): -1, ((0, 2), (1, 2), (1,)): -1}, degree=-5, order=3, n=2, dual=True, torsion=0, sph_aug=True)
        >>> chain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=4, order=3, n=2, dual=False, torsion=0, sph_aug=False)
        >>> chain.partial()
        TensorCochain({((0, 2), (), (1,)): 1, ((0,), (2,), (1,)): -1}, degree=3, order=3, n=2, dual=False, torsion=0, sph_aug=False)

        """
        sol = self.create(degree=self.degree - 1)
        dual_sign = (-1) ** (self.dual * (-self.degree - 1))
        for tensor, a in self.items():
            b = 1
            for j, factor in enumerate(tensor):
                for new_factor, c in Simp(factor, self.n, self.dual,
                                          False).partial():
                    new_tensor = list(tensor)
                    new_tensor[j] = new_factor
                    cond = True
                    if self.sph_aug is True and self.dual:
                        extra = tuple(set(new_factor).difference(factor))
                        cond = any((extra[0] not in _ for _ in new_tensor))
                    elif self.sph_aug is False and not self.dual:
                        extra = tuple(set(factor).difference(new_factor))
                        cond = not all((extra[0] not in _ for _ in new_tensor))
                    if cond:
                        new_tensor = tuple(new_tensor)
                        new_a = a * b * c * dual_sign
                        sol += sol.create({new_tensor: new_a})

                b = b * (-1) ** len(factor)
        return sol

    def alex(self):
        """
        The Alexander (or Poincaré) dual of self.

        Returns
        -------
        TensorCochain

        Examples
        --------
        >>> cochain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=-4, order=3, n=2, dual=True, torsion=0, sph_aug=True)
        >>> cochain.alex()
        TensorCochain({((0, 2), (0, 1), (1,)): -1}, degree=5, order=3, n=2, dual=False, torsion=0, sph_aug=False)
        >>> chain = TensorCochain({((0, 2), (2,), (1,)): 1}, degree=4, order=3, n=2, dual=False, torsion=0, sph_aug=True)
        >>> chain.alex()
        TensorCochain({((0, 2), (0, 1), (1,)): -1}, degree=-5, order=3, n=2, dual=True, torsion=0, sph_aug=False)

        """
        sol = self.create(
            degree=self.degree - (-1) ** self.dual * self.order * (self.n + 1),
            dual=not self.dual)
        if self.sph_aug is not None:
            sol.sph_aug = not self.sph_aug

        for tensor, a in self.items():
            new_tensor = []
            new_sign = 1
            if self.dual:
                par = 0
                for factor in reversed(tensor):
                    new_factor, b = Simp(factor, self.n, self.dual).alex()
                    new_tensor.append(new_factor)
                    new_sign = new_sign * (-1) ** par * b
                    par += (self.n + 1) * len(factor)
                new_tensor = tuple(new_tensor)
            else:
                par = 0
                for factor in tensor:
                    new_factor, b = Simp(factor, self.n, self.dual).alex()
                    new_tensor.append(new_factor)
                    new_sign = new_sign * (-1) ** par * b
                    par += (self.n + 1) * len(factor)
                new_tensor = tuple(reversed(new_tensor))
            sol += sol.create({new_tensor: a * new_sign})
        return sol

    def _valid(self):
        ParametricCounter._valid(self)
        for tensor, a in self.items():
            if len(tensor) != self.order:
                raise ValueError(
                    f'The length of this TensorCochain should be '
                    f'{self.order}: {self}')
            if sum((len(factor) for factor in tensor)) != (
            -1) ** self.dual * self.degree:
                raise ValueError(
                    f'The degree of this TensorCochain should be '
                    f'{self.degree}: {self}')

            # checking that all the factors are faces of the n-simplex
            # checking that there is no common index in all factors (self.dual)
            # or that there is no index missing in all factors (not self.dual)
            sigma = set(range(self.n + 1))
            for factor in tensor:
                if not set(factor).issubset(range(self.n + 1)):
                    raise ValueError(
                        f'The factors of a TensorCochain should be faces of '
                        f'the {self.n}.simplex: {self}')
                if tuple(sorted(factor)) != factor:
                    raise ValueError(f'All faces must be sorted: {self}')
                if self.sph_aug is True:
                    sigma = sigma.intersection(set(factor))
                elif self.sph_aug is False:
                    sigma = sigma.difference(factor)
            if len(sigma) != 0 and self.sph_aug is True:
                raise ValueError(
                    f'There is a common index to all the factors of this '
                    f'TensorCochain: {self}')
            if len(sigma) != 0 and self.sph_aug is False:
                raise ValueError(
                    f'There is a missing index in all the factors of this '
                    f'TensorCochain: {self}')

    @staticmethod
    def _examples(order_tuple=(3, 5, 7)):
        for r in order_tuple:
            for j in range(10):
                for dual in [True, False]:
                    for sph_aug in [True, False, None]:
                        element = []
                        q = 0
                        n = randint(2, 5)
                        sigma = set(range(n + 1))
                        for k in range(r):  # number of joined simplices
                            factor = tuple(
                                sorted(sample(range(n), randint(0, n - 1))))
                            element.append(factor)
                            q += len(factor)
                            sigma = sigma.intersection(set(factor))
                        try:
                            yield TensorCochain({tuple(element): 1},
                                                (-1) ** dual * q, r, n, dual,
                                                sph_aug=sph_aug)
                        except:
                            pass
