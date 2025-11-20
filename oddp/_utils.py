from itertools import combinations
from bisect import bisect_left

def binom(q):
    """returns the mod 2 value of the binomial coefficient q choose 2"""
    return (q // 2) % 2

def sign_perm(sequence):
    """The sign of the permutation that orders the tuple sequence."""
    ordered_set = sorted(set(sequence))
    counts = {v: 0 for v in ordered_set}
    transpositions = 0
    for w in sequence:
        transpositions += sum([counts[v] for v in ordered_set if v > w])
        counts[w] = counts[w] + 1
    return (-1) ** (transpositions % 2)

def sign_reorder(first, second):
    """Returns the sign of splitting a face as the join of a subface and its
    complement. If last = False, then the sign of the splitting of a face as
    the join of its complement and the subface.
    """
    par = 0
    for v in second:
        par += len([w for w in first if w > v]) % 2
    return (-1) ** par


def sign_complement(face, subface, last=True):
    """Returns the sign of splitting a face as the join of a subface and its
    complement. If last = False, then the sign of the splitting of a face as
    the join of its complement and the subface.
    """
    par = 0
    complement = tuple(sorted(set(face).difference(set(subface))))
    if last:
        return sign_reorder(complement, subface)
    else:
        return sign_reorder(subface, complement)

def partitions(face):
    """Yields all possible ways of decomposing a non-empty tuple face as a
    multiple join of non-empty smaller tuples, together with the sign of their
    reorderings.
    """
    if len(face) == 1:
        return ({(face,): 1},)
    else:
        # The partial list of partitions whose last element has length >1
        partitions_1 = [{(face,): 1}]
        # The partial list of partitions whose last element has length =1
        partitions_0 = [{}]
        for k in range(1, len(face)):  # cardinal of partition
            partitions_1.append({})
            partitions_0.append({})
            for partition, a in partitions_1[k - 1].items():
                initial, end = partition[:-1], partition[-1]
                # end is split into new_end_1 and new_end_2,
                # with new_end_1 of len j
                for j in range(1, len(end)):
                    for new_end_1 in combinations(end, j):
                        new_end_2 = tuple(set(end).difference(new_end_1))
                        new_partition = initial + (new_end_1, new_end_2)
                        new_a = a * sign_reorder(new_end_1, new_end_2)
                        if j != len(end) - 1:
                            partitions_1[k].update({new_partition: new_a})
                        else:
                            partitions_0[k].update({new_partition: new_a})
        for k in range(len(partitions_0)):
            partitions_1[k].update(partitions_0[k])
        return tuple(partitions_1)


def admissible_iterator(basic: tuple, rule: dict, p: int):
    M = [iter(basic)]
    first = [None]
    while len(first) > 0:
        first.pop(-1)
        while len(first) < p:
            try:
                first.append(next(M[-1]))
                if len(first) < p:
                    M.append(iter(rule[first[-1]]))
            except StopIteration:
                M.pop(-1)
                try:
                    first.pop(-1)
                except IndexError:
                    break
        else:
            yield first


def admissible_iterator_set(basic: tuple, rule: dict, p: int):
    M = [iter(basic)]
    first = [None]
    first_set = [None]
    # a = [None]
    while len(first) > 0:
        first.pop(-1)
        first_set.pop(-1)
        # a.pop(-1)
        while len(first) < p:
            try:
                new = next(M[-1])
                first.append(new)
                try:
                    first_set.append(first_set[-1].union(new[0]))
                    # a.append(a[-1] * new[1])
                except IndexError:
                    first_set.append(set(new[0]))
                    # a.append(new[1])
                if len(first) < p:
                    M.append(iter(rule[first[-1][0]]))
            except StopIteration:
                M.pop(-1)
                try:
                    first.pop(-1)
                    first_set.pop(-1)
                    # a.pop(-1)
                except IndexError:
                    break
        else:
            yield first, first_set[-1] #, a[-1]


def find(tup, i):
    j = bisect_left(tup, i)
    if j != len(tup) and tup[j] == i:
        return True
    else:
        return False



def pprint(dictionary):
    for key, value in dictionary.items():
        print(key, value)
    print('----------------------')


