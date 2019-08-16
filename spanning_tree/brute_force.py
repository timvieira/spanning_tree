import numpy as np
import networkx as nx
from arsenal.maths import logsumexp
from itertools import product


def brute_force(A, r):
    "Brute-force computation of the log partition function and edge marginals."
    [N,_] = A.shape

    def score(t):
        [root, tree] = t
        s = r[root]
        for h,m in tree:
            assert m != root
            s += A[h,m]
        return s

    scores = {t: score(t) for t in enumerate_dtrees(N)}

    lnz = logsumexp(list(scores.values()))

    R = np.zeros(N)
    M = np.zeros((N,N))
    for (root, tree), s in scores.items():
        p = np.exp(s - lnz)
        R[root] += p
        for h,m in tree:
            M[h,m] += p

    return lnz, R, M


def enumerate_dtrees(n):
    "Enumerate all spanning trees of a complete graph over n nodes."

    # Implementation: We use a rejection-based strategy where the (two) outer
    # loops "propose" singly connected rooted graphs and the body of loops
    # checks whether the singly connected graph is also acyclic (i.e., a tree).
    proposals = 0; accepts = 0
    for root in range(n):

        # Define for each node a set of possible `parents`
        #
        # Outer loop: Each node picks a parent (that's not itself).  Since a
        # `root` has been chosen, we ensure here that no graphs will have edges
        # pointing from it. (In other words, the outer loop is over singly
        # connected graphs.)
        parents = []
        for j in range(n):
            if j == root:
                d = [None]  # `None` is a sentinel value; the important thing is
                            # that the set has size one.
            else:
                d = set(range(n)) - {j}   # minor improvement: drop `j` from the
                                          # set: a tree can't have a self loop.
            parents.append(d)

        # Iterate over the Cartesian product of possible parents to get singly
        # connected graphs (technically, with no self cycles)
        for A in product(*parents):
            proposals += 1

            edges = [(h,m) for m, h in enumerate(A) if m != root and h != None]

            if is_tree(edges):
                accepts += 1
                yield root, frozenset(edges)

    assert proposals == n * (n-1)**(n-1)
    assert accepts == n ** (n-1)


def is_tree(edges):
    "Check if `edges` forms a tree."
    try:
        # Use networkx for cycle detection
        list(nx.topological_sort(nx.DiGraph(list(edges))))
    except nx.exception.NetworkXUnfeasible:
        return False
    return True
