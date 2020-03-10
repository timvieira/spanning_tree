import numpy as np
import networkx as nx
from arsenal.maths import logsumexp
from itertools import product


class brute_force:
    "Brute-force computation for spanning tree distrbutions."

    def __init__(self, A, r):
        [N,_] = A.shape
        self.N = N; self.A = A; self.r = r

        self.scores = {(root, tree): self.score(root, tree) for root, tree in self.domain()}
        lnz = logsumexp(list(self.scores.values()))

        R = np.zeros(N)       # root marginals
        M = np.zeros((N,N))   # edges marginals
        P = {}
        for (root, tree), s in self.scores.items():
            p = np.exp(s - lnz)
            P[root, tree] = p
            R[root] += p
            for h,m in tree:
                M[h,m] += p

        self.P = P
        self.lnz = lnz
        self.R = R
        self.M = M

    def lprob(self, root, tree):
        "Log-probability of a `tree` with a specific `root`."
        return self.scores[root, tree] - self.lnz

    def score(self, root, tree):
        "Unnormalized log probability of a `tree` with a specific `root`."
        s = self.r[root]
        for h,m in tree:
            assert m != root
            s += self.A[h,m]
        return s

    def domain(self):
        "Enumerate the support of the probability distribution"
        return enumerate_dtrees(self.N)


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

            if is_arborescence(edges):
                accepts += 1
                yield root, frozenset(edges)

    assert proposals == n * (n-1)**(n-1)
    assert accepts == n ** (n-1)


import networkx.algorithms.tree.recognition as R
def is_arborescence(edges):
    "Check if `edges` forms a tree."
    return R.is_arborescence(nx.DiGraph(list(edges)))
