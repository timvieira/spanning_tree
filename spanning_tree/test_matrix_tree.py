"""
Test suite for matrix-tree theorem.
"""
import numpy as np
from arsenal.maths import quick_fdcheck

from matrix_tree import matrix_tree_theorem as mt
from brute_force import brute_force as bf, enumerate_dtrees


def test_enumerate_dtrees(N):
    print('[test enumerate dtrees]')
    S = set()
    for t in enumerate_dtrees(N):
        assert t not in S, [t, S]
        S.add(t)
    print(f'number of trees over N={N} nodes -> {len(S)}')
    # Cayley's formula for the number of complete graphs on N vertices
    assert len(S) == N ** (N-1)


def test_mt_bf(A, r):
    """
    Compare the computation of log Z and edge marginals via the matrix-tree
    theorem and brute-force search.  Warning this method is exponential in
    the number of nodes so it should only be used on small graphs.
    """

    [bf_lnz, bf_dr, bf_dA] = bf(A, r)
    [mt_lnz, mt_dr, mt_dA] = mt(A, r)

    print()
    print(f'bf logZ= {bf_lnz:g}')
    print(f'mt logZ= {mt_lnz:g}')

    np.testing.assert_allclose(bf_lnz, mt_lnz)

    print('p(i = root(t))')
    print('\n'.join(f'  {x}' for x in str(mt_dr).split('\n')))

    print('p(<h,m> in t)')
    print('\n'.join(f'  {x}' for x in str(mt_dA).split('\n')))

    np.testing.assert_allclose(bf_dr, mt_dr)
    np.testing.assert_allclose(bf_dA, mt_dA)


def test_mt_self_test(A, r):
    """
    Run self-consistency tests.
    """
    [_, dr, dA] = mt(A, r)

    # Finite-difference gradient test.
    assert quick_fdcheck(lambda: mt(A, r)[0], r, dr, verbose=False).max_rel_err <= 0.0001
    assert quick_fdcheck(lambda: mt(A, r)[0], A, dA, verbose=False).max_rel_err <= 0.0001

    # An additional self-consistency self is the the marginals sum to one as
    # follows, sum_h p(<h,m> \in T) = 1.
    d = dA.sum(axis=0) + dr
    assert np.allclose(d, 1)


def test():

    for N in [2,3,4,5]:
        test_enumerate_dtrees(N)

    for N in [2,3,4,5]:
        A = np.random.normal(size=(N,N))
        r = np.random.normal(size=N)
        test_mt_bf(A, r)

    for N in [10,20,50]:
        A = np.random.normal(size=(N,N))
        r = np.random.normal(size=N)
        test_mt_self_test(A, r)


if __name__ == '__main__':
    test()
