from itertools import combinations


def partition_size(N):
    if N < 2:
        yield [1]*N
        return
    for s in range(1, N+1):
        yield from ([s] + p for p in partition_size(N-s))


def fill_combination(A,parts,R=None):
    if R is None: R = [tuple()]*len(parts)
    size = max(parts)  # fill largest partitions first
    if size < 2:       # when only single element partitions are left
        iA = iter(A)   # fill them with the remaining indexes
        yield [r if p!=1 else (next(iA),) for r,p in zip(R,parts)]
        return
    i,parts[i]= parts.index(size),0    # index of largest partition
    for combo in combinations(A,size): # all combinations of that size
        R[i] = combo                   # fill part and recurse
        yield from fill_combination(A.difference(combo),[*parts],[*R]) 


def partCombo(A):
    for parts in partition_size(len(A)):
        for iParts in fill_combination({*range(len(A))},parts):   # combine indexes
            yield [[A[i] for i in t] for t in iParts] # get actual values