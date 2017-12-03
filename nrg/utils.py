import numpy as np
import scipy.sparse as sps

def is_compact(t):
    if isinstance(t, list):
        return [is_compact(ti) for ti in t]
    return all(np.all(l.bm.sort().compact_form().nblock == l.bm.nblock) for l in t.labels)

def render_string(ops, sites):
    hndim = ops[0].shape[0]
    h = sps.eye(1)
    last_site = -1
    for op, site in zip(ops,sites):
        h = sps.kron(sps.kron(h, sps.eye(hndim**(site-last_site-1))), op)
        last_site = sites
    return h


