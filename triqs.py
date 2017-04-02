import numpy

from pytriqs.operators.util import *
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import *

from Uijkl import generate_U_tensor_SK2

def unique_U_dict4(U_dict4):
    r = {}
    for key, value in U_dict4.iteritems():
        key_tmp = list(key)
        value_tmp = value

        if key_tmp[0] < key_tmp[1]:
            key_tmp[0], key_tmp[1] = key_tmp[1], key_tmp[0]
            value_tmp *= -1

        if key_tmp[2] < key_tmp[3]:
            key_tmp[2], key_tmp[3] = key_tmp[3], key_tmp[2]
            value_tmp *= -1

        key_tmp = tuple(key_tmp)

        if key_tmp in r:
            r[key_tmp] += value_tmp
        else:
            r[key_tmp] = value_tmp
    return r

def unique_U_dict2(U_dict2):
    r = {}
    for key, value in U_dict2.iteritems():
        key_tmp = list(key)

        if key_tmp[0] < key_tmp[1]:
            key_tmp[0], key_tmp[1] = key_tmp[1], key_tmp[0]

        key_tmp = tuple(key_tmp)

        if key_tmp in r:
            r[key_tmp] += value
        else:
            r[key_tmp] = value
    return r


def symmetrize(G_loc, gen, dim_gen):
    G_loc_symm = G_loc.copy()
    U = numpy.array(gen)
    for bname, gf in G_loc:
        print bname, G_loc[bname].data[0,:,:]
    for i in range(dim_gen-1):
        print U
        for bname, gf in G_loc:
            #print bname, numpy.einsum('il,mlj->mij', U, G_loc[bname].data)[0,:,:]
            tmp = numpy.einsum('il,mlj->mij', U.conjugate().transpose(), G_loc[bname].data)
            tmp = numpy.einsum('mil,lj->mij', tmp, U)
            G_loc_symm[bname].data[:,:] += tmp
        U = numpy.dot(gen, U)
    assert (numpy.sum(numpy.abs(U-numpy.identity(U.shape[0])))<1e-8)

    for bname, gf in G_loc:
        G_loc_symm[bname].data[:,:,:] = G_loc_symm[bname].data[:,:,:]/float(dim_gen)

    #PM
    G_loc_symm['up_0'].data[:,:,:] = G_loc_symm['down_0'].data[:,:,:] = 0.5*(G_loc_symm['up_0'].data + G_loc_symm['down_0'].data)

    return G_loc_symm

#Order of operators: (1/2) \sum_{ijkl} U_{ijkl} c^\dagger_i c^\dagger_j c_l c_k
def generate_U_tensor_SK(n_orb, U, Up, JH):
    U_tensor = numpy.zeros((n_orb,n_orb,n_orb,n_orb),dtype=float)

    for iorb1 in xrange(n_orb):
        for iorb2 in xrange(n_orb):
            for iorb3 in xrange(n_orb):
                for iorb4 in xrange(n_orb):
                    coeff = 0.0
                    if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
                        coeff = U
                    elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
                        coeff = Up
                    elif iorb1==iorb3 and iorb2==iorb4 and iorb1!=iorb2:
                        coeff = JH
                    elif iorb1==iorb2 and iorb3==iorb4 and iorb1!=iorb3:
                        coeff = JH

                    U_tensor[iorb1, iorb2, iorb4, iorb3] += coeff

    return U_tensor
