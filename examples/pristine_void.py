# -*- coding: utf-8 -*-
"""
About:
    Following the Drits et al [1]_ description of birnessite polytypes, a, b, c, ...
    describe columns of atoms where origins of hexagonal close packed anion and cation
    layers are colocated. Creating an arbitrary structure (e.g. birnessite, delafossite, ...).
    The following code generates the polytypes of 1, 2, and 3 layer structures with pristine
    (defect free) layers, as described in Drits et al. [1]_ Table 1. In Drits' notation, these 
    polytypes are written concisely as NS_m, where N = number of layers in the cell, S designates
    the layer symmetry (H = hexagonal, O = distorted hexagonal = orthorhombic), and m is the
    index of the unique polytype.

    **Table 1.** List of possible periodic layer stacking modes in birnessite
    consisting of hexagonal (vacancy-bearing) or orthogonal (vacancy-free) layers
    Layer stacking Hexagonal layers Orthogonal layers

    ===========================   ====================   =======================
        Layer stacking              Hexagonal layers        Orthorhombic layers
    ===========================   ====================   =======================
     AbC – AbC ...                       1H                       1O
     AbC = CbA = AbC ...                 2H_1                     2O_1
     AbC – AcB – AbC ...                 2H_2                     2O_2
     AbC = CaB = BcA = AbC ...           3R_1                     1M_1
     AbC – BcA – CaB – AbC ...           3R_2                     1M_2
     AbC – AcB – AcB – AbC ...           3H_1                     3O_1
     AbC – AcB – CaB – AbC ...           3H_2                     3O_2
    ===========================   ====================   =======================


    In Drits et al.'s notation, [1]_ - designated an interlayer arrangement forming an octahedral
    set of terminal oxygen (O-type), while = designates a triangular prism (P-type) arrangement.

    Columns A, B, C will be designated as follows, with respect to a monoclinic C-centered
    lattice with :math:`a\ =\ \sqrt(3)\ b,\ b\ =\ a_h,\ and\ c \approx 1 [\mathring{A}]`.
    
    **Table 2.** column positions reference to a C2 monoclinic cell with b = :math:`a_h`.

    ============   ===================
       symbol        vector [frac.]
    ============   ===================
         A             (0, 0, 0)
         B             (-1/3, 0, 0)
         C             (-2/3, 0, 0)
    ============   ===================

**References:**

..    [1]  V.A. Drits, B. Lanson, A.C. Gaillot (2007) \"Birnessite polytype systematics  
              and identification by powder diffraction.\" Am. Miner. 92(5-6) pp. 771-788.   
              DOI: 10.2138/am.2007.2207  


Created on Tue Apr  2 11:15:02 2019

@author: pce
"""

import numpy as np
from closepackstack import Lattice, Site, Structure, build, write_cif, write_str

#%% configure
# lattice prms
lpb = 2.85
lpa = np.sqrt(3) * lpb
lpc = 1.

# layers
H = Lattice(lpa, lpb, lpc, 90, 90, 90)  # a == sqrt(3) b
O = Lattice(lpa * 1.05, lpb, lpc, 90, 90, 90)  # a >  sqrt(3) b

HO = Structure(sites=[Site('O', 1, 0, 0, 0, 1, H),
                      Site('O', 1, 1/2, 1/2, 0, 1, H)
                      ],
               lattice=H
               )

HMn = Structure(sites=[Site('Mn', 1, 0, 0, 0, 1, H),
                       Site('Mn', 1, 1/2, 1/2, 0, 1, H)
                       ],
                lattice=H
                )

OO, OMn = HO.copy(), HMn.copy()
OO.latt = O
OMn.latt = O

blockperiod = 3  # number of HCP layers in a block

# columns (fractional coordinates)
A = np.array((0, 0, 1), dtype=float)
B = np.array((-1/3, 0, 1), dtype=float)
C = np.array((-2/3, 0, 1), dtype=float)

# cyclic permutations of ABC  ( 3! = 6)
Habc = np.array([(HO, A), (HMn, B), (HO, C)])
Hbca = np.array([(HO, B), (HMn, C), (HO, A)])
Hcab = np.array([(HO, C), (HMn, A), (HO, B)])
Hcba = np.array([(HO, C), (HMn, B), (HO, A)])
Hbac = np.array([(HO, B), (HMn, A), (HO, C)])
Hacb = np.array([(HO, A), (HMn, C), (HO, B)])

Oabc = np.array([(OO, A), (OMn, B), (OO, C)])
Obca = np.array([(OO, B), (OMn, C), (OO, A)])
Ocab = np.array([(OO, C), (OMn, A), (OO, B)])
Ocba = np.array([(OO, C), (OMn, B), (OO, A)])
Obac = np.array([(OO, B), (OMn, A), (OO, C)])
Oacb = np.array([(OO, A), (OMn, C), (OO, B)])

# interlayer vectors (absolute coordinates)
d7 = np.array([(0, 0, 7.1 - 3 * lpc)])  # 7.1 angstrom d-spacing



# ==================================  Hexagonal ========================================== #
#%%     AbC – AbC ...                       1H                       1O
#------------------------------------------------------------------------------
N = 1
sequence = Habc
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\1H.cif')


#%%     AbC - CbA - AbC ...                 2H_1                     2O_1
#------------------------------------------------------------------------------
N = 2
sequence = np.row_stack((Habc, Hcba))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\2H1.cif')


#%%     AbC – AcB – AbC ...                 2H_2                     2O_2
#------------------------------------------------------------------------------
N = 2
sequence = np.row_stack((Habc, Hacb))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\2H2.cif')


#%%     AbC - CaB - BcA - AbC ...           3R_1                     1M_1
#------------------------------------------------------------------------------
N = 3
sequence = np.row_stack((Habc, Hcab, Hbca))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
s = r'Drits_Birnessite_Polytypes\pristine\hexagonal\3R1'
write_cif(rv, s)
write_str(rv, s)

#%%     AbC – BcA – CaB – AbC ...           3R_2                     1M_2
#------------------------------------------------------------------------------
N = 3
sequence = np.row_stack((Habc, Hbca, Hcab))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\3R2.cif')


#%%     AbC – AcB – AcB – AbC ...           3H_1                     3O_1
#------------------------------------------------------------------------------
N = 3
sequence = np.row_stack((Habc, Hacb, Hacb))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\3H1.cif')


#%%     AbC – AcB – CaB – AbC ...           3H_2                     3O_2
#------------------------------------------------------------------------------
N = 3
sequence = np.row_stack((Habc, Hacb, Hcab))
rv = build(sequence, interlayervectors=d7, blockperiod=blockperiod, Nblocks=N)
write_cif(rv, r'Drits_Birnessite_Polytypes\pristine\hexagonal\3H2.cif')






































