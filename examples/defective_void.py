# -*- coding: utf-8 -*-
"""
About:
    Following the Drits et al [1]_ description of birnessite polytypes, a, b, c, ...
    describe columns of atoms where origins of hexagonal close packed anion and cation
    layers are colocated. Creating an arbitrary structure (e.g. birnessite, delafossite, ...).
    The following code generates the polytypes of 1, 2, and 3 layer structures with defective
    layer types as specified by Drits et al. [1]_ in Tables 2  and 3.
    
    **Table 3.** Symbolic notations of birnessite consisting of hexagonal (vacancy-bearing)
    and orthogonal (vacancy-free) layers and diff ering from each other by stacking modes
    and interlayer structures
    
    =======  ======  ===========  =========================================   =======  =======  ============  ========================================
    \                Hexagonal layer symmetry                                 \                Layers with orthogonal symmetry
    -----------------------------------------------------------------------   ------------------------------------------------------------------------
     Polyt.   Model   XRD (Fig.)   Notation                                    Polyt.   Model    XRD (Fig.)     Notation
    =======  ======  ===========  =========================================   =======  =======  ============  ========================================
      1H        1a    5a            AbCb‘A‘C‘b‘AbC...                           1O      1b                      AbCa’B’B’c’AbC...
      3R1       2a    5b, 8b        AbCb‘A‘B‘a‘CaBa‘C‘A‘c‘BcAc‘B‘C‘b‘AbC...     1M1     2c                      AbCc’CaBb’BcAa’AbC...
      3R1       2b    8a, 10a       AbCb’/a’CaBa’/c’BcAc’/b’AbC...              2d                              AbCa’A’B’b’CaBc’C’A’a’BcAb’B’C’c’AbC...
      3R2       3a    6b, 10b       AbCb‘A‘A‘c‘BcAc‘B‘B‘a‘CaBa‘C‘C‘b‘AbC...     1M2     3d        7b            AbCa’B’C’a’BcAb’C’A’b’CaBc’A’B’c’AbC...
      3R2       3b    5c, 6a        AbCb‘B‘C‘c‘BcAc‘C‘A‘a‘CaBa‘A‘B‘b‘AbC...     1M2     3e        7a            AbCa’BcAb’CaBc’AbC...
      3R2       3c    6c            AbCa‘A‘A‘a‘BcAb‘B‘B‘b‘CaBc‘C‘C‘c‘AbC...                                     
      2H1       4a    5d, 8d, 9b    AbCb‘A‘A‘b‘CbAb‘C‘C‘b‘AbC...                2O1     4c        7c, 9c        AbCa’CbAc’AbC...
      2H1       4b    8c            AbCa’/C’CbAc’/A’AbC...                      2O1     4d        7d            AbCa’B’B’a’CbAc’B’B’c’AbC...
                      5e, 9a                                                    2O1     4e                      AbCc’CbAa’AbC...
      2H2       5a                  AbCb‘A‘B‘c‘AcBc‘A‘C‘b‘AbC...                2O2     5d                      AbCa’B’C’b’AcBa’C’B’c’AbC...
      2H2       5b                  AbCb’AcBc’AbC...                            2O2     5e                      AbCa’A’C’b’AcBa’A’B’c’AbC...
      2H2       5c                  AbCb‘A‘C‘c‘AcBc‘A‘B‘b‘AbC...
      3H1       6a    5f, 10c       AbCb‘A‘B‘c‘AcBc‘A‘B‘c‘AcBc‘A‘C‘b‘AbC...
      3H1       6b    5g, 10d       AbCb’AcBc’AcBc’AbC...
      3H2       7a                  AbCb‘A‘B‘c‘AcBc‘A‘B‘a‘CaBa‘C‘C‘b‘AbC...
    =======  ======  ===========  =========================================   =======  =======  ============  ========================================

    In Drits et al.'s notation, [1]_ - designated an interlayer arrangement forming an octahedral
    set of terminal oxygen (O-type), while = designates a triangular prism (P-type) arrangement.
    primed layers b‘A‘, e.g. refer to Mn^{IL} and O^{IL} sites, respectively.


    Columns A, B, C will be designated as follows, with respect to a monoclinic C-centered
    lattice with :math:`a\ =\ \sqrt(3)\ b,\ b\ =\ a_h,\ and\ c \approx 1 [\mathring{A}]`.
    
    **Table 2.** column positions reference to a C2 monoclinic cell with b = a_h.

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

Created on Tue Apr  2 11:15:32 2019

@author: pce
"""

import numpy as np
from closepackstack import Lattice, Site, Structure, build, write_cif, write_str


#%% configure
# lattice prms
lpb = 2.85
lpa = np.sqrt(3) * lpb
lpc = 1.
d001 = 7.1  # largest basal spacing

# layers
H = Lattice(lpa, lpb, lpc, 90, 90, 90)  # a == sqrt(3) b

HO = Structure(sites=[Site('O', 1, 0, 0, 0, 1, H),
                      Site('O', 1, 1/2, 1/2, 0, 1, H)
                      ],
               lattice=H
               )

HMn = Structure(sites=[Site('Mn', 0.8333, 0, 0, 0, 1, H),
                       Site('Mn', 0.8333, 1/2, 1/2, 0, 1, H)
                       ],
                lattice=H
                )

HOIL = HO.copy()
for site in HOIL.sites:
    site.occ = 0.25

HMnIL = HMn.copy()
for site in HMnIL.sites:
    site.occ = 0.0833

# insert explicit void blocks
void = Structure(sites=None, lattice=H)  
void.c = d001 - 7 * H.c # get approximate value of height assuming a 7.1 A d-spacing and 5 block period

# columns (fractional coordinates)
A = np.array((0, 0, 1), dtype=float)
B = np.array((-1/3, 0, 1), dtype=float)
C = np.array((-2/3, 0, 1), dtype=float)

#%% symtab setup
# parse (e.g. AbCb‘A‘B‘c‘AcBc‘A‘B‘a‘CaBa‘C‘C‘b‘AbC...) using a symtable
# need to substitute prime character for something safe in Python syntax
symtab = {'A': (HO, A),
          'B': (HO, B),
          'C': (HO, C),
          'a': (HMn, A),
          'b': (HMn, B),
          'c': (HMn, C),
          'AIL': (HOIL, A),
          'BIL': (HOIL, B),
          'CIL': (HOIL, C),
          'aIL': (HMnIL, A),
          'bIL': (HMnIL, B),
          'cIL': (HMnIL, C),
          '\/' : (void, A)
          }

# actually, this isn't what we need at the moment.
# =============================================================================
# interpreter = Interpreter(usersyms=syms)
# _eval = interpreter.eval
# 
# =============================================================================
# this is manually spaced. In general we could whip up some regex to automate this
# also, Drit's notation includes the 0th AbC layers at the end of each string (these are trimmed)
#%% containing octahedral ^{VI}TC sites (xIL YIL, x != y)
strings = list((r"A b C bIL  AIL  \/ CIL  bIL ",                      
                r"A b C bIL  AIL  \/ BIL  aIL  C a B aIL  CIL  \/ AIL  cIL  B c A cIL  BIL \/ CIL  bIL",
                r"A b C bIL  AIL  \/ AIL  cIL  B c A cIL  BIL  \/ BIL  aIL  C a B aIL  CIL  \/ CIL  bIL",
                r"A b C bIL  AIL  \/ AIL  bIL  C b A bIL  CIL  \/ CIL  bIL",           
                r"A b C bIL  AIL  \/ BIL  cIL  A c B cIL  AIL  CIL  bIL",           
                r"A b C bIL  AIL  \/ BIL  cIL  A c B cIL  AIL  \/ BIL  cIL  A c B cIL  AIL  \/ CIL  bIL",
                r"A b C bIL  AIL  \/ BIL  cIL  A c B cIL  AIL  \/ BIL  aIL  C a B aIL  CIL  \/ CIL  bIL"
                ))

fnames = list(( '1a_1H', 
                '2a_3R1',
                '3a_3R2',
                '4a_2H1',
                '5a_2H2',
                '6a_3H1',
                '7a_3H2'
                ))

# parse
sequences = []
for string in strings:
    sequences.append([symtab[k] for k in string.split()])

VImap = dict(zip(fnames, sequences))

# build
for fname, seq in VImap.items():
    N = len(seq)  # subperiodic unit == periodic unit
    s = 'Drits_Birnessite_Polytypes\\vacancy\\hexagonal\\{}'.format(fname)
    write_cif(build(seq, None, blockperiod=N, Nblocks=1), fname=s)
    write_str(build(seq, None, blockperiod=N, Nblocks=1), fname=s)


#%% containing tetrahedral ^{IV}TC sites (xIL XIL)

# these aren't correctly constrained to d001 == 7.1 Angstroms. They sheet fragments are correct
# but they are more widely spaced

strings = list((r"A b C bIL - BIL \/ CIL - cIL B c A cIL - CIL \/ AIL - aIL C a B aIL - AIL \/ BIL - bIL", # anti-aligned
                r"A b C aIL - AIL \/ AIL - aIL B c A bIL - BIL \/ BIL - bIL C a B cIL - CIL \/ CIL - cIL", # aligned
                r"A b C bIL AIL \/ CIL - cIL A c B cIL AIL \/ BIL - bIL"  # tetrahedral and octahedral alternating
                ))

fnames = list(('3b_3R2',
               '3c_3R2',
               '5c_2H2'  # mixed, need to distinguish IVMnIL and VIMnIL layer heights
               ))


# apical O is further away in tetrahedral config
void.c = d001 - (1 * HMn.c) - (2 * HO.c) - (2 * HMnIL.c) - (2 * HOIL.c)
IVvoid = void.copy()
IVvoid.c = 1.925 - HMnIL.c
symtab.update({'void': (void, A),
               '-' : (IVvoid, A)
               })
# parse
sequences = []
for string in strings:
    sequences.append([symtab[k] for k in string.split()])

IVmap = dict(zip(fnames, sequences))

# build
for fname, seq in IVmap.items():
    N = len(seq)  # subperiodic unit == periodic unit
    s = 'Drits_Birnessite_Polytypes\\vacancy\\hexagonal\\{}'.format(fname)
    write_cif(build(seq, None, blockperiod=N, Nblocks=1), fname=s)
    write_str(build(seq, None, blockperiod=N, Nblocks=1), fname=s)


#%% containing intercalated species (Zn, e.g.) that aren't in the symtab
# =============================================================================
#                 "A b C b’ \/ a’ C a B a’ \/ c’ B c A c’ \/ b’ A b C",         
#                 "A b C a’ \/ C’ C b A c’ \/ A’ A b C",                 
#                 "A b C b’ \/ A c B c’ \/ A b C",                       
#                 "A b C b’ \/ A c B c’ \/ A c B c’ \/ A b C",
# =============================================================================




