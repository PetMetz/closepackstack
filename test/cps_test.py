# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:31:27 2019

@author: pce
"""
from closepackstack import Lattice, Site, Structure, build, write_cif, write_str
import numpy as np


def test_lattice_value_propagation(structure):
    print('> Structure.c == Structure.Lattice.c:  ', structure.c == structure.latt.c)
    print('> Structure.c == Site.c:               ', structure.c == structure.sites[0].c)
    print('> Structure.c == Site.Lattice.c:       ', structure.c == structure.sites[0].latt.c)
    return structure.c == structure.sites[0].c == structure.sites[0].latt.c


def test_lattice_instance_propagation(structure):
    print('> Lattice pointer: ', hex(id(structure.latt)))
    return all([structure.latt is site.latt for site in structure.sites])


#%% test lattice manipulation
latt = Lattice(4.93, 2.85, 1.5, 90, 90, 90)

# site instance
site = Site('Mn', 1, 0, 0.1, 1/3, latt)
site2 = site.copy()
site2.specie = 'O'
site2.z = 2/3

# structure instance
structure = Structure(sites=(site, site2), lattice=latt)
test_lattice_value_propagation(structure)

# change lattice
structure.c = 2.5  # ; print (structure)
test_lattice_value_propagation(structure)

#%% test injecting sites into new lattice
# vectors in *fractional units*
f1 = np.array((0, 0, 1))
f2 = np.array((-1/3, 0, 1))
f3 = np.array((-2/3, 0, 1))

# HCP layers
lpb = 2.85
lpc = 1

Olatt= Lattice(np.sqrt(3) * lpb, lpb, lpc, 90, 90, 90)
O1 = Site('O', 1, 0, 0, 0, Olatt)
O2 = Site('O', 1, 1/2, 1/2, 0, Olatt)
OL  = Structure(sites=(O1, O2), lattice=Olatt)

Mnlatt = Olatt.copy()
Mn1 = Site('Mn', 1, 0, 0, 0, Mnlatt)
Mn2 = Site('Mn', 1, 1/2, 1/2, 0, Mnlatt)
MnL = Structure(sites=(Mn1, Mn2), lattice=Olatt)

# 1H test case with space
interlayervectors = [np.array((0.0, 0.0, 5.0))]
sequence = [(OL,  f1), #  * OL.abc),
            (MnL, f2), # * MnL.abc),
            (OL,  f3)  # * OL.abc) 
            ]
blockperiod = int(3)  # block period
Nblocks = int(5)  # number of blocks in periodic stack

supstr= build(sequence=sequence, interlayervectors=interlayervectors,
              blockperiod=blockperiod, Nblocks=Nblocks)

test_lattice_instance_propagation(supstr)


#%% writing results
cifstr = write_cif(supstr, 'test.cif', debug=True)
print(cifstr)
cifstr = write_cif(supstr, 'test.cif', debug=None)


#%% writing results
strstr = write_str(supstr, 'test.str', debug=True)
print(strstr)
strstr = write_str(supstr, 'test.str', debug=None)


#%% test case with laterally shifted origin
interlayervectors = [np.array((0., 0., 5.)),  # Fat!
                     np.array((1/3, 0., 2.5)) # skinny!
                     ]


supstr2 = build(sequence=sequence, interlayervectors=interlayervectors,
                blockperiod=blockperiod, Nblocks=Nblocks)

cifstr = write_cif(supstr2, 'test.cif', debug=True)
print(cifstr)
cifstr = write_cif(supstr2, 'test.cif', debug=None)



#%% test case with density wave
zint = 1 + 7.5 * np.sin(np.linspace(0, np.pi, 20))
x = zint
y = np.zeros_like(zint)
interlayervectors = np.column_stack((x, y, zint))

blockperiod = 3  # non-defective birnessite
Nblocks = int(20) # period of density wave

supstr3 = build(sequence=sequence, interlayervectors=interlayervectors,
                blockperiod=blockperiod, Nblocks=Nblocks)

cifstr = write_cif(supstr3, 'test.cif', debug=True)
print(cifstr)
cifstr = write_cif(supstr3, 'test.cif', debug=None)
















