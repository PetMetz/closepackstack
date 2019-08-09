# -*- coding: utf-8 -*-
"""

Created on Fri Mar 29 08:09:05 2019

@author: pce
"""

import numpy as np
# from asteval import Interpreter
from collections import OrderedDict, Iterable
from copy import deepcopy

#%% class containers
class Lattice(object):
    """ simple representation of a lattice """
    def __repr__(self):
        return 'Lattice <a={}, b={}, c={}, alpha={}, beta={}, gamma={} >'.format(*self.list)

    def __hash__(self):
        """ overload built in hash """
        return hash((self.a, self.b, self.c, self.al, self.be, self.ga))

    def __eq__(self, other):
        """ overload built-in equal """
        if isinstance(other, Lattice):
            return self.__hash__() == other.__hash__()
        return NotImplemented

    def __ne__(self, other):
        """ overload built-in not equal """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __init__(self, a, b, c, al, be, ga):
        """ intialize unit cell parameters ($\AA$ and $\degree$)"""
        self.a = a
        self.b = b
        self.c = c
        self.al = al
        self.be = be
        self.ga = ga

    def copy(self):
        """ return new instance of self """
        return deepcopy(self)

    # properties
    @property
    def abc(self):
        return np.array((self.a, self.b, self.c), dtype=float)

    @property
    def angles(self):
        return np.array((self.al, self.be, self.ga), dtype=float)

    @property
    def list(self):
        return np.array((self.a, self.b, self.c, self.al, self.be, self.ga), dtype=float)

    @property
    def a(self):
        return float(self._a)

    @property
    def b(self):
        return float(self._b)

    @property
    def c(self):
        return float(self._c)

    @property
    def al(self):
        return float(self._al)

    @property
    def be(self):
        return float(self._be)

    @property
    def ga(self):
        return float(self._ga)

    # set methods
    @abc.setter
    def abc(self, listvalues):
        self.abc = listvalues

    @angles.setter
    def angles(self, listvalues):
        self.angles = listvalues

    @a.setter
    def a(self, value):
        self._a = value

    @b.setter
    def b(self, value):
        self._b = value

    @c.setter
    def c(self, value):
        self._c = value

    @al.setter
    def al(self, value):
        self._al = value

    @be.setter
    def be(self, value):
        self._be = value

    @ga.setter
    def ga(self, value):
        self._ga = value

    # Lattice


class Site(Lattice):
    """ Simple representation of a site, inheriting an associated lattice """
    def __repr__(self):
        return 'Site< {}  occ={}  fx={}, fy={}, fz={} >'.format(*[str(s) for s in (self.name, self.occ, self.fx, self.fy, self.fz)])

    def __init__(self, specie, occupancy, fx, fy, fz, Biso, lattice):
        # Site inherits Lattice attributes
        self.latt = lattice
        super(Site, self).__init__(*self.latt.list)
        # Decorate with atoms
        self.name = specie  # customizable label
        self.specie = specie  # element symbol
        self.occ = occupancy
        self.biso = Biso
        # fractional coords
        self.fx = fx
        self.fy = fy
        self.fz = fz
        # absolute coords
        self.x = fx * self.a
        self.y = fy * self.b
        self.z = fz * self.c
        return

    def copy(self):
        """ return new instance of self """
        return deepcopy(self)


    # properties
    @property
    def x(self):
        return float(self._x)

    @property
    def y(self):
        return float(self._y)

    @property
    def z(self):
        return float(self._z)

    @property
    def xyz(self):
        return np.array((self._x, self._y, self._z))

    @property
    def fx(self):
        return float(self._fx)

    @property
    def fy(self):
        return float(self._fy)

    @property
    def fz(self):
        return float(self._fz)

    @property
    def fxyz(self):
        return  np.array((self._fx, self._fy, self._fz))
    
    @property
    def biso(self):
        return float(self._biso)

    # set methods
    @x.setter
    def x(self, value):
        self._x = value
        self._fx = self._x / self.a

    @y.setter
    def y(self, value):
        self._y = value
        self._fy = self.y / self.b

    @z.setter
    def z(self, value):
        self._z = value
        self._fz = self.z / self.c

    @xyz.setter
    def xyz(self, listvalues):
        self.x, self.y, self.z = listvalues[:]

    @fx.setter
    def fx(self, value):
        self._fx = value
        self._x = self._fx * self.a

    @fy.setter
    def fy(self, value):
        self._fy = value
        self._y = self._fy * self.b

    @fz.setter
    def fz(self, value):
        self._fz = value
        self._z = self._fz * self.c

    @fxyz.setter
    def fxyz(self, listvalues):
        self.fx, self.fy, self.fz = listvalues[:]
    
    @biso.setter
    def biso(self, biso):
        self._biso = biso

    # superceed lattice setters with update methods
    def setlatt(self, listvalues):
        """ listvalues (len==6) """
        self.a, self.b, self.c, self.al, self.be, self.ga = listvalues[:]

    @property
    def latt(self):
        return self._latt

    @latt.setter
    def latt(self, lattice):
        if not hasattr(self, '_latt'): # instantiate
            self._latt = lattice
        else:  # if self._latt != lattice:  # override
            # print('setting Site.latt' )
            self._latt = lattice
            self.setlatt(self._latt.list)

    @Lattice.a.setter
    def a(self, value):
        self._a = value           # set
        self.latt.a = value
        if hasattr(self, '_x'):    # override
            self.fx = self.x / value

    @Lattice.b.setter
    def b(self, value):
        self._b = value          # set
        self.latt.b = value
        if hasattr(self, '_y'):   # override
            self.fy = self.y / value

    @Lattice.c.setter
    def c(self, value):
        self._c = value          # set
        self.latt.c = value
        # print( value )
        if hasattr(self, '_z'):   # override
            self.fz = self.z / value
            # print ('setting new Site.latt.c')
            # print (self.fz)

    # Site


class Structure(Lattice, Iterable):
    """ general structure container without symmetry operations """
    def __repr__(self):
        [site.__repr__() for site in self.sites]
        return'Structure<\n   {latt}\n   {sites}\n   >'.format(
                latt=self.latt.__repr__(),
                sites='\n   '.join([site.__repr__() for site in self.sites])
                )

    def __iter__(self):
        return self

    def __reset__(self):
        self.current = 0

    def __next__(self):
        if self.current > self.high:
            self.__reset__()
            raise StopIteration
        else:
            self.current += 1
            return self.sites[self.current - 1]

    # Py2
    next = __next__

    def __init__(self, sites=None, lattice=None):
        """
        A general structure container without enabled symmetry operations. The Lattice is a 
        shared attribute of both the Structure and Site instances, so that changes to the 
        Lattice influence the positions, etc. of the Sites through a common data structure.
        
        Structure, Lattice, and Site objects have copy methods which return a deepcopy of 
        the object. Lattice objects are hashed based on their scalar parameterization for
        comparison.
        
        Changes to the Lattice instance at the Structure level are propagated to the subordinate
        objects.
        
        Args:
            - sites (closepackedstack.Site instance(s))
            - lattice (closepackedstack.Lattice instance)

        """
        # sites
        if sites is None:
            sites = []
        self.sites = sites
        # lattice
        self.latt = lattice
        super(Structure, self).__init__(*self.latt.list)
        # iter support
        self.current = 0
        self.lo = 0
        self.high = len(self.sites) - 1
        return

    def copy(self):
        """ return new instance of self """
        return deepcopy(self)

    # some class specific overrides here too (there's probably a cleaner way to mix these in)
    @property
    def latt(self):
        return self._latt

    @latt.setter
    def latt(self, lattice):
        self._latt = lattice
        for site in self.sites:  # flatten lattice
            # if site.latt != self._latt:
            # print('setting Structure.latt')
            site.latt = self._latt
                
    @Lattice.a.setter
    def a(self, value):
        self._a = value           # set
        self.latt.a = value       # why aren't these at the same pointer??
        self.latt = self.latt     # push 
        
    @Lattice.b.setter
    def b(self, value):
        self._b = value          # set
        self.latt.b = value
        self.latt = self.latt
        
    @Lattice.c.setter
    def c(self, value):
        self._c = value          # set
        self.latt.c = value
        self.latt = self.latt

    # Layer


class PeriodicCycle(Iterable):
    """
    Resets index of iterable to 0 after max index exceeded

    Reference:
        SO # 19151 - build-a-basic-python-iterator
    """
    def __init__(self, iterable):
        """ cycles over iterable infinitely from idx = 0 to idx = len(iterable) - 1 """
        self.iterable = iterable
        self.current = 0
        self.low = 0
        self.high = len(iterable) - 1

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        """ if self.current > self.high, self.current = self.low """
        if self.current > self.high:
            self.current = self.low
        self.current += 1
        return self.iterable[self.current - 1]
    
    next = __next__ # Python 2

    # PeriodicCycle


#%% build functionality
def build(sequence, interlayervectors, blockperiod, Nblocks, *args, **kwargs):
    """
    build a collection of sites and a corresponding lattice based on a sequence mapping
    layers (Structure objects) and position vectors.
    
    dimension of structure will be **Nblocks x blockperiod x vector components x layer.c dimensions **
    
    Note:
        - current implimentation deals with scalar lattice, so gamma != 90 will produce some problems
    
    Args:
        - sequence (iterable)  [(Structure, FRACTIONAL_VECTOR), .... ]
        - interlayervectors (list(1x3)) cycle of interlayer vectors to insert every blockperiod
              in ABSOLUTE_UNITS
        - blockperiod (int) how many single atom layers comprise a block
        - Nblocks (int) howmany block cycles to inject into a supercell
    Defaults:
        - origin (0, 0, 0) in ABSOLUTE UNITS
        
    
    """
    # initialize ---------------------
    if interlayervectors is None:
        interlayervectors = [np.array((0,0,0))]
    ss = PeriodicCycle(sequence)  # cyclable mapping of (layer, vector)
    iv = PeriodicCycle(interlayervectors)  # cyclable set of interlayer vectors to inject (shape (N, (1,3)))
    origin = np.array((0, 0, 0), dtype=float)  # initialize origin |  reference for column position |  modified by non-zero interlayer vectors
    sites = []

    # build sites --------------------
    for idx in range(Nblocks * blockperiod):
        # get layer and position
        layer, vector = next(ss)
        vx, vy, vz = vector.T
        
        print(idx, origin)
# =============================================================================
#         # force first layer to origin, no matter what! No choices for the user! None!
#         if idx == 0:
#             shift = (origin - vector) * layer.abc
#             origin += shift
# =============================================================================

        # add sites to site list
        for site in layer:
            rv = site.copy()
            # rv.xyz += (vector * layer.abc + origin)  # add in layer position + interlayer operations
            rv.x += vx * layer.a
            rv.y += vy * layer.b
            rv.z += origin[-1]
            sites.append(rv)

        #  + every blockperiod insert interlayer adjustments
        if (idx != 0) and ((idx + 1) % blockperiod == 0):  # look ahead
            origin += next(iv)   # with memory for subsequent layers

        # increase origin by layer height & do next
        origin[-1] += (vector[-1] * layer.c)
        
# =============================================================================
#     # give back the adjustment of the 0th layer
#     origin[-1] -= shift[-1]
# =============================================================================
    
    # make superlattice ----------------
    # lattice
    scalar_lattice = layer.latt.list  # unpack lattice from last layer
    scalar_lattice[2] = origin[-1]   # c = sum of all appended layers + vector operations
    slatt = Lattice(*scalar_lattice)

    # apply periodic constraint
    for site in sites:
        for i, coord in enumerate(site.fxyz[:2]):
            if abs(coord) > 1.:
                site.fxyz[i] = coord % 1

    # pop sites into new lattice
    supstr = Structure(sites=sites, lattice=slatt)

    # if nothing's gone wrong, we're done
    return supstr


def unique_labels(Structure):
    """
    TOPAS requires unique labels for sites.
    """
    for specie in set([site.specie for site in Structure.sites]):
        l = [x for x in Structure.sites if x.specie is specie]
        n = iter(range(1, len(l) + 1))
        for site in [x for x in Structure.sites if x.specie is specie]:
            site.name = specie + str(next(n))
    return
    

def write_cif(Structure, fname, debug=None):
    """
    write closepackstack.Structure to .cif file in P1 symmetry
    
    Args:
        - Structure (closepackstack.Structure)
        - fname (string)
    
    Optional:
        - debug == True will return formatted string instead of writing file
        
    Note:
        - current Structure object has no thermal displacement attribute -> default to 1 (Biso)
    """
    # NAME, METHOD, lpa, lpb, lpc, lpal, lpbe, lpga, SG, SITE_BLOCK
    from peter.stringtemplates.templates import template_cif
    from os.path import abspath
    
    # get unique labels
    unique_labels(Structure)
    
    sites = Structure.sites
    SITE_BLOCK = '\n'.join([' '.join([str(getattr(sites[idx], attr)) for attr in 
                                          ('name', 'specie', 'occ', 'fx', 'fy', 'fz', 'biso')])  #  + '  1'
                                for idx in range(len(sites))])
    
    inp = dict(NAME=fname, METHOD='clospackstack.py', SG='P1', SITE_BLOCK=SITE_BLOCK) 
    inp.update(zip(('lpa', 'lpb', 'lpc', 'lpal', 'lpbe', 'lpga'), Structure.latt.list))
    rv = template_cif.format(**inp)
    
    if debug is not None:  # optionally return str to console for debugging
        return rv
    
    with open(abspath(fname) + '.cif', 'w+') as f:
        f.write(rv)
    

def write_str(Structure, fname, debug=None):
    """
    write closepackstack.Structure to .str file in P1 symmetry (TOPAS)
    
    Args:
        - Structure (closepackstack.Structure)
        - fname (string)
    
    Optional:
        - debug == True will return formatted string instead of writing file
        
    Note:
        - current Structure object has no thermal displacement attribute -> default to 1 (Biso)   
    """
    from templates import template_str
    from os.path import abspath
    
    # get unique labels
    unique_labels(Structure)
    
    # METHOD NAME SG lpa lpb lpc lpal lpbe lpga SITE_BLOCK
    sites = Structure.sites
    # load site x y z occ biso {}
    SITE_BLOCK = '\n        '.join([' '.join([str(getattr(sites[idx], attr)) for attr in 
                                          ('name', 'fx', 'fy', 'fz', 'specie', 'occ', 'biso')])   # + '  1'
                                for idx in range(len(sites))])
    
    inp = dict(NAME=fname, METHOD='clospackstack.py', SG='P1', SITE_BLOCK=SITE_BLOCK) 
    inp.update(zip(('lpa', 'lpb', 'lpc', 'lpal', 'lpbe', 'lpga'), Structure.latt.list))
    rv = template_str.format(**inp)
    
    if debug is not None:  # optionally return str to console for debugging
        return rv
    
    with open(abspath(fname) + '.str', 'w+') as f:
        f.write(rv)
        
    
    
    
    