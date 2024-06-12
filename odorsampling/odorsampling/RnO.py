# Odor Spaces!
# Mitchell Gronowitz
# Spring 2014

# Edited by Christopher De Jesus
# Summer 2023

"""
QSpace, Odototope, Odorscene, and Receptor objects
Index of document:
1. Global variables
2. Defining all objects
3. Simple functions to create and activate each object
4. Experiments/Simulations utilizing objects
"""


from __future__ import annotations

import math
import copy
import time
import logging
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pylab
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse

from odorsampling import layers, config, utils

# Used for asserts
from numbers import Real

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Optional, Any, Iterable
    from numbers import Number
    from odorsampling import cells

logger = logging.getLogger(__name__)
utils.default_log_setup(logger)


# SD_NUMBER = 1.5
# SD_NUMBER = params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION

# TODO: Replace some of the tuples w/ np arrays (acts like vectors)
#  also sum of squares could be accelerated using (arr**2).sum()

# TODO: Move to params.py
###Global Variables when c != 1 (multiple conn between rec and glom)
# glom_penetrance = .68  # primary glom:rec connection weight if c != 1
s_weights = [0.12, 0.08, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01] # The remaining glom:rec connection weights if c != 1
# numRow = 6 # num of rows of glom
# numCol = 5 # num of cols of glom  (numRow*numCol = total number of Glom)
# constant_attachments = True


class QSpace:
    """Defines the size of the sample space
    ***ex: [(0,5),(0,5)] means qspace extends from 0 to 5 (not including 5) 

    Attributes
    ----------
    _size : list[tuple[float]]
        defines the size of the sample space"""
    
    @property
    def size(self) -> list[tuple[float, float]]:
        #Returns the odors.
        return self._size

    @size.setter
    def size(self, value: list[tuple[float, float]]) -> None:
        """Sets size equal to value.
        Precondition: Value is a list of tuples"""
        # TODO: assert values are floats for consistency
        # assert all(isinstance(val, float) for val in value), "Values must be floats"
        self._size = list(map(tuple, value))
    
    def __init__(self, size: list[tuple[float, float]]):
        self.size = size
    
    def __str__(self):
        return " ".join(map(str, self._size))


class Ligand:
    """
    Represents a simple smell embedded in Q space of dimensionality Q.

    Attributes
    ----------

    _id : int
        Identifies the odor
    _loc : list[float]
        Coordinates in chemical space. Dim=Q
    _dim : int
        Dimension of ligand (Q)
    _conc : float
        Concentration in molar, e.g., 1.5e-5 (NOT 0)
    _aff : float
        Temporary affinity value for a specific receptor
    _eff : float
        Temporary efficacy value for a specific receptor [0..1]
    _occ : float
        Temporary partial occupancy value for a specific receptor [0..1]
    _affs : list[float]
        affs for all recepters
    _effs : list[float]
        effs for all recepters
    _odors2 : list[Ligand] nested for faster calcs

    """
    
    @property
    def id(self) -> int:
        """Returns the id."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets id equal to value.
        Precondition: Value is an int"""
        self._id = int(value)
    
    @property
    def loc(self) -> tuple[float, float]:
        """Returns the loc."""
        return self._loc

    @loc.setter
    def loc(self, value: tuple[float, float]) -> None:
        """Sets loc equal to value.
        Precondition: Value is a List"""
        # TODO: make tuple everywhere
        self._loc = tuple(map(float, value))
    
    @property
    def dim(self) -> int:
        """Returns the dim."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        """Sets dim equal to value.
        Precondition: Value is an int"""
        self._dim = int(value)
    
    @property
    def conc(self) -> float:
        """Returns the conc."""
        return self._conc

    @conc.setter
    def conc(self, value: float) -> None:
        """Sets conc equal to value.
        Precondition: Value is a nonZero number"""
        assert value != 0, "Conc can't be 0!"
        self._conc = float(value)
    
    @property
    def aff(self) -> float:
        """Returns the aff."""
        return self._aff

    @aff.setter
    def aff(self, value: float) -> None:
        """Sets aff equal to value.
        Precondition: Value is a float"""
        self._aff = float(value)
    
    @property
    def eff(self) -> float:
        """Returns the eff."""
        return self._eff

    @eff.setter
    def eff(self, value: float) -> None:
        """Sets eff equal to value.
        Precondition: Value is a float btwn 0..1"""
        self._eff = float(value)
    
    @property
    def occ(self) -> float:
        """Returns the occ."""
        return self._occ

    @occ.setter
    def occ(self, value: float) -> None:
        """Sets occ equal to value.
        Precondition: Value is an float btwn 0..1"""
        assert value >= 0.0 and value <= 1.0, "Occ is not btwn 0 and 1"
        self._occ = float(value)

# Below used in dPsiBarSaturation
    def appendToAffs(self, value: float) -> None:
        """adds aff equal to value.
        Precondition: Value is a float"""
        self._affs.append(float(value))

    def appendToEffs(self, value: float) -> None:
        """adds eff equal to value.
        Precondition: Value is a float"""
        self._effs.append(float(value))

    def appendToOdors2(self, value: Ligand) -> None:
        """adds odor2 equal to value.
        Precondition: odor2 is type of Ligand"""
        self._odors2.append(value)

    # TODO: return copy
    def getOdors2(self) -> list[Ligand]:
        """Returns _odors2."""
        return self._odors2


    #Initializer
    def __init__(self, id_, loc_, conc):
        """
        Initializes ligand
        
        loc is randomly (uniformly) generated point coordinates in QSpace if None.
        """
        self.id = id_
        self.loc = loc_
        self.dim = len(loc_)
        self.conc = conc
        self.aff = 0.0
        self.eff = 0.0
        self.occ = 0.0
        # Used in sumOfSquaresVectorized
        self._affs: list[float] = []
        self._effs: list[float] = []
        self._odors2 = []

    # TODO: Change order to be more consistent with constructor
    @classmethod
    def create(cls, dim, conc, id_=0):
        """
        Returns a ligand with randomly (uniformly) generated ligand point coordinates
        in Q space.
        """
        # TODO: make sure within bounds
        loc = [utils.RNG.uniform(-1000, 1000) for _ in range(dim)]
        return Ligand(id_, loc, conc)
    
    @classmethod
    def load(cls, name: str, helper = False):
        """Returns an ligand from a CSV file with the given name.
        If helper is true, then it's being called from loadOdorscene and we
        don't want to skip the first line.
        Precondition: name exists and it's in CSV format AND the file is <= 2 lines"""
        assert type(name) == str, "name isn't a string"

        if not helper:
            i = 0
            with open(name) as f:
                for l in f.readlines(): #essentially just skip the first line and save the second
                    if i == 1:
                        line = l
                    i += 1
        else:
            line = name
        comma1 = line.find(",")
        comma2 = line.find(",", comma1 + 1)
        comma3 = line.find(",", comma2 + 1)  #Comma before first loc coord
        commas = [line.find(",", comma3 + 1)]
        k = 0
        
        # Find all comma positions in the 'line' string
        while commas[-1] != -1:
            commas.append(line.find(",", commas[k] + 1))
            k+=1

        # Parse 'id_', 'conc', and the first element of 'loc' from 'line'
        id_ = int(line[:comma1])
        conc = float(line[comma2 + 1:comma3])
        index = 0
        loc = [float(line[comma3 + 1:commas[index]])]
        #when commas[index + 1] = -1 it cuts off the last digit
        while commas[index] != -1:
            loc.append(float(line[commas[index] + 1:commas[index + 1]]))
            index += 1
        loc[index] = float(str(loc[index]) + line[-1]) #Accounts for missing digit
        return cls(id_, loc, conc)

    def save(self, name: str):
        """Stores odor as one row in a CSV file with the following columns:
        A = ID# of ligand
        B = text label ('odorant membership')
        C = concentration of ligand in molar
        D...(Q-3) = the Q coordinates of the ligand point
        Precondtion: Name is a str"""
        assert type(name) == str, "name is not a string"
        content = (f"ID, Label, Conc {', coord '.join(map(str, self.dim))}\n {self.id}"
                f",' ',{self.conc}, {','.join(map(str, self.loc))}")
        with open(f"{name}.csv", "w") as f:
            f.write(content)

    def __str__(self):
        """Returns description of Ligand"""
        return f"ID: {self._id} Loc: {self._loc} Conc: {self._conc}"


class Odorscene:
    """Represents a list of ligands embedded in Q-Space. Can resemble an odorant, a complex odorant,
    or a mixture of odorants.
    
    Attributes
    _id : int
        identifies the odor
    _dim : int
        dimension of odorscene (same as dim of attached ligands)
    _odors : list[Ligand]
        List of ligands that define the odorscene
    """

    @property
    def id(self) -> int:
        """Returns the id."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets id equal to value.
        Precondition: Value is an int"""
        self._id = int(value)

    @property
    def dim(self) -> int:
        """Returns the dim."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        """Sets dim equal to value.
        Precondition: Value is an int"""
        self._dim = int(value)

    @property
    def odors(self) -> list[Ligand]:
        """Returns the odors."""
        return self._odors

    @odors.setter
    def odors(self, value: list[Ligand]) -> None:
        """Sets odors equal to value.
        Precondition: Value is a List and dim of elements are equal."""
        # TODO: simplify w/ checks using reduce
        value = list(value)
        self._odors = []
        i = 0
        while i < len(value):
            assert value[0].dim == value[i].dim, "Odors aren't all the same dim."
            self._odors.append(value[i])
            i += 1

    def __init__(self, id_, odors: list[Ligand]):
        """Initializes odorscene"""
        self.id = id_
        self.dim = odors[0].dim
        self.odors = odors
    
    @classmethod
    def create(cls, dim: int, conc: list[float], amt: list[int], qspace: QSpace, odor_id = 0):
        """Returns an odorscene object with a certain amount of randomly generated ligands.
        Conc is a list of concentrations and amt is the amt of each conc (matched by index).
        Ex: conc = [1e-5, 5e-6, 1e-6] and amt = [6, 4, 2] This means:
        six randomly generated ligands at 1x10e-5 molar, four at 5x10e-6 molar,
        and two at 1x10e-6 molar.
        qspace is a qspace object.
        Precondition: Conc and amt are lists of equal length"""
        assert len(conc) == len(amt), "conc and amt are not lists of equal length"
        odors = []
        lig_id = 0
        i = 0
        while i < len(amt):
            ind = 0
            while ind < amt[i]:
                odor = Ligand.create(dim, conc[i], lig_id)
                
                odor = modifyLoc(odor, qspace, dim) 
                odors.append(odor)
                ind += 1
                lig_id += 1
            i += 1
        return cls(odor_id, odors)
    
    def save(self, name: str):
        """
        Stores odor as a CSV file with the following format:
        First and second row: contains ID and dim of odorscene
        Every row after that symbolizes an odor with the following columns:
        A = ID# of ligand
        B = text label ('odorant membership')
        C = concentration of ligand in molar
        D...(Q-3) = the Q coordinates of the ligand point
        Precondition: Name is a str
        """
        assert type(name) == str, "name is not a string"
        st = "\n OdorSceneID, dim \n"
        st = f"{st}{self.id},{len(self.odors)}\n"
        st = st + "ID, Label, Conc "
        
        i = 0

        # Append coordinates to 'st' for each dimension of the first odor
        while i < self.odors[0].dim:
            st = f"{st}, coord {i}"
            i += 1

        # For each odor, append its id and concentration to 'st'
        for odor in self.odors:
            st = f"{st}\n{odor.id},' ',{odor.conc}"
            for _, loc in enumerate(odor.loc):
                st = st + "," + str(loc)

        with open(f"{name}.csv", "a") as f:
            f.write(st)

    @classmethod
    def load(cls, name: str):
        """Returns an odorscene from a CSV file with the given name.
        Precondtion: name existsand it's in CSV format"""
        assert type(name) == str, "name isn't a string"
        i = 0
        odors = []
        with open(name) as f:
            for line in f.readline():
                # If it's the first line, parse the Id
                if i == 1:
                    comma1 = line.find(",")
                    Id = int(line[:comma1])
                # If it's the third line or later, load the odor and append it to the list
                if i > 2:
                    odors.append(Ligand.load(line, True))
                i += 1
        return Odorscene(Id, odors)

    def __str__(self):
        """Returns description of Odor"""
        # Hotfix until Python 3.12
        n_ = '\n'
        return f"ID: {self.id}\nOdors: \n{n_.join(map(str, self._odors))}"


class Receptor:
    """Represents an odor receptor with center (x,y,z...) and radius of 
    sensitivity r.

    Attributes
    ----------
    _id : int
        identifies the receptor
    _mean : list
        list of means for affinity and efficacy Gaussian distributions. Length = Q
    _sdA : list
        List of standard deviations for affinity. Length = Q
    _sdE : list
        List of standard deviations for efficacy. Length = Q
    _covA : list
        sdA ^^ 2
    _covE : list
        sdE ^^ 2
    # TODO: change _scale to _affScale
    _scale : float
        A heuristic scalar value for the "strongest available affinity"
    _effScale : float
        A heuristic scalar value for the "strongest available affinity"
    _activ : float
        Total activation level of receptor
    _occ : float
        Total occupancy of receptor
    _affs : np.ndarray
        aff values of all odors for this receptor
    _effs : np.ndarray
        eff values of all odors for this receptor

    """
    
#Getters and Setters
    @property
    def id(self):
        """Returns id of receptor."""
        return self._id

    @id.setter
    def id(self, value):
        """Sets id to value
        Precondtion: value is an int"""
        self._id = int(value)
    
    @property
    def mean(self):
        """Returns mean of receptor."""
        return self._mean

    @mean.setter
    def mean(self, value: Iterable[float]):
        """Sets id to value
        Precondtion: value is an list"""
        self._mean = tuple(value)
        self._mean_sd_change = True
    
    @property
    def sdA(self) -> tuple:
        """Returns the standard deviations for Affinity."""
        return self._sdA

    @sdA.setter
    def sdA(self, value: tuple):
        """Sets sdA equal to value.
        Precondition: Value is a List with dim Q"""
        self._sdA = tuple(value)
        self._mean_sd_change = True

    @property
    def sdE(self) -> tuple:
        """Returns the standard deviations for Efficacy."""
        return self._sdE

    @sdE.setter
    def sdE(self, value: tuple):
        """Sets sdE equal to value.
        Precondition: Value is a List with dim Q"""
        self._sdE = tuple(value)
        self._mean_sd_change = True
    
    @staticmethod
    def _update_cov_scale(getter):
        # Define a wrapper function that updates covariance and scale if needed
        def wrapper(self: 'Receptor') -> Any:
            if self._mean_sd_change:
                # Update the covariance for A and E
                self._covA = [_sdA**2 for _sdA in self.sdA]
                self._covE = [_sdE**2 for _sdE in self.sdE]
                # FIXME: Cast to float?
                # Update the scale and effective scale using the multivariate normal PDF
                self._scale = mvn.pdf(self.mean, self.mean, self._covA)
                self._effScale = float(mvn.pdf(self.mean, self.mean, self._covE))
                # Reset the change flag
                self._mean_sd_change = False
            return getter(self)
        return wrapper

    @property
    @_update_cov_scale
    def covA(self):
        """Returns the covariance for affinity"""
        return self._covA
    
    @property
    @_update_cov_scale
    def covE(self):
        """Returns the covariance for Efficacy"""
        return self._covE
    
    @property
    @_update_cov_scale
    def scale(self):
        """Returns scale of receptor."""
        return self._scale

    @property
    @_update_cov_scale
    def effScale(self):
        """Returns eff scale of receptor."""
        return self._effScale

    @property
    def activ(self):
        return self._activ

    @activ.setter
    def activ(self, value):
        """Sets activation level for receptor"""
        self._activ = value
    
    def setOcc(self, value):
        """Sets occupancy for receptor"""
        self._occ = value
    
    def setOdoAmt(self,value):
        """sets amount of ligands 'adjacent' (<2SD) to receptor"""
        self._odoAmt = value

    @property
    def affs(self):
        """Returns affs for all ordors"""
        return self._affs

    @affs.setter
    def affs(self, value):
        """Sets affs for all ordors"""
        self._affs = value

    @property
    def effs(self):
        """Returns effs for all ordors"""
        return self._effs

    @effs.setter
    def effs(self, value):
        """Sets effs for all ordors"""
        self._effs = value

#Initializer
    def __init__(self, id_, mean, sda, sde, activ = 0, occ = 0, odoAmt = 0):
        """Initializes a receptor."""
        self.id = id_
        # Note: have to manually set mean or else setter gets called before _covA has value
        # Resolves covA value when sdA is set using setter.
        self.mean = mean
        self.sdA = sda
        self.sdE = sde
        self._covA: list
        self._covE: list
        self._scale: float
        self._effScale: float
        self.activ = activ
        self.setOcc(occ)
        self.setOdoAmt(odoAmt)

    @classmethod
    def create(cls, dim: int, qspace: QSpace, scale: tuple[float, float] = (0.5, 1.5),
               scaleEff: Optional[tuple[float, float]] = None, constMean = False, id_ = 0):
        """Creates a receptor using meanType sdAtype and sdEtype as descriptions
        to how to randomly distribute those values.
        scaleEff is empty unless want to have a diff sd for aff and eff.
        Precondition: qspace must have "dim" dimensions"""
        # print(qspace)
        # print(qspace._size)
        # print(dim)
        assert len(qspace._size) == dim, f"QSpace dims do not match. (QSpace:{qspace._size}, passed:{dim})"
        mean = _distributeMean(dim, qspace, constMean)
        sdA = _distributeSD(dim, scale)
        sdE = _distributeSD(dim, scaleEff) if scaleEff is not None else sdA
        return cls(id_, mean, sdA, sdE)

    # TODO: Use pandas to store this data, then use properties to hide/prevent DataFrame access/object leakage
    def save(self, name: str, helper=False):
        """Stores receptor as one row in a CSV file with the following columns:
        A = ID# of receptor
        B = text label ('receptor membership')
        C...X = list of mean
        X...Y = list of SD for affinity
        y...Z = list of SD for efficacy
        Precondtion: Name is a str"""
        assert type(name) == str, "name is not a string"
        dim = len(self.mean)
        i = 0
        st = ''
        m = ''
        a = ''
        e = ''
        mean = ''
        aff = ''
        eff = ''
        while i < dim:
            m = m + ", Mean " + str(i)
            a = a + ", Aff " + str(i)
            e = e + ", Eff " + str(i)
            mean = mean + "," + str(self.mean[i])
            aff = aff + "," + str(self.sdA[i])
            eff = eff + "," + str(self.sdE[i])
            i += 1
        if helper == False:
            st = st + "ID, Label" + m + a + e + '\n'
        st = st + str(self.id) + ",' '" + mean + aff + eff
        if helper:
            return st
        with open(name + ".csv", "w") as f:
            f.write(st)

    @classmethod
    def load(cls, name: str, helper=False) -> Receptor:
        """Returns a receptor from a CSV file with the given name.
        
        Args:
            name (str): The name of the CSV file.
            helper (bool, optional): Indicates if the method is being called from loadEpithelium. Defaults to False.
        
        Returns:
            Receptor: The receptor object created from the CSV file.
        
        Raises:
            AssertionError: If the name parameter is not a string.
        
        Precondition:
            - The CSV file with the given name exists.
            - The CSV file is in the correct format.
            - The CSV file has a maximum of 2 lines.
        """
        assert type(name) == str, "name isn't a string"
        if helper == False:
            text = open(name)
            i = 0
            for l in text: #essentially just skip the first line and save the second
                if i == 1:
                    line = l
                i += 1
        else:
            line = name
        comma1 = line.find(",")
        comma2 = line.find(",", comma1 + 1) #Comma before first mean coord
        commas = [line.find(",", comma2 + 1)]
        i = 0
        while commas[i] != -1:
            commas.append(line.find(",", commas[i]+1))
            i += 1
        dim = len(commas) // 3
        Id = int(line[:comma1])
        mean = [float(line[comma2 + 1:commas[0]])]
        index = 1
        # Parse 'mean' values from the line based on comma-separated indices
        while index < dim:
            mean.append(float(line[commas[index - 1] + 1:commas[index]]))
            index += 1
        aff = []
        # Parse 'aff' values from the line based on comma-separated indices
        while index < (2 * dim):
            aff.append(float(line[commas[index - 1] + 1:commas[index]]))
            index += 1
        eff = []
        # Parse 'eff' values from the line based on comma-separated indices
        while index < (3 * dim):
            eff.append(float(line[commas[index - 1] + 1:commas[index]])) #Last index of aff loses last digit due to -1
            index += 1
        eff[dim - 1] = float(str(eff[dim - 1]) + line[-1]) #Accounts for missing digit
        if helper == False:
            text.close()
        return Receptor(Id, mean, aff, eff)

    def __str__(self):
        """Returns receptor description"""
        # st = ""
        # for num in self.mean:
        #     st = st + str(num) + ", "
        return f"ID {self.id} Mean: {', '.join(map(str, self.mean))}."
        # return "ID " + str(self._id) + " Mean: " + st[:-2] + "."  #Can add mean if prefer
    
    def __repr__(self) -> str:
        """
        Returns a developer friendly string reperesentation of the Receptor.
        """
        return f"Receptor({self.id}, {self.mean}, {self.sdA}, {self.sdE}, {self.activ}, {self._occ}, {self._odoAmt})"
    

class Epithelium:
    """Represents a list of receptors.
    Instance Attributes:
    _recs : list[Receptor]
    """
    
    @property
    def recs(self) -> list[Receptor]:
        """Returns the receptors."""
        return self._recs

    @recs.setter
    def recs(self, value):
        """Sets receptors equal to value.
        Precondition: Value is a List"""
        # assert isinstance(value, Sequence), f"Value is not a Sequence! {type(value)}"
        self._recs = tuple(value)
    
    def __init__(self, recs):
        """Initializes a epithelium."""
        self.recs = recs

    @classmethod
    def create(cls, n: int, dim: int, qspace: QSpace, scale: tuple[float, float]=(.5, 1.5), scaleEff: Optional[tuple[float, float]] = None, constMean = False):
        """Returns an epithelium with n receptors by calling createReceptor n times.
        SD of each receptor is a uniformly chosen # btwn scale[0] and scale[1]
        Precondition: n is an int"""
        assert type(n) == int, "n is not an integer"
        return Epithelium((Receptor.create(dim, qspace, scale, scaleEff, constMean, i) for i in range(n)))

    def save(self, name : str):
        """Stores each receptor as one row in a CSV file with the following columns:
        A = ID# of receptor
        B = text label ('receptor membership')
        C...X = list of mean
        X...Y = list of SD for affinity
        y...Z = list of SD for efficacy
        Precondtion: Name is a str"""
        assert type(name) == str, "name is not a string"
        m, a , e = "", "", ""
        i = 0
        # Append Mean, Aff, and Eff labels with their respective indices to 'm', 'a', and 'e'
        while i < len(self.recs[0].mean):
            m = f"{m}, Mean {i}"
            a = f"{a}, Aff {i}"
            e = f"{e}, Eff {i}"
            i += 1
        st = "ID, Label" + m + a + e + '\n'
        
        for rec in self.recs:
            st = st + rec.save(name, True) + '\n'
        with open(name + ".csv", "w") as f:
            f.write(st)

    @classmethod
    def load(cls, name: str) -> Epithelium:
        """Returns an epithelium from a CSV file with the given name.
        Precondition: name exists and it's in CSV format"""
        assert type(name) == str, "name isn't a string"
        recs = []
        i = 0
        with open(name) as f:
            for line in f.readlines():
                if i > 0:
                    recs.append(Receptor.load(line, True))
                i += 1
        return cls(recs)

    def __str__(self):
        """Returns epithelium description"""
        n_ = '\n'
        return f"Epithelium contains the following receptors: \n{n_.join(map(str, self.recs))}"

# TODO: delete this class
class Text:
    """Holding experimental text to later store in text file"""
    
    def __init__(self, st, name):
        self._st = st
        self._name = name
        self._st2 = ""

######Functions for objects


    
def modifyLoc(odorant: Ligand, qspace: QSpace, dim: int):
    """Modifies an odorant's location to be within the given qspace of the odorscene
    Precondition: QSpace dimensions are consistent with dim"""
    assert len(qspace.size) == dim, "QSpace dimensions are not consistent with ligand locations"
    i = 0
    loc = list(odorant.loc)
    while i < dim:
        loc[i] = ((loc[i] + (abs(qspace.size[i][0]))) % (abs(qspace.size[i][0]) +
                                abs(qspace.size[i][1]))) + -1 * abs(qspace.size[i][0])
        i += 1
    odorant.loc = loc
    return odorant

def _distributeMean(dim: int, qspace: QSpace, constMean: bool):
    """Returns a list of means randomly distributed within the qspace based on the Type"""
    if constMean:
        mean = [qspace.size[i][1] / 2.0 for i in range(dim)]
    else:
        # TODO: Simplify later
        mean = []
        i = 0
        while i < dim:
            if config.DIST_TYPE_UNIF:
                mean.append(utils.RNG.uniform(qspace.size[i][0], qspace.size[i][1]))
            elif config.DIST_TYPE_GAUSS:
                while True:
                    g = utils.RNG.normal(config.MU, config.SIG)
                    #if ((i==0 and g <= qspace.size[i][1] and g >= 0) or (i==1 and g <= qspace.size[i][1]) and g >= 0):
                    if (g <= qspace.size[i][1] and g >= 0):
                        mean.append(g)
                        break
                #mean.append(utils.RNG.gauss(params.MU, params.SIG))

            i += 1
    return mean

def _distributeSD(dim, scale: tuple[float, float]):
    """Returns a list of standard deviations between scale[0] and scale[1] randomly distributed based on the Type
    Precondition: scale is a 2d list with #'s>=0"""
    assert scale[0] > 0, "scale is not a valid list"
    return [utils.RNG.uniform(*scale) for _ in range(dim)]


######## Activating Receptors/corresponding GL

def activateGL_QSpace(epith: Epithelium, odorscene: Odorscene, gl: layers.GlomLayer, fixed = True, c = 1, sel = "avg"):
    """Given an epithelium, odorscene, and Glomerular Layer, the GL is activated based on
    its corresponding 1:1 receptors. Returns string of data about odorscene and epithelium.
    If c!=1 (convergence ratio of glom:rec, then use updated function to calc non 1:1 glom:rec activation.
    Precondition: Epith and odorscene have the same dimension Q and # of receptors = len(gl)"""    
    assert len(epith.recs[0].mean) == odorscene.odors[0].dim, "Dimensions aren't equal"
    assert len(epith.recs) == len(gl), "Receptors:GL is not 1:1"
    logger.info("Activating GL QSpace.")
    gl.clear_activations()
    
    #Loop through each receptor and eventually calculate activation level
    for rec in epith.recs:
        #Set everything to 0
        activ = 0.0
        odors: list[Ligand] = []
        df = 0
        
        # Calculate the scale for efficacy
        effScale = float(mvn.pdf(rec.mean, rec.mean, rec.covE))
        
        for odor in odorscene.odors:
            #First odorscene
            aff = mvn.pdf(odor.loc, rec.mean, rec.covA)
            aff = aff / rec.scale #Scales it from 0 to 1
            
            #Now convert gaussian aff to kda 
            aff = 10**((aff * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##config.PEAK_AFFINITY etc. are global variables
            #print("aff is " + str(aff))
            odor.aff = float(aff)
            
            # Calculate efficacy
            if fixed:
                odor.eff = 1.0
            else:
                eff = mvn.pdf(odor.loc, rec.mean, rec.covE)
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.eff = eff
                #print("eff is " + str(eff))
            odors.append(odor)
            # TODO: Move all private attribute handling to the class responsible (Ligand) or just create Ligand._aff getter
            df += odor.conc/odor._aff
        
        i = 1

        # Calculate occupancy and activation for each odor
        for odor in odors:
            odor.occ = 1 / (1 + ((odor._aff / odor.conc) * (1 + df - odor.conc / odor._aff))**config.HILL_COEFF) #m=1
            activ += odor.eff * odor.occ
            
            #print("Occ is: " + str(odor._occ))
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            i += 1
        #print("activ level for rec " + str(activ))
        # Set activation level for receptor and corresponding glomerulus
        rec.activ = activ
        gl[rec.id].activ = activ

    # If the convergence ratio is not 1, update the glomerular-receptor connections    
    if c != 1:
        glomRecConnNew(epith.recs, gl, c)

#######Loading and Saving objecsts using CSV files

##### Making a list of sequentially different odorscenes
#createOdorscene(dim, conc, amt, qspace, Id = 0)
def sequentialOdorscenes(n: int, amt: int, dim: int, change: int, qspace: QSpace):
    """Returns a list of n odorscenes, each one differs by change
    Amt = amount of ligand per odorscene
    warning: Doesn't call modify loc so loc could be out of qspace range"""
    odorscenes: list[Odorscene] = []
    ligands: list[Ligand] = []
    
    #make amt ligands starting with [0,0,0...],[.1,.1,.1...]
    i = 0
    while i < amt:
        loc = []
        x = 0
        while x < dim:
            loc.append(i/10.0)
            x += 1
        ligands.append(Ligand(i, loc, 1e-5))
        i += 1

    # Add the first odorscene to the list
    odorscenes.append(Odorscene(0, ligands))
    
    #Creating rest of odorscenes
    i = 1    
    while i < n:
        ligands = []
        numOdors = 0
        odors = odorscenes[i - 1].odors
        while numOdors < amt:
            newOdorLoc = []
            # Create a new odor location by adding 'change' to the previous odor's location
            for num in odors[numOdors].loc:
                newOdorLoc.append(num + change)

            # Add the new ligand to the list    
            ligands.append(Ligand(numOdors, newOdorLoc, 1e-5))
            numOdors += 1
        
        # Add the new odorscene to the list
        odorscenes.append(Odorscene(i, ligands))
        i += 1
    
    return odorscenes


####### Sum of Squares differentiation calculation - original
##Definitions: phi = receptor activation
##             dphi = difference in receptor activation due to two diff odorscenes
##             dpsi = difference in epithelium activation due to two diff odorscenes
## Maximum dpsi value = # of receptors in epithelium (if the first odorscene
## always activates the receptor = 1.0 and the other activates = 0.0)

def sumOfSquares(epithelium: Epithelium, odorscene: Odorscene, dn: list[int], fixed=False, c=1, gl: layers.GlomLayer = None): 
    """Calculates differentiation between epithelium activation of odorscene before
    and after dn using sum of squares. Returns dpsi of the epithelium.
    If fixed=true, then efficacy will be fixed at 1 (only agonists)
    If c!=1, then use function to activate glom with 1:c ratio of Glom:Rec
    Precondtion: dn=list in correct dim"""
    assert odorscene.dim == len(dn), "dimension not consistent with dn"
    logger.debug("Performing sumOfSquares.")
    gl = layers.GlomLayer() if gl is None else gl
    dPsi = 0
    recs2 = copy.deepcopy(epithelium.recs)
    gl.clear_activations() #Sets gl activations and recConn back to 0.0
    
    counter = 0 #for storing info in rec2
    for rec in epithelium.recs:    
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        totOcc = 0.0
        odors: list[Ligand] = []
        odors2: list[Ligand] = []
        rec._activ = 0.0
        rec._occ = 0.0
        rec._odoAmt = 0.0
        df = 0
        df2 = 0
        dphi = 0
        effScale = float(mvn.pdf(rec.mean, rec.mean, rec.covE)) 
        
        for odor in odorscene.odors:
            #First odorscene
            aff = mvn.pdf(odor.loc, rec.mean, rec.covA)
            aff = aff / rec.scale #Scales it from 0 to 1
            #Now convert gaussian aff to kda
            aff = 10**((aff * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##config.PEAK_AFFINITY etc. are global variables
            
            #print("aff is " + str(aff))
            odor.aff = float(aff)
            if fixed:
                odor.eff = 1.0
            else:
                eff = mvn.pdf(odor.loc, rec.mean, rec.covE)
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.eff = eff
            odors.append(odor)
            df += odor.conc / odor._aff
                
            
            #Second Odorscene
            newLoc = []  #Calculating new location
            index = 0
            while index < len(dn):
                newLoc.append(odor.loc[index] + dn[index])
                index += 1
            newOdor = Ligand(odor.id, newLoc, odor.conc)
            
            # Calculate affinity for the second odorscene
            aff2 = mvn.pdf(newLoc, rec.mean, rec.covA)
            aff2 = aff2 / rec.scale #Scales it from 0 to 1
            aff2 = 10**((aff2 * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY)
            #print("aff2 is " + str(aff2))
            newOdor.aff = float(aff2)
            if fixed:
                newOdor.eff = 1.0
            else:
                eff2 = mvn.pdf(newLoc, rec.mean, rec.covE)
                eff2 = float(eff2) / effScale #Scales it from 0 to 1
                newOdor.eff = eff2
            odors2.append(newOdor)
            df2 += newOdor.conc / newOdor._aff
                
        i = 1
        for odor in odors:
            odor.occ = 1 / (1 + ((odor.aff / odor.conc) * (1 + df - odor.conc / odor.aff))**config.HILL_COEFF) #m=1
            activ_1 += odor.eff * odor._occ
            rec._occ += odor.occ #Solely for printing individual receptor activations in experiments
            rec._odoAmt += adjOdors(rec, odor)
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            i += 1
        i = 1
        
        rec.activ = activ_1 #Solely for printing individual receptor activations in experiments
        
        for odor2 in odors2:
            odor2.occ = (1 / (1 + ((odor2.aff / odor2.conc) * (1 + df2 - odor2.conc / odor2.aff))**config.HILL_COEFF)) #m=1
            activ_2 += odor2.eff * odor2.occ
            i += 1
        
        recs2[counter].activ = activ_2
        
        #print("activ_1 is " + str(activ_1))
        #print("activ_2 is " + str(activ_2))
        dPhi = activ_1-activ_2 #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
        
        
        counter += 1
    
    #print("rec dPsi is: " + str(math.sqrt(dPsi))
    
    if c != 1:
        
        gl2 = copy.deepcopy(gl)
        
        conn = glomRecConnNew(epithelium.recs, gl, c, [])
        glomRecConnNew(recs2, gl2, c, conn)

        count = 0
        dPsi = 0
        # Loop through each element in gl
        while count < len(gl):
            dPhi = (gl[count].activ - gl2[count].activ)
            # print("gl 1 activ is " + str(gl[count].activ))
            # print("gl 2 activ is " + str(gl2[count].activ))
            dPsi += dPhi**2
            count += 1
        #print("dPsi is " + str(math.sqrt(dPsi)))
        return math.sqrt(dPsi)
    else:
        return math.sqrt(dPsi)




####### Sum of Squares differentiation calculation - vectorized
##Definitions: phi = receptor activation
##             dphi = difference in receptor activation due to two diff odorscenes
##             dpsi = difference in epithelium activation due to two diff odorscenes
## Maximum dpsi value = # of receptors in epithelium (if the first odorscene
## always activates the receptor = 1.0 and the other activates = 0.0)

def sumOfSquaresVectorized(epithelium: Epithelium, odorscene: Odorscene, dn, repIndex: int, fixed = False, c = 1, gl: layers.GlomLayer = None): 
    
    """Calculates differentiation between epithelium activation of odorscene before
    and after dn using sum of squares. Returns dpsi of the epithelium.
    If fixed=true, then efficacy will be fixed at 1 (only agonists)
    If c!=1, then use function to activate glom with 1:c ratio of Glom:Rec
    Precondtion: dn=list in correct dim"""
    
    logger.debug("Performing sumOfSquaresVecotrized.")

    #assert odorscene.dim== len(dn), "dimension not consistent with dn"
    gl = layers.GlomLayer() if gl is None else gl

    dPsi = 0
    recs2 = copy.deepcopy(epithelium.recs)
    gl.clear_activations() #Sets gl activations and recConn back to 0.0
    
    
    #print("before loop levl 1:" + str(time.time()))
    counter = 0 #for storing info in rec2
    for rec in epithelium.recs:
        
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        odors: list[Ligand] = []
        odors2: list[Ligand] = []
        rec._activ = 0.0
        rec._occ = 0.0
        rec._odoAmt = 0.0
        df = 0
        df2 = 0
        
        '''
        affs = rec.affs
        effs = rec.effs
        '''
        oi = 0
        for odor in odorscene.odors:
            # print(counter)
            # print(len(odor._affs), len(odor._effs))
            odor.aff = odor._affs[counter]
            odor.eff = odor._effs[counter]
            
            newOdors = odor.getOdors2()
            odor2 = newOdors[repIndex]
            
            odor2.aff = odor2._affs[counter]
            odor2.eff = odor2._effs[counter]
                
            #df += prepareOdor(odor, rec, fixed, odors, effScale) 
            odors.append(odor)
            df += odor.conc / odor._aff
            #print("time elapsed prepareOdor:"+ str((time.time() - startTime)))
            
            #Second Odorscene

            odors2.append(odor2)
            df2 += odor2.conc / odor2._aff
            
            #df2 += prepareOdor(newOdor, rec, fixed, odors2, effScale) 
                
            oi += 1
            
        #i = 1
        for odor in odors:
        # for odor in odorscene.odors:    
            #startTime = time.time()
            odor.occ = 1 / (1 + ((odor.aff / odor.conc) * (1 + df - odor.conc / odor.aff))**config.HILL_COEFF) # m=1
            # print("time elapsed odor setOcc:"+ str((time.time() - startTime)))
            activ_1 += odor.eff * odor.occ
            rec._occ += odor.occ #Solely for printing individual receptor activations in experiments
            #startTime = time.time()
            rec._odoAmt += adjOdors(rec, odor)
            #print("time elapsed adjOdors:"+ str((time.time() - startTime)))
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            #i += 1
        #i = 1
        
        rec.activ = activ_1 #Solely for printing individual receptor activations in experiments
        
        for odor2 in odors2:
            odor2.occ = 1 / (1 + ((odor2.aff / odor2.conc) * (1 + df2 - odor2.conc / odor2.aff))**config.HILL_COEFF) # m=1
            activ_2 += odor2.eff * odor2.occ
        
        recs2[counter].activ = activ_2
        
        dPhi = activ_1 - activ_2 #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
        
        counter += 1
    
    if c != 1:
        gl2 = copy.deepcopy(gl)

        conn = glomRecConnNew(epithelium.recs, gl, c, [])
        glomRecConnNew(recs2, gl2, c, conn)

        #count = 0
        dPsi = 0
        for count, glItem in enumerate(gl):
            dPhi = glItem.activ - gl2[count].activ
            dPsi += dPhi**2
    return math.sqrt(dPsi)

# TODO: double check effScale type, and perhaps decouple odor preparation from addition to odors collection
def prepareOdor(odor: Ligand, rec: Receptor, fixed: bool, odors: list[Ligand], effScale: int): 
    
    aff = mvn.pdf(odor.loc, rec.mean, rec.covA)
    #print("time elapsed odor pdf:"+ str((time.time() - startTime)))
    aff = aff/rec.scale #Scales it from 0 to 1
    #Now convert gaussian aff to kda
    #startTime = time.time()
    aff = 10**((aff * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##config.PEAK_AFFINITY etc. are global variables
    #print("time elapsed odor aff:"+ str((time.time() - startTime)))        
    #print("aff is " + str(aff))
    odor.aff = float(aff)
    if fixed:
        odor.eff = 1.0
    else:
        eff = mvn.pdf(odor.loc, rec.mean, rec.covE)
        eff = float(eff) / effScale #Scales it from 0 to 1
        odor.eff = eff
    odors.append(odor)
    return odor.conc / odor.aff

def adjOdors(rec: Receptor, odor: Ligand):
    """Returns 1 if odor is within 2 SD of the rec mean. Otherwise returns 0"""
    #rec._sdA and rec._mean and odor.loc
    #First find avg of sdA
    i = 0
    avg = 0
    dim = odor.dim
    while i < dim:
        avg += rec.sdA[i]
        i += 1
    avg = float(avg)/float(dim)
    #Find Euc distance
    index = 0
    num = 0.0
    while index < dim:
        num += (float(rec._mean[index]) - float(odor.loc[index]))**2
        index += 1
    num = math.sqrt(num)
    if num <= (2.0 * avg):
        return 1
    else:
        return 0


def sumOfSquares2(epithelium: Epithelium, odorscene1: Odorscene, odorscene2: Odorscene, fixed=False):
    """Calculates dPsi for two given odorscenes (instead of one odorscene and dn).
    Precondtion: odorscene1 and odorscene 2 have the same dim and same # of odors"""""
    #assert odorscene1.dim==odorscene2.dim, "ligand dimensions are different"
    #assert len(odorscene1.odors) == len(odorscene2.odors), "two odorscenes don't have the same number of ligands to compare"
    config.PEAK_AFFINITY = -8     # literally 10e-8, not influenced by minimum_affinity value
    minimum_affinity = 2   # asymptotic affinity exponent, negligible
    dPsi = 0
    for rec in epithelium.recs:
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        odors: list[Ligand] = []
        odors2: list[Ligand] = []
        df = 0
        df2 = 0
        dphi = 0
        effScale = float(mvn.pdf(rec.mean, rec.mean, rec.covE)  ) 
        
        i = 0
        while i < len(odorscene1.odors):
            #First odorscene
            odor = odorscene1.odors[i]
            aff = mvn.pdf(odor.loc, rec.mean, rec.covA)
            aff = aff / rec.scale #Scales it from 0 to 1
            #if aff > 1e-128:   #Need so don't divide by 0 when calc df
            #Now convert gaussian aff to kda
            aff = 10**((aff * (config.PEAK_AFFINITY - minimum_affinity)) + minimum_affinity)
            odor.aff = float(aff)
            if fixed:
                odor.eff = 1.0
            else:
                eff = mvn.pdf(odor.loc, rec.mean, rec.covE)
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.eff = eff
            odors.append(odor)
            df += odor.conc/odor.aff
              
            odor2 = odorscene2.odors[i]
            aff2 = mvn.pdf(odor2.loc, rec.mean, rec.covA)
            aff2 = aff2 / rec.scale #Scales it from 0 to 1
            #if aff2 > 1e-128:   #Need so don't divide by 0 when calc df
            #Now convert gaussian aff to kda
            aff2 = 10**((aff2 * (config.PEAK_AFFINITY - minimum_affinity)) + minimum_affinity)
            odor2.aff = float(aff2)
            if fixed:
                odor2.eff = 1.0
            else:
                eff2 = mvn.pdf(odor2.loc, rec.mean, rec.covE)
                eff2 = float(eff2) / effScale #Scales it from 0 to 1
                odor2.eff = eff2
            odors2.append(odor2)
            df2 += odor2.conc / odor2.aff
            i += 1
            
        i = 0
        while i < len(odors):
            odor = odors[i]
            odor.occ = 1/(1+((odor.aff / odor.conc) * (1 + df - odor.conc / odor.aff))**config.HILL_COEFF) # m=1
            activ_1 += odor.eff * odor.occ
                
            odor2 = odors2[i]
            odor2.occ = 1 / (1 + ((odor2.aff / odor2.conc) * (1 + df2 - (odor2.conc / odor2.aff)))**config.HILL_COEFF) # m=1
            activ_2 += odor2.eff * odor2.occ                
            i += 1
                
        dPhi = activ_1 - activ_2 #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
    return math.sqrt(dPsi)


# TODO: Determine type of `r`
def dPsiBarCalcDiag(epithelium: Epithelium, odorscene: Odorscene, r, fixed=False):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but different directions based on
    diagnols.
    fixed is fixing efficacy=1"""
    totalDpsi = 0
    #calc new dn (for now just do 4 diagnals regardless of dim)
    dim = odorscene.dim
    dn = []
    dn2 = []
    dn3 = []
    dn4 = []
    i = 0
    sign = 1
    while i < dim:
        dn.append(r * (1 / math.sqrt(dim)))
        dn2.append(-1 * r * (1 / math.sqrt(dim)))
        dn3.append(sign * r * (1 / math.sqrt(dim)))
        dn4.append(-1 * sign * r * (1 / math.sqrt(dim)))
        sign = -1 * sign
        i += 1
    #calc all dPsi values
    totalDpsi += sumOfSquares(epithelium, odorscene, dn, fixed) + sumOfSquares(epithelium, odorscene, dn2, fixed)
    totalDpsi += sumOfSquares(epithelium, odorscene, dn3, fixed) + sumOfSquares(epithelium, odorscene, dn4, fixed)
    
    return totalDpsi/4.0

def dPsiBarCalcAnglesOrig(epithelium: Epithelium, odorscene: Odorscene, r, fixed = False, text = None, c = 1, gl: layers.GlomLayer = None):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""
    
    rep = 10.0
    amtOfDir = 0
    totalDpsi = 0
    dim = odorscene.dim
    while amtOfDir < rep:
        #Create randomized list of angles
        angles = []
        for i in range(dim - 1):
            if i == dim - 2: #if last angle
                angles.append(utils.RNG.uniform(0,(2 * math.pi)))
            else:
                angles.append(utils.RNG.uniform(0, math.pi))
        #Create dn = amount of change (length of line in each dim given vector r)
        dn = []
        for i in range(dim):
            dn.append(r)
            if i == dim - 1: #if last angle
                for angle in angles:
                    dn[dim - 1] *= math.sin(angle)
            else:
                j = 0
                while j < i:
                    dn[i] *= math.sin(angles[j])
                    j += 1
                dn[i] *= math.cos(angles[i])
        totalDpsi += sumOfSquares(epithelium, odorscene, dn, fixed, c, gl)
        amtOfDir += 1

    if text != None:
        recToText(epithelium, gl, c, text)
    return totalDpsi/rep


# HERE
def dPsiBarCalcAngles(epithelium: Epithelium, odorscene: Odorscene, r, fixed = False, text = None, c = 1, gl: layers.GlomLayer = None):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""
    gl = layers.GlomLayer() if gl is None else gl
    #print("start of dPsiBarCalcAngles")
    gl = layers.GlomLayer() if gl is None else gl
    
    #rep = 10.0
    rep = config.ANGLES_REP
    amtOfDir = 0
    totalDpsi = 0
    
    while amtOfDir < rep: 
        dn = []
        totalDpsi += sumOfSquaresVectorized(epithelium, odorscene, dn, amtOfDir, fixed, c, gl)
        
        amtOfDir += 1
        
    if text != None:
        recToText(epithelium, gl, c, text)
    return totalDpsi / rep

# TODO: Rewrite all of this. (conversions, string handling, etc)
def recToText(epithelium: Epithelium, gl: list[cells.Glom], c: int, text: Text):
    """Stores rec activ and rec occ from epi into a text obj"""
    if c != 1:
        num = convStrToNum(text._st[-4:])
    if text._name == "exp1":
        for rec in epithelium.recs:
            text._st += f",{rec._activ}"
        for rec in epithelium.recs:
            text._st += f",{rec._occ}"
        for rec in epithelium.recs:
            text._st += f",{rec._odoAmt}"
        text._st += '\n'
    elif text._name == "exp2":
        n = 0
        for rec in epithelium.recs:
            text._st += f"Rec{n},{rec._activ},{rec._occ},{rec._odoAmt}\n"
            n += 1
        text._st += '\n' #extra space
    if c!= 1:
        text._st2 += "glom_numOdo=" + str(num)
        for glom in gl:
            text._st2 += "," + str(glom._activation)
        text._st2 += '\n'
        
def convStrToNum(s: str):
    """Given string s with either 3, 2 or 1 num at the end, converts that num to a int"""
    try:
        num = int(s[-3:])
    except:
        try:
            num = int(s[-2:])
        except:
            num = int(s[-1])
    return num


def colorMapSumOfSquares(epithelium: Epithelium, odorscenes: list[Odorscene], r, qspace: QSpace):
    """Creates a colorMap with Q-Space as the x and y axis and dPsi_bar as the
    color for each ligand. dPsi_bar = avg differentiation that occurs in each
    point in Q-Space given many small changes in the odor at that loc.
    Preconditions: Odorscenes is a list of odorscenes containing one ligand. All the ligands
    fill up Qspace. dim = 2d
    WARNING: only works in 2D"""
    assert odorscenes[0].dim == 2, "dimension must be 2D!"
    ##Create graph of all 0's
    graph: list[list[int]] = []
    maxX = qspace.size[0][1]
    maxY = qspace.size[1][1]
    x = 0 
    y = 0
    while y < maxY * config.PIXEL_PER_Q_UNIT:
        row = []
        x = 0
        while x < maxX * config.PIXEL_PER_Q_UNIT:
            row.append(0)
            x += 1
        graph.append(row)
        y += 1
    # print(graph)
    # print("before \n")
    # print(len(odorscenes))
    ##Similar code to above - calc individual dPsi for ligands
    for odorscene in odorscenes:
        dPsiBar = dPsiBarCalcAnglesOrig(epithelium, odorscene, r, True) 
        # print(odorscene.odors[0].loc)

        graph[int(config.PIXEL_PER_Q_UNIT*(odorscene.odors[0].loc[1]))][int(config.PIXEL_PER_Q_UNIT*(odorscene.odors[0].loc[0]))] = dPsiBar
    
    # print("-----------------------------")
    # print(graph)


    #     #TESTING FOR RECEPTOR ELLIPSE ADD-ON
    ells_sda: list[Ellipse] = []
    ells_sde: list[Ellipse] = []
    ii = -1
    for i, rec in enumerate(epithelium.recs): 
            if config.RECEPTOR_INDEX == 'ALL':
                ii = i
            else:
                ii = config.RECEPTOR_INDEX      
            if i == ii:   
                # print("rec" + str(rec.id) + ":")
                # print("Mean:" + str(rec.mean))
                # print("SdA:" + str(rec.sdA))
                # print("eff: ")
                # for e in rec.effs:
                #     print(e)
                # print("aff: " + str(rec.affs))
                print("Qspace size is " + str(qspace.size[1][1]))
        
                qspaceBoundary = qspace.size[1][1]
                # ang = utils.RNG.rand()*360
                ang, ang1, ang2 = utils.RNG.random(3)
                """
                ells.append(Ellipse(xy=rec.mean, width=rec.sdA[0]*standardDeviationNumber, height=rec.sdE[0]*standardDeviationNumber, angle=ang))
                ells.append(Ellipse(xy=rec.mean, width=rec.sdA[1]*standardDeviationNumber, height=rec.sdE[1]*standardDeviationNumber, angle=ang))

                """
                if config.SHOW_SDA_ELLIPSE:
                    # ells_sda.append(Ellipse(xy=rec.mean, width=rec.sdA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.sdA[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))
                    
                    ells_sda.append(Ellipse(xy=config.MOCK_RECEPTOR_MEAN, width=config.MOCK_RECEPTOR_SDA[0]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=config.MOCK_RECEPTOR_SDA[1]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    ells_sda.append(Ellipse(xy=config.MOCK_RECEPTOR_MEAN1, width=config.MOCK_RECEPTOR_SDA1[0]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=config.MOCK_RECEPTOR_SDA1[1]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang1))
                    ells_sda.append(Ellipse(xy=config.MOCK_RECEPTOR_MEAN2, width=config.MOCK_RECEPTOR_SDA2[0]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=config.MOCK_RECEPTOR_SDA2[1]*config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang2))
        

                    # ***************** UNCOMMENT HERE TO PLOT RECEPTORS IN TORUS **************************
                    # newMeanLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1]]
                    # print("newMeanLeft = " + str(newMeanLeft))
                    # print("newMeanLeft x value is " + str(rec.mean[0] - qspaceBoundary) + "!!!!! mean is " + str(rec.mean) + "!!!!! qspace boundary is " + str(qspaceBoundary))
                    # ells_sda.append(Ellipse(xy = newMeanLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanRight = [rec.mean[0] + qspaceBoundary, rec.mean[1]]
                    # ells_sda.append(Ellipse(xy = newMeanRight, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanBottom = [rec.mean[0], rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMeanBottom, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanTop = [rec.mean[0], rec.mean[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMeanTop, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # # bottom left
                    # newMean1bottomLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1bottomLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))            

                    # # bottom right
                    # newMean1bottomRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1bottomRight, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))            
            
                    # # top left
                    # newMean1topLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1topLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))

                    # # top right
                    # newMean1topRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1topRight, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))

                    
                if config.SHOW_SDE_ELLIPSE:
                    # ells_sde.append(Ellipse(xy = rec.mean, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    
                    ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN, width = config.MOCK_RECEPTOR_SDE[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN1, width = config.MOCK_RECEPTOR_SDE1[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE1[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang1))
                    ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN2, width = config.MOCK_RECEPTOR_SDE2[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE2[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang2))

                    # ***************** UNCOMMENT HERE TO PLOT RECEPTORS IN TORUS **************************
                    # newMeanLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1]]
                    # ells_sde.append(Ellipse(xy = newMeanLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanRight = [rec.mean[0] + qspaceBoundary, rec.mean[1]]
                    # ells_sde.append(Ellipse(xy = newMeanRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanBottom = [rec.mean[0], rec.mean[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMeanBottom, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    # newMeanTop = [rec.mean[0], rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMeanTop, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))
                    
                    # # bottom left
                    # newMean1bottomLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1bottomLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))            

                    # # bottom right
                    # newMean1bottomRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1bottomRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))            
            
                    # # top left
                    # newMean1topLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1topLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))

                    # # top right
                    # newMean1topRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1topRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang, zorder = 10))


                #ells.append(Ellipse(xy = rnd.rand(2) * 10, width = rnd.rand(), height = rnd.rand(), angle = rnd.rand() * 360))
    
    fig = plt.figure()

    ax = fig.add_subplot(111, aspect = 'equal')
    #ax2 = fig.add_subplot(111, label = "2", frame_on = False)


    if config.SHOW_SDA_ELLIPSE:
        for e in ells_sda:
            logger.debug("The center is " + str(e.center))
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('red')
            e.set_edgecolor(config.SDA_COLOR)
            e.set_fill(config.SDA_FILL)
            if config.SDA_FILL:
                e.set_facecolor(config.SDA_COLOR)
            e.set_label("SDA")
            e.set_linewidth (config.LINE_WIDTH)
            
    
    if config.SHOW_SDE_ELLIPSE:
        for e in ells_sde:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('blue')
            e.set_edgecolor(config.SDE_COLOR)
            if config.SDE_FILL:
                e.set_facecolor(config.SDE_COLOR)
            e.set_fill(config.SDE_FILL)
            e.set_label("SDE")
            e.set_linewidth (config.LINE_WIDTH)

        logger.debug(qspace.size[0])
        ax.set_xlim(qspace.size[0])
        ax.set_ylim(qspace.size[0])

    im = matplotlib.pylab.imshow(graph, cmap = matplotlib.pylab.cm.YlOrRd, interpolation = "nearest", vmin = 0, vmax = 1, origin = "lower", extent = [0,4,0,4]) #Black = fully active

    plt.title("Differentiation in QSpace")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.colorbar(im)
    pp = PdfPages('ColorMap' + str(qspace.size[0]) + '.pdf')
        
    pp.savefig()
    pp.close()

    #Not closing it will add odor locations to it
    plt.close()

def dPsiBarCalcDns(odorscene: Odorscene, r, rep: int):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""

    #print("start of dPsiBarCalcDns")

    
    amtOfDir = 0
    dim = odorscene.dim
    
    while amtOfDir < rep:
        
        #startTime = time.time()
        #print("start of a rep:" + str(startTime))

        #Create randomized list of angles
        angles = []
        for i in range(dim-1):
            #print("start of first for:" + str(i) + ":" + str(time.time()))

            if i == dim - 2: #if last angle
                angles.append(utils.RNG.uniform(0, (2 * math.pi)))
            else:
                angles.append(utils.RNG.uniform(0, math.pi))
        #Create dn = amount of change (length of line in each dim given vector r)
            #print("end of first for:" + str(i) +":"+ str(time.time()))

        dn = []
        for i in range(dim):
            #print("start of 2nd for:" + str(i) +":"+ str(time.time()))

            dn.append(r)
            if i == dim - 1: #if last angle
                for angle in angles:
                    dn[dim - 1] *= math.sin(angle)
            else:
                j = 0
                while j < i:
                    dn[i] *= math.sin(angles[j])
                    j += 1
                dn[i] *= math.cos(angles[i])
                
            #print("end of 2nd for:" + str(i) + ":" + str(time.time()))
        
       
        amtOfDir += 1

    return dn

@utils.verbose_if_debug
def dPsiBarSaturation(epithelium: Epithelium, r, qspace: QSpace, pdfName: str, labelName: str,
                      excelName: str, fixed = False, c = 1, plotTitle = "", close = False, purp = '', graphIt = True):
    """
    Define x amount of odorscenes with one ligand per odorscene, then with two ligands...
    then calculate dPsibar for each group of odorscene and graph to find saturation at certain
    ligand number.
    if fixed=true than efficacy=1
    if close = True, then graph is closed after this round of data.
    precondition: c = integer, fixed and close = Boolean
    """
    
    startTime = time.time()
    logger.debug("start of dPsiBarSaturation:" + str(startTime))

    size = config.ODOR_REPETITIONS #amount of odorscenes we want to avg out
    #conc = 1e-5
    conc = config.ODOR_CONCENTRATION
    gl = layers.GlomLayer.create(len(epithelium.recs)) #Only if using newly modified gl:rec n:1 ratio
    
    #Holds data for odorscene activations
    st = "Odorscenes,"
    st2 = ""
    st3 = ""
    p = 0
    while p < len(epithelium.recs):
        st += "activ " + str(p) + ","
        st2 += "occ " + str(p) + ","
        st3 += "odoAmt " +str(p) + ","
        p += 1
    text = Text(st + st2 +st3[:-1] + '\n', "exp1")
    
    #If c != 1, also hold info about glom activ
    if c != 1:
        string = "Glom,"
        string2 = ""
        p = 0
        while p < len(gl):
            string2 += "activ " + str(p) + ","
            p += 1
        text._st2 = string + string2 +'\n'
    
    
    
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400] #If change here, change xAxis in expFromRnO
    yaxis = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    # TODO: Make this more clear
    odorscenesArray: list[list[Odorscene]] = [[] * size for x in range(len(xaxis))]
    pdfOdorLocsInput = []
    affs = np.array([])
    effs = np.array([])
    

    pdfOdorLocsInput2 = []
    affs2 = np.array([])
    effs2 = np.array([])

    dns = []
    rep = config.ANGLES_REP

    ligandsArray =[]    
    
    for i in range(size):    
        #k = 0
        #while k < len(xaxis):
            #j = xaxis[k]
        for k, j in enumerate(xaxis):
            n = 0
            while n < j: 
                odor = Ligand(n, createLoc(qspace), conc)
                #ligandsArray.append(Ligand(i, createLoc(qspace), conc))
                ligandsArray.append(odor)
                pdfOdorLocsInput.append(odor.loc)
                n += 1
            #odorscenesArray[k].append(Odorscene(i, ligandsArray[k]))
            odorscene = Odorscene(k, ligandsArray)
            odorscenesArray[k].append(odorscene)
            
            #prepare pdf inputs for ordors2
            dns = dPsiBarCalcDns(odorscene, r, rep)
            
            #Second odors
            for oriOdor in odorscene.odors:
                for i in range(rep):
                    newLoc = []  #Calculating new location
                    for index, dnItem in enumerate(dns):    
                        newLoc.append(oriOdor.loc[index] + dnItem)
                    newOdor = Ligand(oriOdor.id, newLoc, oriOdor.conc)
                    pdfOdorLocsInput2.append(newOdor.loc)
                    oriOdor.appendToOdors2(newOdor)
            
            ligandsArray =[]

        #i += 1



    #draw ellispse for all receptors
    drawEllipseGraph(qspace, epithelium, odorscenesArray, useMockData=False)
    
    for rec in epithelium.recs:
        affs_rec = mvn.pdf(pdfOdorLocsInput, rec.mean, rec.covA)
        
        affs_rec = affs_rec / rec.scale #Scales it from 0 to 1
        #Now convert gaussian aff to kda
        #startTime = time.time()
        affs_rec = 10**((affs_rec * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##config.PEAK_AFFINITY etc. are global variables
        #print("time elapsed odor aff:"+ str((time.time() - startTime)))
        #print("aff is " + str(aff))
        
        rec.affs = affs_rec
        affs = np.append(affs,affs_rec)
        
        
        if not fixed:
            effs_rec = mvn.pdf(pdfOdorLocsInput, rec.mean, rec.covE)
            effs_rec = np.asarray(effs_rec, dtype = np.float64) / rec.effScale #Scales it from 0 to 1


            effs = np.append(effs, effs_rec)
            
        else:
            effs_rec = np.repeat(1.0, affs_rec.size)
            effs = np.repeat(1.0, affs.size)
        rec.effs = effs_rec



        # now do odors2 calc
        affs_rec2: Union[Number, np.ndarray] = mvn.pdf(pdfOdorLocsInput2, rec.mean, rec.covA)
        
        affs_rec2 = affs_rec2 / rec.scale #Scales it from 0 to 1
        #Now convert gaussian aff to kda
        #startTime = time.time()
        affs_rec2 = 10**((affs_rec2 * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##config.PEAK_AFFINITY etc. are global variables
        affs2 = np.append(affs2, affs_rec2)
        
        
        if not fixed:
            effs_rec2 = mvn.pdf(pdfOdorLocsInput2, rec.mean, rec.covE)
            effs_rec2 = np.asarray(effs_rec2, dtype = np.float64) / rec.effScale #Scales it from 0 to 1


            effs2 = np.append(effs2, effs_rec2)
            
        else:
            effs_rec2 = np.repeat(1.0, affs_rec2.size)
            effs2 = np.repeat(1.0, affs2.size)

    locXaxis = []
    locYaxis = []

    vi = 0
    vi2 = 0
    for i in range(size):
        for k, j in enumerate(xaxis):
            #saveOdorscene(odorscenesArray[k][i], "Saved Odors" + str(qspace.size[0]) + purp)

            for odor in odorscenesArray[k][i].odors: #odorscenesArray[k][i].odors
                for li, loc in enumerate(odor.loc):
                    if li == 0:
                        locXaxis.append(loc)
                    if li == 1:    
                        locYaxis.append(loc)
                """
                for odor2 in odor.getOdors2():
    
                    for li, loc in enumerate(odor2.loc):
                        if li == 0:
                            locXaxis.append(loc)
                        if li == 1:    
                            locYaxis.append(loc)
                """

            #n = 0
            #while n < j: 
                #print("vi=" + str(vi))
                #print("affs size," + str(affs[vi]))
                #print("effs size," + str(effs[vi]))
                for rec in epithelium.recs:
                    
                    odor.appendToAffs(float(affs[vi]))
                    odor.appendToEffs(float(effs[vi]))
                    vi += 1
                    
                    #now set resuts to ordor2
                    for odor2 in odor.getOdors2():
                        odor2.appendToAffs(float(affs2[vi2]))
                        odor2.appendToEffs(float(effs2[vi2]))
                        vi2 += 1
                '''
                odor.setAff(float(affs[vi]))
                odor.setEff(float(effs[vi]))
                
                vi += 1
                '''
            text._st += "Odorscene" + str(k + 1)    
            yaxis[k] += dPsiBarCalcAngles(epithelium, odorscenesArray[k][i], r, fixed, text, c, gl)

    count = 0
    while count < len(yaxis):
        yaxis[count] = yaxis[count] / float(size)
        count += 1
    
    #Saving Activated Epithelium data in excel
    # print(text._st)
    # input('Press enter to save data in excel')
    with open(f"{excelName}.csv", "w") as f:
        f.write(text._st)
    
    if c != 1:
        # FIXME: This seems incorrect. (only edit was updating name of qspace.size attr)
        with open("Glom_act with c = " + str(c) + " with " + str(qspace.size[0]) + " qspace.csv", "w") as f:
            f.write(text._st2)
    
    #Saving dPsi data in excel
    st = "Odorscenes, dPsiBar" + '\n'
    i = 0
    while i < len(xaxis):
        st += str(xaxis[i]) + "," + str(yaxis[i]) + '\n'
        i += 1
    n = "dPsi, qspace = (0, " + str(qspace.size[0][1]) + ")" + purp
    with open(n + ".csv", "w") as f:
        f.write(st)

    if graphIt:
        plt.plot(xaxis,yaxis, label = labelName)
        plt.legend()
        plt.title(plotTitle)
        plt.xlabel("Number of Ligands")
        plt.ylabel("dPsiBar")
    
        #Set y_axis limit
        axes = plt.gca()
        axes.set_ylim([0, 0.1]) #*****Change if using >30 recs
    
        #plt.show()
        with PdfPages(pdfName + '.pdf') as f:
            f.savefig()
        if close == True:
            plt.close()

    logger.debug("time elapsed each qspace:" + str(time.time() - startTime))

def createLoc(qspace: QSpace):
    """Given a qspace, return a list of randomized numbers (len = dim) within the qspace"""
    loc = []
    for tup in qspace.size:
        loc.append(utils.RNG.uniform(*tup))
    return loc

#NOT IN USE
def drawOdorLocations(locXaxis,locYaxis, qspace: QSpace, close: bool):
    plt.scatter(locXaxis,locYaxis, s = 1, label = "Odor Location")

    '''
    zAffs = affs_rec
    X,Y = np.meshgrid(affs_rec, affs_rec)
    Z = np.sqrt(X**2 + Y**2)
    
    plt.contourf(X, Y, Z)
    #plt.contourf(X, Y, Z, 24, alpha=0.75, cmap=plt.cm.hot)

    #C=plt.contour(X, Y, Z, 24, colors="black", linewidth=0.5)
    '''

    #plt.legend()
    plt.title("Odor Locations - QSpace "+ str(qspace.size[0]))
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    
    #plt.show()
    pp = PdfPages("Odor Locations " + str(qspace.size[0]) + '.pdf')
    pp.savefig()
    pp.close()
    if close: #No more data to add to the graph
        plt.close()


#NOT IN USE
def drawContourGraph(qspace: QSpace):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X**2 + Y**2)
    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')

    plt.legend()
    plt.title("Odor Locations - QSpace "+ str(qspace.size[0]))
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    #Set y_axis limit
    #axes = plt.gca()
    #axes.set_ylim([0,1.0]) #*****Change if using >30 recs
    
    #plt.show()
    pp = PdfPages("Contour " + str(qspace.size[0]) + '.pdf')
    pp.savefig()
    pp.close()
    #if close == True: #No more data to add to the graph
        #plt.close()
    #plt.close()


def runReceptorOdorGraphToolStandAlone():
    print('need to convert mock arrays data into obj of qspace, epitheliumm, odorscenesArray so we can call the method below')
    drawEllipseGraph('', '', '', True)

def drawEllipseGraph(qspace: QSpace, epithelium: Epithelium, odorscenesArray: list[list[Odorscene]], useMockData = False):
    #print("odorscene 2nd dim len = " + str(len(odorscenesArray[0])))
    assert config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > 0, "Standard Deviation number must be greater than 0"
    if useMockData == False and config.USE_MOCK_ODORS_EVEN_WHEN_RUNNING_CALCS == False:
        assert config.ODORSCENE_INDEX >= 0 and config.ODORSCENE_INDEX < len(odorscenesArray), "Odorscene index must be within the range of xaxis"
        assert config.ODORSCENE_REP_NUMBER >= 0 and config.ODORSCENE_REP_NUMBER < config.ODOR_REPETITIONS, "Odorscene rep number must be within the range 0 - number of repetitions"

    ells_sda: list[Ellipse] = []
    ells_sde: list[Ellipse] = []

    # ang = utils.RNG.random()*360            
    # ang1 = utils.RNG.random()*360
    # ang2 = utils.RNG.random()*360
    ang, ang1, ang2 = utils.RNG.random(3)

    if config.USE_MOCK_RECEPTORS_EVEN_WHEN_RUNNING_CALCS or useMockData:
        if config.SHOW_SDA_ELLIPSE:
            ells_sda.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN, width = config.MOCK_RECEPTOR_SDA[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDA[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
            ells_sda.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN1, width = config.MOCK_RECEPTOR_SDA1[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDA1[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang1))
            ells_sda.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN2, width = config.MOCK_RECEPTOR_SDA2[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDA2[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang2))
        
        #if params.MOCK_RECEPTOR_MEAN[0] + params.MOCK_RECEPTOR_SDA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > params.MOCK_QSPACE_DIMENSION[1]:
            logger.debug("xy = " + str(config.MOCK_RECEPTOR_MEAN[0] + config.MOCK_RECEPTOR_SDA[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION ))
            logger.debug("dim = " + str(config.MOCK_QSPACE_DIMENSION[1]))

            # # left side
            # newMean = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN[0]) * -1,params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sda.append(Ellipse(xy = newMean, width = params.MOCK_RECEPTOR_SDA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang))            
            # # right side
            # newMean = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN[0]), params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sda.append(Ellipse(xy = newMean, width = params.MOCK_RECEPTOR_SDA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang))
            
            # # Mean 1
            # # left side
            # newMean1left = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN1[0]) * -1, params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy = newMean1left, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            # # right side
            # newMean1right = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[0]), params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy = newMean1right, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # bottom side
            # newMean1bottom = [params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy = newMean1bottom, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            # # top side
            # newMean1top = [params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy = newMean1top, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # bottom left
            # newMean1bottomLeft = [params.MOCK_RECEPTOR_MEAN1[0] - params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy = newMean1bottomLeft, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            

            # # bottom right
            # newMean1bottomRight = [params.MOCK_RECEPTOR_MEAN1[0] + params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy = newMean1bottomRight, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            
            # # top left
            # newMean1topLeft = [params.MOCK_RECEPTOR_MEAN1[0] - params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy = newMean1topLeft, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # top right
            # newMean1topRight = [params.MOCK_RECEPTOR_MEAN1[0] + params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy = newMean1topRight, width = params.MOCK_RECEPTOR_SDA1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # Mean 2
            # # left side
            # newMean2left = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN2[0]) * -1, params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy = newMean2left, width = params.MOCK_RECEPTOR_SDA2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))            
            # # right side
            # newMean2right = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN2[0]), params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy = newMean2right, width = params.MOCK_RECEPTOR_SDA2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))

            # # bottom side
            # newMean2bottom = [params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_RECEPTOR_MEAN2[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy = newMean2bottom, width = params.MOCK_RECEPTOR_SDA2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))            
            # # top side
            # newMean2top = [params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy = newMean2top, width = params.MOCK_RECEPTOR_SDA2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDA2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))

        if config.SHOW_SDE_ELLIPSE:
            ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN, width = config.MOCK_RECEPTOR_SDE[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
            ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN1, width = config.MOCK_RECEPTOR_SDE1[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE1[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
            ells_sde.append(Ellipse(xy = config.MOCK_RECEPTOR_MEAN2, width = config.MOCK_RECEPTOR_SDE2[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = config.MOCK_RECEPTOR_SDE2[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))

        #if params.MOCK_RECEPTOR_MEAN[0] + params.MOCK_RECEPTOR_SDE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > params.MOCK_QSPACE_DIMENSION[1]:
            # # left side
            # newMean = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN[0]) * -1, params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sde.append(Ellipse(xy = newMean, width = params.MOCK_RECEPTOR_SDE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang))            
            # # right side
            # newMean = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN[0]), params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sde.append(Ellipse(xy = newMean, width = params.MOCK_RECEPTOR_SDE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang))

            # # left side
            # newMean1left = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN1[0]) * -1, params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy = newMean1left, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            # # right side
            # newMean1right = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[0]), params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy = newMean1right, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # bottom side
            # newMean1bottom = [params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy = newMean1bottom, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            # # top side
            # newMean1top = [params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy = newMean1top, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))
            
            # # bottom left
            # newMean1bottomLeft = [params.MOCK_RECEPTOR_MEAN1[0] - params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy = newMean1bottomLeft, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            

            # # bottom right
            # newMean1bottomRight = [params.MOCK_RECEPTOR_MEAN1[0] + params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy = newMean1bottomRight, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))            
            
            # # top left
            # newMean1topLeft = [params.MOCK_RECEPTOR_MEAN1[0] - params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1topLeft, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # top right
            # newMean1topRight = [params.MOCK_RECEPTOR_MEAN1[0] + params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy = newMean1topRight, width = params.MOCK_RECEPTOR_SDE1[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE1[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang1))

            # # Mean 2
            # # left side
            # newMean2left = [(params.MOCK_QSPACE_DIMENSION[1] - params.MOCK_RECEPTOR_MEAN2[0]) * -1,params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy = newMean2left, width = params.MOCK_RECEPTOR_SDE2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))            
            # # right side
            # newMean2right = [(params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN2[0]), params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy = newMean2right, width = params.MOCK_RECEPTOR_SDE2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))

            # # bottom side
            # newMean2bottom = [params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_RECEPTOR_MEAN2[1] - params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy = newMean2bottom, width = params.MOCK_RECEPTOR_SDE2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))            
            # # top side
            # newMean2top = [params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_QSPACE_DIMENSION[1] + params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy = newMean2top, width = params.MOCK_RECEPTOR_SDE2[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height = params.MOCK_RECEPTOR_SDE2[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle = ang2))

    else:
        #for rec in epithelium.recs:
        ii = -1
        for i, rec in enumerate(epithelium.recs): 
            if config.RECEPTOR_INDEX == 'ALL':
                ii = i
            else:
                ii = config.RECEPTOR_INDEX      
            if i == ii:   
                logger.debug("rec" + str(rec.id) + ":")
                logger.debug("Mean:" + str(rec.mean))
                logger.debug("SdA:" + str(rec.sdA))
                logger.debug("Qspace size is " + str(qspace.size[1][1]))
        
                qspaceBoundary = qspace.size[1][1]
                # ang = rnd.rand() * 360
                ang = utils.RNG.random()
                """
                ells.append(Ellipse(xy = rec.mean, width = rec.sdA[0] * standardDeviationNumber, height = rec.sdE[0] * standardDeviationNumber, angle = ang))
                ells.append(Ellipse(xy = rec.mean, width = rec.sdA[1] * standardDeviationNumber, height = rec.sdE[1] * standardDeviationNumber, angle = ang))

                """
                if config.SHOW_SDA_ELLIPSE:
                    ells_sda.append(Ellipse(xy = rec.mean, width = rec.sdA[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    
                    # newMeanLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1]]
                    # print("newMeanLeft = " + str(newMeanLeft))
                    # print("newMeanLeft x value is " + str(rec.mean[0] - qspaceBoundary) + "!!!!! mean is " + str(rec.mean) + "!!!!! qspace boundary is " + str(qspaceBoundary))
                    # ells_sda.append(Ellipse(xy = newMeanLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle = ang))
                    # newMeanRight = [rec.mean[0] + qspaceBoundary, rec.mean[1]]
                    # ells_sda.append(Ellipse(xy = newMeanRight, width = rec.sdA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    # newMeanBottom = [rec.mean[0], rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMeanBottom, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    # newMeanTop = [rec.mean[0], rec.mean[1] + qspaceBoundary] 
                    # ells_sda.append(Ellipse(xy = newMeanTop, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle = ang))
                    # # bottom left
                    # newMean1bottomLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1bottomLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))            
                    # # bottom right
                    # newMean1bottomRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1bottomRight, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))            
            
                    # # top left
                    # newMean1topLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1topLeft, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))

                    # # top right
                    # newMean1topRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy = newMean1topRight, width = rec.sdA[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdA[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))

                    
                if config.SHOW_SDE_ELLIPSE:
                    ells_sde.append(Ellipse(xy = rec.mean, width = rec.sdE[0] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * config.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    
                    # newMeanLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1]]
                    # ells_sde.append(Ellipse(xy = newMeanLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    # newMeanRight = [rec.mean[0] + qspaceBoundary, rec.mean[1]]
                    # ells_sde.append(Ellipse(xy = newMeanRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    # newMeanBottom = [rec.mean[0], rec.mean[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMeanBottom, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    # newMeanTop = [rec.mean[0], rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMeanTop, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))
                    
                    # # bottom left
                    # newMean1bottomLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] - qspaceBoundary] 
                    # ells_sde.append(Ellipse(xy = newMean1bottomLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))            

                    # # bottom right
                    # newMean1bottomRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1bottomRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))            
            
                    # # top left
                    # newMean1topLeft = [rec.mean[0] - qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1topLeft, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))

                    # # top right
                    # newMean1topRight = [rec.mean[0] + qspaceBoundary, rec.mean[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy = newMean1topRight, width = rec.sdE[0] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, height = rec.sdE[1] * params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION * 2, angle = ang))

 
                #ells.append(Ellipse(xy = rnd.rand(2) * 10, width = rnd.rand(), height = rnd.rand(), angle = rnd.rand() * 360))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 'equal')

    if config.SHOW_SDA_ELLIPSE:
        for e in ells_sda:
            logger.debug("The center is " + str(e.center))
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('red')
            e.set_edgecolor(config.SDA_COLOR)
            e.set_fill(config.SDA_FILL)
            if config.SDA_FILL:
                e.set_facecolor(config.SDA_COLOR)
            e.set_label("SDA")
            e.set_linewidth (config.LINE_WIDTH)
            
    
    if config.SHOW_SDE_ELLIPSE:
        for e in ells_sde:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('blue')
            e.set_edgecolor(config.SDE_COLOR)
            if config.SDE_FILL:
                e.set_facecolor(config.SDE_COLOR)
            e.set_fill(config.SDE_FILL)
            e.set_label("SDE")
            e.set_linewidth (config.LINE_WIDTH)

    if useMockData:
        ax.set_xlim(config.MOCK_QSPACE_DIMENSION)
        ax.set_ylim(config.MOCK_QSPACE_DIMENSION)
        
    else:
        ax.set_xlim(qspace.size[0])
        ax.set_ylim(qspace.size[0])

    """
    sda_patch = patches.Patch(color = 'red', label = 'SDA')
    plt.legend(handles = [sda_patch], loc = 1)

    sde_patch = patches.Patch(color = 'blue', label = 'SDE')
    plt.legend(handles = [sde_patch], loc = 4)
    """

    #draw odor locations for specified odorscean
    locXaxis = []
    locYaxis = []
    locSizes = []

    if config.USE_MOCK_ODORS_EVEN_WHEN_RUNNING_CALCS or useMockData:
        # locSizes.append((math.log10(params.ODOR_CONCENTRATION)+10)*10)
        #locXaxis = params.MOCK_ODORS_X
        #locYaxis = params.MOCK_ODORS_Y
        logger.info('Using mock values')
        for li, loc in enumerate(config.MOCK_ODORS):
                
            locSizes.append((math.log10(config.ODOR_CONCENTRATION) + 10) * 10)
            #print('odor conc = ' + str(odor.conc))
            #print('odor size = ' + str((math.log10(odor.conc)+10)*10))

            locXaxis.append(loc[0])
            locYaxis.append(loc[1])

    else:
        logger.info('NOT using mock values')
        for odor in odorscenesArray[config.ODORSCENE_INDEX][config.ODORSCENE_REP_NUMBER].odors: #odorscenesArray[k][i].odors
            # FIXME: Hotfix. Inspected data suggests they were arrays of the same value. Additionally, the lists were exactly 2*[expected_len].
            #  Moving this into this if case fixes the issue, and provides results, however, I am not sure if the results are correct.
            #  @checkin with Thom
            locSizes.append((math.log10(odor.conc) + 10) * 10)
            for li, loc in enumerate(odor.loc):
                #print('odor conc = ' + str(odor.conc))
                #print('odor size = ' + str((math.log10(odor.conc)+10)*10))

                if li == 0:
                    locXaxis.append(loc)
                if li == 1:    
                    locYaxis.append(loc)
    #plt.scatter(locXaxis,locYaxis, s=100, c='black')
    # print(type(locSizes))
    # print(locSizes)
    # print(len(locXaxis), len(locYaxis))
    # print(len(locSizes))
    plt.scatter(locXaxis,locYaxis, s = locSizes, c = config.ODOR_COLOR)


    #plt.legend()
    #plt.title("Receptors - QSpace "+ str(qspace.size[0])+ "Std Dev "+ str(params.ODORSCENE_REP_NUMBER))
    plt.title(config.GRAPH_TITLE)
    plt.xlabel(config.XLABEL)
    plt.ylabel(config.YLABEL)

    #plt.show()

    if useMockData:
        pp = PdfPages(config.GRAPH_FILE_NAME + str(config.MOCK_QSPACE_DIMENSION) + '.pdf')    
    else:    
        pp = PdfPages(config.GRAPH_FILE_NAME + str(qspace.size[0]) + '.pdf')
        
    pp.savefig()
    pp.close()

    #Not closing it will add odor locations to it
    plt.close()
    
@utils.verbose_if_debug
def graphFromExcel(name: str, xaxis: list[int], numRecs: int, labelName: str, titleName: str, pdfName: str, toggle: str, rep = 10.0, close = False):
    """Given a CSV file from dpsiBarSaturation, create a graph of average receptor
    activation vs num of ligands.
    numRecs=the amount of receptors used in dpsiBarSaturation code(number
    of activation levels etc in excel doc)
    **xaxis=list of numbers that correspond to the odoptes in each odorscene
    Ex: If odorscene with 1 ligand all the way to odorscene with 10 ligands:
        [1,2,3,4,5,6,7,8,9,10]
    name is a string that represent an existing CSV file created from dpsiBarSaturation
    rep = # of repetitions that occur in dpsiBarSaturation
    toggle = "Act" or "Occ"
    close = True when calling graph for last time. Else, False"""
    length = len(xaxis)
    
    text = open(name)
    activ = []
    for x in xaxis:
        activ.append(0)
    
    i = 0
    for line in text:
        if i > 0 and i <= (length * rep): #excludes top line
            #Add all the numbers after comma 1 before comma numRecs+1 and store it
            activ[(i - 1) % length] += _parseAndSum(line, numRecs, toggle)
        i += 1
    
    #Iterate through activ and divide everything by float(numRecs) * rep to get avg
    ind = 0
    while ind < length:
        activ[ind] = activ[ind] / (float(numRecs) * rep)
        ind += 1

    text.close()
    
    # print(activ)
    
    plt.plot(xaxis,activ, label=labelName)
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Number of Ligands")
    
    yAxisName = "Avg Rec "
    if "Glom_act with c = " in name:
        yAxisName = "Avg Glom "
    
    plt.ylabel(yAxisName + toggle + " Lvl")
    #Set y_axis limit
    axes = plt.gca()
    axes.set_ylim([0, 1.0]) #*****Change if using >30 recs
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()

@utils.verbose_if_debug
def dPsiGraphFromExcel(name: str, qspace: QSpace, titleName: str, pdfName: str, close = False):
    """Given an excel doc with dPsiBar data (generated in simulation)
    this function returns a valid graph
    name = name of excel file *Don't forget .csv at end
    titleName = title of graph
    pdfName = name of PDF file
    close = Done adding data to graph"""
    
    text = open(name)
    xaxis = []
    yaxis = []
    label = "(0," + str(qspace.size[0][1]) + ") qspace"
    if titleName[-3:] == "dim":
        label = str(len(qspace.size)) + "D"
    
    i = 0
    for line in text:
        if i > 0:
            comma = line.find(",")
            xaxis.append(int(line[:comma]))
            yaxis.append(float(line[comma + 1:]))
        i+=1

    text.close()
    
    plt.plot(xaxis, yaxis, label = label)
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Number of Ligands")
    plt.ylabel("dPsiBar")
    
    #Set y_axis limit
    axes = plt.gca()
    axes.set_ylim([0,0.12]) #*****Change if using >30 recs
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()

@utils.verbose_if_debug
def dPsiOccActGraphFromExcel(nameDpsi: str, nameAO: str, xaxis: list[int], numRecs: int, labelName: str,
                             titleName: str, pdfName: str, color = "b", rep = 200.0, close = False):
    """Given three excel docs (DpsiBar, Act, Occ) generated from the simulation,
    this function returns a dPsiBar vs Act and Occ graph with a given qspace
    preconditions:
    nameDpsi = valid excel doc name that holds dPsi info with .csv extension
    nameAO = valid excel doc name that holds Act and Occ info with .csv extension
    xaxis = list of ints that resemble num of odorscenes (xaxis in normal dPsi vs Odorscenes graph)"""
    assert nameDpsi[-3:] == "csv", "nameDpsi doesn't have .csv extension"
    dPsi = []
    activ = []
    occ = []
    
    ###extract dPsi info
    with open(nameDpsi) as f:
        dPsi = [float(line[line.find(",") + 1:]) for i, line in enumerate(f.readlines()) if i > 0]
    # print(dPsi)
    ###extract act and occ info
    length = len(xaxis)
    text = open(nameAO)
    for x in xaxis:
        activ.append(0)
        occ.append(0)
    i = 0
    for line in text:
        if i > 0 and i <= (length * rep): #excludes top line
            #Add all the numbers after comma 1 before comma numRecs+1 and store it
            activ[(i - 1) % length] += _parseAndSum(line, numRecs, "Act")
            occ[(i - 1) % length] += _parseAndSum(line, numRecs, "Occ")
        i += 1

    #Iterate through and divide everything by float(numRecs)*rep to get avg
    ind = 0
    while ind < length:
        activ[ind] = activ[ind] / (float(numRecs) * rep)
        occ[ind] = occ[ind] / (float(numRecs) * rep)
        ind += 1
    text.close()
    
    #Graph it
    plt.plot(occ, dPsi, '-.', color = color, label = labelName + ", occ")
    plt.plot(activ, dPsi, color = color, label = labelName + ", act")
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Occ and Act")
    plt.ylabel("dPsiBar")
    
    #Set axes limits
    axes = plt.gca()
    axes.set_ylim([0, 0.1])
    axes.set_xlim([0,1])
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()


def _parseAndSum(line: str, numCommas: int, toggle: str):
    """Given a line from an excel csv file, add all the numbers
    after comma1 before numCommas + 1.
    Example: if numCommas = 30, then summate all numbers between comma 1 and comma 31
    If activ = True, than 
    Return summation"""
    sum_ = 0
    track = 0

    while track < numCommas:
        comma1 = line.find(",")
        comma2 = line.find(",", comma1 + 1)
        sum_ += float(line[comma1 + 1:comma2])
        
        line = line[comma2:]
        track += 1

    if toggle=="Occ":
        sum_ = 0
        track = 0
        while track < numCommas:
            comma1 = line.find(",")
            comma2 = line.find(",", comma1 + 1)
            sum_ += float(line[comma1 + 1:comma2])
        
            line = line[comma2:]
            track += 1
        
    return sum_


def recDensityDpsiGraph(r, qspace: QSpace, odorscene: Odorscene, dim: int, name: str,
                        labelName: str, excelName: str, sd = .5, fixed = False):
    """This function uses a qspace with different # of rec to create different
    receptor densities (receptors are all equally spaced within qspace). Odorscene
    will be adjusted to fit into qspace. Return graph of different dPsi_Bar values
    vs dist between receptors (opp of density).
    r is distance ligands 'move' when calc dPsiBar."""
    recDist: list[float] = []     #x_axis
    dPsiValues = []  #y_axis
    
    #Create a list of receptors
    receptors = []
    # TODO: Remove magic numbers
    receptors.append(35**2)
    receptors.append(30**2)
    i = 20
    while i > 1:
        receptors.append(i**2)
        i -= 1

    # print(receptors)
    text = Text("Receptors, Activ_Lvl, Occ, Num_Odo" + '\n', "exp2")
    
    #Calculate values for graph for each qspace
    for i, n in enumerate(receptors):
        text._st += str(n) + " recs" + '\n'
        
        epi, dist = recInQspace(n, dim, qspace, sd) #creating uniformly spread receptor field (epi) based on qspace
        logger.debug("Odorscene affs effs: %s", list(map(lambda lig: (lig._affs, lig._effs), odorscene.odors)))
        # logger.debug("Generating dPsiBar w/ values epi = %s\nodorscene = %s\nr = %s\nfixed = %s\ntext = %s", epi, odorscene, r, fixed, text)
        dPsibar = dPsiBarCalcAngles(epi, odorscene, r, fixed, text)
        recDist.append(dist)
        dPsiValues.append(dPsibar)
    
    #Store data in csv file
    test = open(excelName + ".csv", "w")
    test.write(text._st)
    test.close
    
    #Plot graph ###Figure out how to label graphs!!
    plt.plot(recDist,dPsiValues, label = labelName)
    plt.legend()
    plt.title("receptor distance vs dPsi: varying ligands2")
    plt.xlabel("Distance between receptors")
    plt.ylabel("dPsiBar")
    #plt.show()
    pp = PdfPages(name + '.pdf')
    pp.savefig()

# TODO: Reimplement this
def recInQspace(n: int, dimen: Real, qspace: QSpace, sd = .5):
    """Given n number of receptors and qspace, returns an epithelium with
    receptors at equally incremented distances from one another and the
    distance between them.
    Precondition: qspace must be a square that has origin at 0"""
    
    dimen = float(dimen)
    n = float(n)
    length = float(qspace.size[0][1])
    chgDim = dimen
    numInRow = math.floor(n**(1 / dimen))
    safe = numInRow**dimen
    resetChgDim = (numInRow**(dimen - 1)) + 1
    dHigh = length / math.floor(n**(1 / dimen))
    dLow = length / math.ceil(n**(1 / dimen))
    switch = safe + 1
    dimen = int(dimen)
    
    coord: list[list[int]] = []
    i = 0
    while i < n+1:
        j = 0
        coord.append([])
        while j < dimen + 1:
            coord[i].append(0)
            j += 1
        i += 1
    
    repeat = True
    while repeat:
        r = 2
        repeat = False
        d = dimen
        while r <= n:
            k = 1
            while k <= dimen:
                coord[r][k] = copy.deepcopy(coord[r - 1][k])
                k += 1
            if (r > switch and d>=chgDim):
                coord[r][d] += dLow
            else:
                coord[r][d] += dHigh
            while coord[r][d] >= length:
                coord[r][d] = 0
                d -= 1
                if d == 0:
                    repeat = True
                    switch -= numInRow
                    resetChgDim -= 1
                    if resetChgDim == 0:
                        chgDim -= 1
                        resetChgDim = (numInRow**(chgDim - 1.0)) + 1
                    break
                if (r > switch and d >= chgDim):
                    coord[r][d] += dLow
                else:
                    coord[r][d] += dHigh
                if (coord[r][d] < length):
                    d = dimen
            if repeat:
                break
            r += 1
    n = int(n)
    numLastRow = 0
    last = coord[n][dimen - 1]
    while last == coord[n - numLastRow][dimen - 1]:
        numLastRow += 1
    newD = length/float(numLastRow)
    count = n - numLastRow + 2
    while count <= n:
        coord[count][dimen] = coord[count - 1][dimen] + newD
        count += 1
    
    avgDist = 0
    i = 2
    while i <= n:
        a = dimen
        while coord[i][a] == 0:
            a -= 1
        avgDist += coord[i][a] - coord[i - 1][a]
        i += 1
    
    avgDist = float(avgDist) / (float(n) - 1)
    
    recs = []
    i = 1
    while i <= n:
        pos = 1
        loc = []
        aff = []
        while pos <= dimen:
            loc.append(coord[i][pos])
            aff.append(sd)
            pos += 1
        recs.append(Receptor(i, loc, aff, aff))
        i += 1
    return Epithelium(recs), avgDist

def recDensityDpsiGraphRandomized(r, qspace: QSpace, odorscene: Odorscene, dim: int, name: str, fixed = False):
    """Returns graph of dPsi vs # of receptors in a given qspace. Values
    are averaged multiple times to get accurate results. qspace is constant
    while number of receptors varies.
    r is distance ligands 'move' when calc dPsiBar.
    This is different from recDensityDpsiGraph since it doesn't equally allign
    receptors in qspace. Just randomly distributes it and calculated multiple times to get an average."""
    #Create xAxis
    receptorNum = []
    i = 2
    while i <10:
        receptorNum.append(i)
        i += 2
    receptorNum.append(i)
    i += 40
    while i < 100:
        receptorNum.append(i)
        i += 50
    while i <= 1000:
        receptorNum.append(i)
        i += 100

    #Create yAxis template
    dPsi = []
    i = 0
    while i < len(receptorNum):
        dPsi.append(0)
        i += 1
    # print(receptorNum)

    repeats = 0
    text = Text("Receptors, Activ_Lvl, Occ, Num_Odo" + '\n', "exp2")
    while repeats < 10:
        num = 0
        while num < len(receptorNum):
            text._st += "Rec # " + str(receptorNum[num]) + "\n"
            epi = Epithelium.create(receptorNum[num], dim, qspace, scale=(.5, 1.5))
            dPsi[num] += dPsiBarCalcAngles(epi, odorscene, r, fixed, text)
            num += 1
        #     print(num)
        # print(repeats)
        text._st += "Repeat again" + "\n"
        repeats += 1
    #Average the dPsi calculations
    i = 0
    while i < len(dPsi):
        dPsi[i] = dPsi[i] / 10.0
        i += 1
    
    #Store data in csv file
    test = open(name + ".csv", "w")
    test.write(text._st)
    test.close
    

    plt.plot(receptorNum,dPsi)
    plt.title("dPsi vs # of Receptors")
    plt.xlabel("Number of Receptors")
    plt.ylabel("dPsiBar")
    plt.show()

def glomRecConnNew(recs: list[Receptor], gl: list[cells.Glom], c = 9, conn: list[list[int]] = []) -> list[list[int]]:
    """New function that deploys gl into olfactory bulb space (config.NUM_ROW X config.NUM_COL)
    and connects to primary and secondary recs with given weights.
    c = num of recs connected to each glom
    conn = Used for random assignment to ensure dPsi is calculated using odors with identical connections
    The following variables are global variables:
    config.GLOM_PENETRANCE = primary weight
    # FIXME: Why is s_weights global?
    s_weights = a list of floats for the remaining weights. If empty, the rest of
    the weights are (1-p_weight)/(c-1)
    Preconditons: # of recs in epi == config.NUM_ROW*config.NUM_COL == len(gl)
                  if constant=True then c must be 9"""
    assert len(gl) == len(recs), "# of recs != # of glom"
    assert len(gl) == config.NUM_ROW * config.NUM_COL, "Glomeruli don't fit in space. Change config.NUM_ROW and config.NUM_COL global variables on top of RnO.py"
    assert len(s_weights) == 0 or c - 1, "weights is an incorrect length"

    #Deploy each glom into space and add primary receptor
    i = 0
    for row in range(config.NUM_ROW):
        for col in range(config.NUM_COL):
            gl[i].loc = (row,col)
            gl[i].rec_conn_map[recs[i]] = config.GLOM_PENETRANCE
            i += 1
    
    if not s_weights:
        w = float(1 - config.GLOM_PENETRANCE) / float(c - 1)
        for i in range(c):
            s_weights.append(w)

    ####Loop through glom and attach remaining recs
    conn = attachSecondaryRecs(gl, recs, c, conn)


    #Activate each glom given rec connections
    i = 0
    for glom in gl:
        activ = 0.0
        for rec in glom._recConn.keys():
            activ += float(rec.activ) * float(glom._recConn[rec]) #rec Activ * weight of conn
        glom.activ = min(activ, 1.0)
        #print("ACTIV IS:" + str(min(activ, 1.0)))
    
    return conn

def attachSecondaryRecs(gl: list[cells.Glom], recs: list[Receptor], c: int, conn: list[list[int]]) -> list[list[int]]:
    """Given gl with primary rec attachments, this function attaches the remaining recs
    if constant=true, then attaches surrounding 8 recs, otherwise it's random assignment.
    If conn = [], randomly assign. If conn isn't empty, than use the given conn info to attach recs
    Returns conn"""

    if config.CONSTANT_ATTACHMENTS: #Loop through glom and find all surrounding gloms to add their recs as secondary
        for glom in gl: 
            locations = getLocations(glom.loc)
            i = 0
            for glom2 in gl:
                if glom2.loc in locations:
                    for rec in glom2._recConn.keys():
                        # FIXME: Shouldn't access private vars outside of class
                        if glom2._recConn[rec] == config.GLOM_PENETRANCE:
                            glom._recConn[rec] = s_weights[i]
                    i += 1

    else: #Randomly assign secondary rec attachments
        ints = range(0,len(recs))
        
        if conn == []: #No prior connection restrictions
            for glom in gl:
                connections = []
                i = 0
                while i < c - 1:
                    index = utils.RNG.choice(ints)
                    rec = recs[index]
                    # FIXME: Shouldn't access private vars outside of class
                    while rec in glom._recConn.keys(): #Choose random rec until find one that isn't connected yet
                        index = utils.RNG.choice(ints)
                        rec = recs[index]
                    glom._recConn[rec] = s_weights[i]
                    connections.append(index)
                    i += 1
                conn.append(connections)

        else: #need glom-rec conn to be identical to past conn
            x = 0
            for glom in gl:
                i = 0
                while i < c - 1:
                    glom._recConn[recs[conn[x][i]]] = s_weights[i]
                    i += 1
                x += 1

    return conn


def getLocations(location: tuple[float, float]):
    """Returns a list of 8 locations (2d list) that surround loc (modular
    config.NUM_ROW and config.NUM_COL to create tourus."""
    x = location[0]
    y = location[1]
    locations = [[x - 1, y - 1], [x, y - 1], [x + 1, y - 1], [x - 1, y], [x + 1, y], [x - 1, y + 1], [x, y + 1], [x + 1, y + 1]]
    for loc in locations:
        loc[0] = loc[0] % config.NUM_ROW
        loc[1] = loc[1] % config.NUM_COL
    return locations
    
    
##############################To make everything more efficient - at the moment sum of squares calculates the same activ_1
#over and over again when called by dPsi angles (it only calculates a new activ_2) - can make a lot more efficient