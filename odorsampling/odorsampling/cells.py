# Glomeruli and Mitral Cell Objects
# Mitchell Gronowitz
# Spring 2015

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

from collections import Counter
import logging
import builtins
from abc import ABC
# Used for asserts
if builtins.__debug__:
    from numbers import Rational, Integral

from odorsampling import config, utils

# Type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import MutableMapping, Optional, Mapping, Hashable
    from RnO import Receptor
    

logger = logging.getLogger(__name__)
utils.default_log_setup(logger)

cell_counter = Counter()
"""
Maps cell type to the number of cells of that type that have been created.
"""
cell_counter_rw_lock = utils.ReaderWriterSuite()
"""
Reader and writer locks for the `cell_counter`. 
"""

# FIXME: Fix `dec`.
def add_count(cell_type: Hashable, dec: int = True) -> int:
    """
    Increments the number of cells of the given type that have been created.
    
    Parameters
    ----------
    cell_type
        The Cell type who's ID to increment.
    dec
        WIP, low-priority; An optional boolean argument whether to start IDs from 0 or 1.
        True by default. Currently, if set to False, may break the program.
        Also supports decrementing the 
    """
    # Locking as edits are made to cell_counter
    with cell_counter_rw_lock.writer():
        cell_counter[cell_type] += 1
        return cell_counter[cell_type] - dec

def reset_count(cell_type: Hashable) -> None:
    """
    Resets the counter for the given type.
    """
    with cell_counter_rw_lock.writer():
        del cell_counter[cell_type]


class Cell(ABC):
    """
    Abstract class to be extended by `Glom` and `Mitral`.
    """
    # # Augment add_count w/ this class's most-specific subclass.
    # def _add_count(self, start_0: bool = True) -> int:
    #     return add_count(self.__class__, dec=start_0)

    @property
    def id(self) -> int:
        """
        ID of the cell.

        As a property, any Integral type can be assigned to id, as it will be converted to an int.
        If set to None, will assign a new id using add_count. 
        """
        return self._id
    
    @id.setter
    def id(self, value: Optional[Integral]) -> None:
        if value is None:
            self._id = add_count((self.__class__))
        else:
            self._id = int(value)

    @property
    def activ(self) -> float:
        """
        Returns activation level of the cell.

        As a property, any Rational type can be assigned to activ, as it will be converted to a float.
        Rounds `value` and sets it to activation level.

        Parameters
        ----------
        value
            Value is a float between 0 and 1.
        """
        return self._activation

    @activ.setter
    def activ(self, value: Rational) -> None:
        N_DIGITS = 6
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activation = float(round(float(value), N_DIGITS))

    @property
    def loc(self) -> tuple[float, float]:
        """
        Returns location of the glom.

        As a property, any pair of Rational numbers can be assigned to loc, as it will be converted to a tuple[float, float].
        Sets value to loc.

        Parameters
        ----------
        value
            is a tuple of Rational numbers
        """
        return self._loc

    @loc.setter
    def loc(self, value: tuple[Rational, Rational]) -> None:
        assert len(value) == 2 and all(map(lambda x: isinstance(x, Rational), value)) and isinstance(value, tuple), "Not a pair of numbers!"
        self._loc = tuple(map(float, value))
    
    def __init__(self, id_: Optional[Integral], activ: Rational, loc: tuple[Rational, Rational]) -> None:
        """
        Initializes a cell with an id, activation level, and location.

        Parameters
        ----------
        id_ - Optional[Integral]
            the ID to assign the cell.
        activ - Rational
            The initial activation level of the cell.
        loc - tuple[Rational, Rational]
            The initial location of the cell.
        """
        # Internal field type hints
        self._id: int
        self._activ: float
        self._loc: tuple[float, float]

        # Assign properties, and override type hints
        self.id = id_
        self.activ = activ
        self.loc = loc
        

# TODO: update numpy docstrings
class Glom(Cell):
    """
    Represents a glomerulus cell that communicates with a single receptor.

    Attributes
    ----------
    id : int
        Identifies the glomerulus cell.
    activ : float
        Between (0,1) - activation level of glomerulus cell.
    loc : tuple[float, float]
        x,y coordinates of the glom on the surface of the Olfactory bulb
    dim : tuple[float, float]
        row x columns 
    conn : int
        Number of mitral cells connected to
    rec_conn_map : dict[Receptor, float]
        Mapping connecting receptors to their weights
    """

    @property
    def dim(self) -> tuple[int, int]:
        """
        Returns the dimensions of the glom.
        Sets dim to value.
        
        Parameters
        ----------
        value
            A length 2 list of numbers.
        """
        return self._dim

    @dim.setter
    def dim(self, value: tuple[Integral, Integral]) -> None:
        assert len(value) == 2 and isinstance(value[0], int), "Not a 2D list of numbers!"
        self._dim = tuple(map(int, value))
        
    @property
    def conn(self) -> int:
        """
        Returns conn of glom.
        Sets value to conn.
        
        Parameters
        ----------
        value - int
            Value to set conn.
        """
        return self._conn
    
    @conn.setter
    def conn(self, value: Integral) -> None:
        self._conn = int(value)
    
    @property
    def rec_conn_map(self) -> dict['Receptor', float]:
        return self._recConn
    
    @rec_conn_map.setter
    def rec_conn_map(self, value: Mapping['Receptor', Rational]):
        if config.DEBUG:
            try:
                from odorsampling.RnO import Receptor
            except ImportError as e:
                logger.error("Debug is True, but unable to import Receptor.")
                raise ValueError("DEBUG is True, but unable to import Receptor.") from e
        assert all(map(
            lambda k,v: isinstance(k, Receptor) and isinstance(v, Rational), value.items())
        ), "Not a mapping of Receptor->Rational"
        self._recConn = dict(map(lambda k,v: (k, float(v)), value.items()))
    

    def __init__(self, id_: Optional[int], activ: Rational = 0.0, loc: tuple[Rational, Rational] = (0,0),
                 dim: tuple[Integral, Integral] = (0,0), conn: Integral = 0):
        """
        Initializes the Glom object
        
        Parameters
        ----------
        id_
            The ID with which to initialize the glom cell.
        activ
            The initial activation level of the cell.
        loc
            The initial location of the cell.
        dim
            The initial dims of the cell.
        conn
            The initial conn of the cell.
        """
        super().__init__(id_, activ, loc)

        # Internal field type hints
        self._dim: tuple[float, float]
        self._conn: int
        self._recConn: dict['Receptor', float]

        # Assign properties
        self.dim = dim
        self.conn = conn
        self.rec_conn_map = {}

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return f"Id: {self.id} activ: {self.activ}"

    # TODO: Double check if implemented correctly
    @staticmethod
    def generate_random_loc(xBounds: tuple[int, int], yBounds: tuple[int, int]) -> tuple[int, int]:
        """
        Returns a random glom location.
        
        Parameters
        ----------
        xBounds
            Tuple of low-highs
        yBounds
            Tuple of low-highs
        """
        (xLowerBound, xUpperBound), (yLowerBound, yUpperBound) = xBounds, yBounds
        randomGlomX = utils.RNG.integers(xLowerBound, xUpperBound, endpoint = False)
        
        if randomGlomX == xLowerBound or randomGlomX == xUpperBound:
            randomGlomY = utils.RNG.integers(yLowerBound, yUpperBound, endpoint = False)
        else:
            randomGlomY = utils.RNG.choice([yLowerBound, yUpperBound])
        return randomGlomX, randomGlomY


class Mitral(Cell):
    """Represents a mitral cell that samples from glomeruli.
    Attributes
    ----------
    _id : int
        identifies the mitral cell
    _activ : float
        value between [0,1]: activation level of mitral cell
    _loc : list[float, float]
        coordinates of the mitral cell on the surface of the bulb
    _glom : dict
        where the keys are glom and the values are weights
    
    """
        
    @property
    def glom(self) -> dict[Glom, float]:
        """
        Returns dictionary of connected glom
        
        Sets glomeruli to value.
        Precondition: Value is a mapping containing glomeruli id's and weights.
        """
        return self._glom

    @glom.setter
    # Sets the `_glom` attribute to a new dictionary created from the provided mapping.
    def glom(self, value: Mapping[Glom, float]) -> None:
        self._glom = dict(value)


    def __init__(self, id_: Optional[int], activ: Rational = 0.0, loc: tuple[Rational, Rational]=(0.0,0.0),
                 glom: Optional[Mapping[Glom, float]] = None):
        """Initializes a Mitral cell"""
        # Call the superclass initializer
        super().__init__(id_, activ, loc)
        # Set the `glom` attribute, defaulting to an empty dictionary if `glom` is None
        self.glom = {} if glom is None else glom

    def __str__(self) -> str:
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        return f"Mitral ID: {self.id} Mitral Activ: {self.activ} Glom: {self.glom.keys()}"
