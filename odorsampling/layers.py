# Glomeruli and Mitral Layers and related functions
# Mitchell Gronowitz
# Spring 2015

# Edited by Christopher De Jesus
# Summer of 2023

"""
    This module builds layers of Glomeruli and Mitral cells, including connections between them.
    This includes:
        - Building glom layer and initializing activation levels
        - Building a list of similarly activated glom layers given a gl
        - Building mitral layer
        - Building connections and initializing mitral activation levels
        - Saving GL, MCL, or Maps
        - Euclidean distance between two layers
        - Graphical representations of a layer
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.pylab
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from odorsampling import config, cells, utils

# Used for asserts
from numbers import Rational

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Iterable, Optional


logger = logging.getLogger(__name__)
utils.default_log_setup(logger)

ConnMap = list[tuple[int, int, float]]
"""
Connections from Glom to Mitral cells, and their weights.
(m_id, g_id, weight)
"""


# Considering changing cells.Glom to GlomType = cells.Glom
#  to alias the structure- though not sure if worth it.
class GlomLayer(list[cells.Glom]):
    def __init__(self, cells: Iterable[cells.Glom] = tuple()):
        super().__init__(cells)
    
    @classmethod
    def create(cls, n: int, reset_id_count: bool = True) -> GlomLayer:
        # TODO: Implement count reset
        # if reset_id_count:
        #     cells.reset_count(cells.Glom)
        logger.debug("Creating glom layer of %s cells.", n)
        return cls((cells.Glom(i) for i in range(n)))
    
    @classmethod
    def createGL_dimensions(cls, x: int, y: int) -> GlomLayer:
        """
        Returns an array of x number of glom objects with activation levels
        and loc set to defaults. ID refers to its index in the array.
        
        PARAMETERS
        ----------
        x - int
        
        y - int

        """
        assert isinstance(x, int), "x is not an int"
        assert isinstance(y, int), "y is not an int"
        return cls(
            cells.Glom(None, 0.0, [countX, countY], [y, x], 0)
            for countY in range(y) for countX in range(x)
        )


    def clear_activations(self) -> None:
        for glom in self:
            glom.activ = 0.0
            glom.rec_conn_map = {}
        logger.debug("Glom cell layer activations cleared.")

    #For now, if a number is generated to be over 1 or under 0 in Gaussian or
    #exponential, the function will be called again to generate a different number.
    # TODO: Change sel to an enum or some other fixed type
    def activate_random(self, dist_func: utils.DistributionFunc = utils.uniform_activation, **kwargs) -> None:
        """
        Initializes the activation levels for the glom layer instance.

        PARAMETERS
        ----------
        kwargs
            The function used to get the value of the activation levels of the gloms.
            Ensure that you pass the appropriate arguments for a given distribution function, or a
            TypeError is raised. Expects `utils.uniform_activation`, `utils.gaussian_activation`, or 
            `utils.expovar_activation`, however it is capable of handling other distributions.

            Examples of parameters dist_func takes are:
            - `mu` [aliased as `mean`], `sigma` [aliased as `sd`] - Used by gaussian distributions. Note:
                the aliasing used is one-way- for example if `mu` is not defined, it will have the value of `mean`
                if it is defined. Otherwise, no default is set, and the method fails with a TypeError as expected.
            - `a`, `b` (defaults to 0., 1. respectively) - Used by uniform distributions for the range.
            - `lambd` (defaults to 1/`mean`, if `mean` is defined) - Used by expovariate.
            
            See utils.DistributionFunc for more details.
        
        """
        
        # defaults
        utils.init_dist_func_kwargs(kwargs)

        for glom in self:
            try:
                glom.activ = dist_func(**kwargs)
            except TypeError as e:
                raise TypeError(
                    "Invalid keyword argument passed to activation_func. See error that raised this error."
                ) from e
        logger.info("Glom cell layer activation levels initialized to `%s`.", dist_func.__name__)
    #For now, any number that is incremented to be over 1 or under 0 is just set
    # to 1 or 0 respectively.
    def activate_similar(self, dist_func: utils.DistributionFunc = utils.uniform_activation, new_ = True, **kwargs) -> GlomLayer:
        """Returns a glomeruli layer with activation levels similar to gl, by randomly
        picking a number between -num and num and incrementing (or if gaussian then picked
        using mean and sd). If new_ is True, then create new gl, otherwise this method fills
        its member object with the generated activation levels.
        
        PARAMETERS
        ----------
        dist_func
            See utils.DistributionFunc
        
        new_
            Whether to create a new layer or not
        
        kwargs
            mapping of arguments for dist_func.
        
        preconditions: num is between 0 and 1, sel is 'g' for gaussian or 'u' for uniform"""
        # FIXME: This assertion fails: why?
        # assert kwargs.get('a', 1) > 0 and kwargs.get('b', 0) < 1, "num must be between 0 and 1" # auto-succeed if not defined.
        assert dist_func in (utils.uniform_activation, utils.choice_gauss_activation)

        # defaults
        utils.init_dist_func_kwargs(kwargs)

        gl2 = GlomLayer.create(len(self)) if new_ else self

        for i, glom in enumerate(gl2):
            try:
                glom.activ = min(max(dist_func(**kwargs) + self[i].activ, 0.0), 1.0)
            except TypeError as e:
                raise TypeError(
                    "Invalid keyword argument passed to activation_func. See error that raised this error."
                ) from e

        logger.info("Activate GLSimilar called on glom layer.")
        return gl2
    
    #Creating array of GL's with similar activation
    @staticmethod
    def create_array(glom_layer: GlomLayer, n_layers: int, opt: str, dist_func: utils.DistributionFunc, num: float, mean=0, sd=0) -> list[GlomLayer]:
        """Given a glomeruli layer, returns x amount of similar gl's using
        the similarity method specified with opt and sel. Original activ lvl is incremented by <= num.
        Preconditions: gl is a list of glomeruli, x is an int sel is star or ser"""
        assert type(n_layers) == int, "x is not an int"
        #Everything else is asserted in helper function below
        if not n_layers: # 0
            logger.debug("createGLArray called with x=0.")
            return []
        new_glom = glom_layer.activate_similar(a=-num, b=num, dist_func=dist_func, mean=mean, sd=sd)
        snd_gl = glom_layer if opt == 'star' else new_glom
        logger.info("GlomLayer Array created with depth %s using sel param %s and opt %s.", n_layers, dist_func, opt)
        return [new_glom] + GlomLayer.create_array(snd_gl, n_layers-1, opt, dist_func, num, mean, sd)
    
    #####Loading and Storing a GL, MCL, and Map
    def save(self, name: str):
        """Saves GL as a file on the computer with .GL as extention
        Precondtion: GL is a valid gl and name is a string."""
        assert type(name) == str, "name is not a string"
        content = "\n".join((f"{glom.id},{glom.activ},{glom.loc[0]}:{glom.loc[1]},{glom.conn};" for glom in self))
        filename = f"{name}.{config.GL_EXT}"
        with open(filename, 'w') as f:
            logger.info("Glom layer saved to `%s`.", filename)
            f.write(content)

    @classmethod
    def load(cls, name: str) -> GlomLayer:
        """Returns GL with given name from directory
        precondition: name is a string with correct extension"""
        assert type(name) == str, "name isn't a string"
        pattern = re.compile(r"(\d+),(\d+),(\d+):(\d+),(\d+);")

        glom_layer = []
        # TODO: regex it, and use a generator into GlomLayer constructor
        with open(name) as f:
            for line in f.readlines():
                data = pattern.match(line)
                if not data:
                    continue
                glom_layer.append(
                    cells.Glom(int(data.group(1)), float(data.group(2)),
                    (int(data.group(3)), int(data.group(4))),
                    int(data.group(5)))
                )
        logger.info("Glom layer loaded from `%s`.", name)
        return cls(glom_layer)
    
    def addNoise(self, dist_func: utils.DistributionFunc, **kwargs):
        """Increments activation levels in GL by a certain value
        If noise is 'u', then mean = scale for uniform distribution."""

        # hotfix
        try:
            kwargs.setdefault('b', kwargs['mean'])
        except KeyError:
            logger.info("No mean supplied, not setting special defaults for normal noise distribution.")
        utils.init_dist_func_kwargs(kwargs)

        inc = dist_func(**kwargs)
        for g in self:
            g.activ = min(max(g.activ + utils.RNG.choice([1,-1])*inc, 0.0), 1.0)
        logger.info("Added noise[%s] to Glomlayer.", dist_func)
        return self

    def connect_mitral(self, mitral_layer: MitralLayer, s=1):
        """
        Replacement for _prevDuplicates as that implementation is over-complex.
        
        PARAMETERS
        ----------
        mitral_layer
            The MitralLayer to which this GlomLayer is connecting.
        s
        """
        # conn: list[int] = [] # prolly unnecessary
        # TODO: Use https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        #   to select mitral indexes
        
        # utils.RNG.choice(len(mitral_layer), size=len(mitral_layer), replace=False)
        return False

        
# NOTE: Make sure that this was replaced properly
    # def _prevDuplicates(self, num: int, conn: list[int], weights=None, s=1):
    #     """If a mitral cell already connects to glom at index num, then pick
    #     a new number. To prevent infinite loop, if a certain number of loops
    #     occur, just allow duplicate but print a warning message.
        
    #     PARAMETERS
    #     ----------
    #     num
    #         The index to connect
    #     conn
    #         The connections to check
    #     weights
    #         None by default, if supplied, bias towards the weights.
    #     s
    #         #TODO: What is this used for specficially?
    #     """
    #     # NOTE: Uniform, no-replacement
    #     # TODO: Find simpler, uniform way to select these. I know there is one. Above is WIP
        
    #     MAX_CHECKS = 100
    #     check = 0
    #     if weights is None:
    #         while num in conn and check < MAX_CHECKS:
    #             num = utils.RNG.integers(0,len(self))
    #             check += 1
    #     else:
    #         while num in conn and check < MAX_CHECKS:
    #             num = 0
    #             rand = utils.RNG.integers(1, s, endpoint=True)
    #             while rand > 0:                 #Picking an index based on weight
    #                 rand = rand - weights[num]
    #                 num += 1
    #             num -= 1
    #             check += 1
    #     if check == MAX_CHECKS:
    #         logger.warning("Mitral cell may be connected to same Glom cell twice in order to prevent infinite loop")
    #     return num
    
    # def _check_dups(self, num, conn, weights=None):
    #     """Checks if a mitral cell already connects to glom at index num"""
    #     if weights is None:
            

    #     return 

    def _buildWeights(self, bias: str, scale: int) -> list[int]:
        """Returns a list len(gl), with each index starting with the same number
        which depends on bias"""
        weights: list[int] = []
        counter = 0
        if bias == "lin":
            while counter < len(self):
                weights.append(scale)
                counter += 1
        else:
            while counter < len(self):
                weights.append(2**scale)
                counter += 1
        return weights

#####Graphing
    def graph_activation(self, n, m) -> None:
        graph = [[0,0,0],[0,0,0],[0,0,0],[0,0.5,0],[0.0,1.0,0.0],[0,0.4,0],[0,0,0.4],[0,0,1],[0,0,0.8]]
        plt.imshow(graph, cmap=matplotlib.pylab.cm.YlOrRd, interpolation='nearest', origin='lower', extent=[0,3,0,3])
        plt.title("Glom Activation")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        pp = PdfPages("GlomActivation.pdf")
        pp.savefig()
        pp.close()

        plt.close()

    #How do weights play in? Right now I just do activlvl*weight
    def activate_mcl(self, mcl: MitralLayer, sel, map_=None, noise=None, mean=0, sd=0) -> None:
        """Builds glom connections to mitral cells and calculates mitral activ lvl based on
        connections and weights. Sel decides how to calculate the values, and noise
        adds some variation.
        **If noise = "u" then mean is the the scale for uniform distribution of 0 to mean.
        Preconditions: Map holds valid connections for GL and MCL if not empty.
        Sel = "add", "avg" or "sat". Noise = None, u, g, or e."""
        assert sel in ['add', 'avg', 'sat'], "select value isn't valid"
        assert noise in [None, 'u', 'g', 'e'], "noise isn't a valid string"
        #Build MCL - GL connections
        if map_ is not None:
            mcl, gl = apply_sample_map(gl, mcl, map_)
        #Add noise
        if noise is not None:
            gl = gl.addNoise(noise, mean=0, sd=0)
        #Activate Mitral cell activ lvls in MCL
        if sel == 'add' or sel == 'avg':
            for m in mcl:
                activ = addActivationMCL(m, gl)
                if sel == 'avg':
                    activ = activ/(len(m.glom))
                m._activation = activ   #Bypassing assertion that activ lvl < 1 TODO:<-- is this ok?
            # MCL = normalize(MCL)
        if sel == 'sat':
            pass

class MitralLayer(list[cells.Mitral]):
    def __init__(self, cells: Iterable[cells.Mitral]):
        super().__init__(cells)

    @classmethod
    def create(cls, n: int, reset_id_count: bool = True):
        assert isinstance(n, int)
        if reset_id_count:
            cells.reset_count(cells.Mitral)
        logger.debug("Creating mitral layer of %s cells.", n)
        return cls((cells.Mitral(i, 0.0, (0, 0), {}) for i in range(n)))
    
    def save(self, name: str):
        """Saves MCL as a file on the computer with .MCL as extention.
        Precondition: Map is a valid map and name is a string."""
        assert type(name) == str, "name is not a string"
        st = ""
        for m in self:
            # TODO: redo with regex and pythonically
            x, y = loc = m.loc
            loc = str(x) + ":" + str(y)
            #Storing glom
            glom = str(m.glom.items())
            end = glom.index("]")
            i = 0
            s = ""
            while i < end-1:
                ind1 = glom.index("(", i)
                comma = glom.index(",",ind1)
                ind2 = glom.index(")", ind1)
                s = s + glom[ind1+1:comma] + "|" + glom[comma+2:ind2] + "+"
                i = ind2
            st = f"{st}{m.id},{m.activ},{loc},{s};\n"
        filename = f"{name}{config.ML_EXT}"
        with open(filename, 'w') as f:
            logger.info("Mitral layer saved to `%s`.", filename)
            f.write(st)
    
    @classmethod
    def load(cls, name: str) -> MitralLayer:
        """Returns MCL with given name from directory.
        precondition: name is a string with correct extension"""
        assert type(name) == str, "name isn't a string"
        pattern = re.compile(r"(\d+),(\d+),(\d+):(\d+),(\d+);")

        mcl = []
        with open(name, 'r') as f:
            for line in f.readlines():
                data = pattern.match(line)
                if not data:
                    continue
                mcl.append(
                    cells.Mitral(int(data.group(1)), float(data.group(2)),
                    (int(data.group(3)), int(data.group(4))),
                    int(data.group(5)))
                )
            return cls(mcl)

    #****For now all weights are chosen uniformly
    #connections are either fixed or cr serves as the mean with a sd for amount of connections
    def createSamplingMap(self, gl: GlomLayer, cr: Rational, fix: bool, sel: str, sd=0, bias='lin') -> ConnMap:
        """Returns a map in the form of [[Mi, G, W],[Mi, G, W]...] where
        Mi and G are ID's and W is the weight.
        1. The convergence ratio (cr) determines the amount of glom sample to each
        mitral cell. Fix determines whether the cr is a fixed number or just the mean
        with sd.
        2. Sampling can be done randomly or biased (decided by sel).
        3. If biased, sampling can be done with linear bias or exponential.
        There may also be a cleanup function at the end.
        preconditions: fix is a boolean, cr is an int or float less than len(gl),
        bias is "lin" or "exp", and sel is "bias", "simple", "balanced", or "location"."""
        assert type(fix) == bool, "Fix is not a boolean"
        assert sel in ['bias', 'simple', 'balanced', 'location'], "Sel is not a specific type!"
        assert type(cr) in [int, float] and cr + sd <= len(gl), "cr isn't a valid number"
        assert bias in ['lin', 'exp']

        if cr == 1 and fix == True:  #Needed to avoid two Mitral cells sampling the same Glom
            logger.debug("MCLSamplingMap with oneToOneSample.")
            return self.oneToOneSample(gl)
        elif sel == 'simple':
            logger.debug("MCLSamplingMap with simpleSampleRandom.")
            return self.simpleSampleRandom(gl, cr, fix, sd)
        elif sel == 'balanced':
            logger.debug("MCLSamplingMap with simpleSampleBalanced.")
            assert (len(self)*cr)%len(gl) == 0, "cannot balance. recreate mitrals."
            return self.simpleSampleBalanced(gl, cr, fix, sd)
        elif sel == 'location':
            logger.debug("MCLSamplingMap with simpleSampleLocation.")
            assert gl[0].dim[0]>=3 and gl[0].dim[1]>=3, "glom need to be at least 3X3. recreate gloms."
            return self.simpleSampleLocation(gl, cr, fix, sd)
        else: #sel == 'biased'
            logger.debug("MCLSamplingMap with biasSample.")
            return self.biasSample(gl, cr, fix, bias, sd)
        #Can call a clean up function here if we want
        #print(unsampledGlom(gl, self, Map)               #PRINTING HERE

# TODO: rewrite samplers

    def oneToOneSample(self, gl: GlomLayer) -> ConnMap:
        """
        For 1:1 sampling, each mitral cell chooses an unselected glom cell
        Precondition: len of gl >= len of mcl
        
        Returns
        -------
        ConnMap
            The mitral_id, glom_idx, and weights for each mitral cell
        """
        assert len(gl) >= len(self)
        indexes = utils.RNG.choice(len(gl), len(gl), replace=False)
        map_ = [
            (mitral.id, indexes[i], 1) for i, mitral in enumerate(self)
            # ******Changed for weights to always be 1
        ]
        return map_

    def simpleSampleRandom(self, gl: GlomLayer, cr: float, fix: bool, sd: Optional[float] = 0) -> ConnMap:
        """
        Builds a map by randomly choosing glomeruli to sample to mitral cells.
        If fix != true, cr serves as mean for # of glom sample to each mitral cell.
        Weights are randomly chosen uniformly.
        ***Weights of a mitral cell's gloms add up to 1.0***
        """

        map_ = []
        for mitral in self:
            gloms_to_choose = cr if fix else int(max(utils.RNG.normal(cr, sd), 1))
            gloms: list[cells.Glom] = list(utils.RNG.choice(gl, gloms_to_choose, replace=False))
            if fix:
                leftover, weight = 1, 0
                for glom in gloms:
                    weight = utils.RNG.uniform(0, leftover)
                    leftover -= weight
                    map_.append((mitral.id, glom.id, weight))
                # Correct leftovers
                try:
                    map_[-1] = (map_[-1][0], map_[-1][1], map_[-1][2]+leftover)
                except IndexError:
                    logger.error("Expected cr>0. cr=%s. Returning empty map.", cr)
                    return map_
            else:
                map_.extend((
                    (mitral.id, glom.id, utils.RNG.uniform(0, .4)) for glom in gloms
                ))
        return map_


    def simpleSampleBalanced(self, gl: GlomLayer, cr: float, fix: bool = True,
                             sd: float = 0) -> ConnMap:
        """
        Builds a map by randomly choosing glomeruli to sample to mitral cells.
        If fix != true, cr serves as mean for # of glom sample to each mitral cell.
        Weights are randomly chosen uniformly. Limits number of mitral cells that 
        glom can project to `(#MC * cr) / #Glom` aka `fanout_ratio`.
        ***Weights of a mitral cell's gloms add up to 1.0***

        PARAMETERS
        ----------
        gl
            The GlomLayer to index glom cells from.
        cr
            Mean # of gloms to sample to each mitral cell.
        fix
            Unused
        sd
            Unused
        """
        map_ = []
        fanout_ratio = (len(self) * cr)/len(gl) #fanout ratio
        # ^ number of entries per id
        glom_ids = [glom.id for glom in gl]
        density_map = Counter({id_: fanout_ratio for id_ in glom_ids})

        for mitral in self:
            leftover, weight = 1, 0
            
            for _ in range(cr):
                # Choose a glom cell
                d_map_sum = sum(density_map.values()) # num of glom 
                prob_map = [density_map[idx]/d_map_sum for idx in glom_ids]
                chosen_glom: int = utils.RNG.choice(glom_ids, p=prob_map)
                density_map[chosen_glom] -= 1
                # Get weight for connection
                weight = utils.RNG.uniform(0, leftover)
                leftover -= weight
                
                map_.append((mitral.id, chosen_glom, weight))
            # Correct leftovers by patching last in map
            try:
                map_[-1] = (map_[-1][0], map_[-1][1], map_[-1][2]+leftover)
            except IndexError:
                logger.error("Expected cr>0. cr=%s. Returning empty map.", cr)
                break
        return map_

    def simpleSampleLocation(self, glom_layer: GlomLayer, cr, fix: Optional[bool] = True, sd=0) -> ConnMap:
        """
        Builds a map by randomly choosing glomeruli to sample to mitral cells.
        If fix != true, cr serves as mean for # of glom sample to each mitral cell.
        Weights are randomly chosen uniformly. Glomeruli are drawn randomly from the
        surrounding glomeruli that surround the parent glomerulus.
        ***Weights of a mitral cell's gloms add up to 1.0***
        """
        map_ = []

        # TODO: Fix "magic" numbers
        num_layers = math.ceil((-4+math.sqrt(16-16*(-(cr-1))))/8) # Where do these values come from? (4, 16, 8)
        # logger.debug("numLayers: " + str(numLayers))
        num_to_select = int((cr-1) - (8*(((num_layers-1)*num_layers)/2)))  # FIXME: Where do these values come from? (8, 2)
        # logger.debug("numToSelect: " + str(int(numToSelect))) # number to select in the outermost layer
        # logger.debug("dimensions: " + str(gl[0].dim[0]) + "X" + str(gl[0].dim[1]))

        # if fix:
        for mitral in self:
            gl_index = utils.RNG.integers(0,len(glom_layer))
            x, y = glom_layer[gl_index].loc
            x_bounds, y_bounds = (x-num_layers,x+num_layers), (y-num_layers,y+num_layers)
            
            gloms = []
            weight = utils.RNG.uniform(0,1)
            map_.append((mitral.id, gl_index, weight))

            leftover = 1-weight
            if int(num_layers) == 1:
                glom_loc = cells.Glom.generate_random_loc(x_bounds, y_bounds)
                for selected in range(num_to_select):
                    weight = utils.RNG.uniform(0, leftover)
                    leftover -= weight
                    # TODO: Make this deterministic
                    while glom_loc in gloms:
                        glom_loc = cells.Glom.generate_random_loc(x_bounds, y_bounds)
                    gloms.append(glom_loc)
                    selected += 1
                    for g in glom_layer:
                        if g.loc == (glom_loc[0]%(glom_layer[0].dim[1]), glom_loc[1]%(glom_layer[0].dim[0])):
                            gl_index = g.id
                    # print(f"GlomID: {num}")
                    map_.append((mitral.id, gl_index, weight))
                # Correct leftovers by patching last in map
                try:
                    map_[-1] = (map_[-1][0], map_[-1][1], map_[-1][2]+leftover)
                except IndexError:
                    logger.error("Expected cr>0. cr=%s. Returning incomplete map.", cr)
                    break
            elif int(num_layers) == 2:
                # print("numLayers == 2")
                # get first layer first (the surrounding 8)
                (x_inner, x_outer), (y_inner, y_outer) = (x-1, x+1), (y-1, y+1)
                # Construct firstLayer
                firstLayer: list[tuple[int, int]] = []
                while x_inner <= x_outer:
                    y_inner = y-1
                    while y_inner <= y_outer:
                        if (x_inner, y_inner) != (x, y):
                            firstLayer.append((x_inner%(glom_layer[0].dim[1]), y_inner%(glom_layer[0].dim[0])))
                        y_inner += 1
                    x_inner += 1

                # Iterate over firstLayer and glom cells
                for gl_dim1, gl_dim0 in firstLayer:
                    for g in glom_layer:
                        # NOTE: Why are we checking gl_dim1 and gl_dim0 specifically?
                        if g.loc == (gl_dim1, gl_dim0):
                            gl_index = g.id
                            weight = utils.RNG.uniform(0, leftover)
                            leftover -= weight
                            # print(f"GlomID: {num}")
                            map_.append((mitral.id, gl_index, weight))
                        # else:
                        #     logger.debug("glom locs don't match: ", g.loc, " and ", (a[0], a[1]))

                # second layer
                glom_loc = cells.Glom.generate_random_loc(x_bounds, y_bounds)
                for selected in range(num_to_select):
                    # Recalc act
                    if selected == num_to_select-1:
                        weight = leftover
                    else:
                        weight = utils.RNG.uniform(0, leftover)
                        leftover -= weight
                    # Pick glom until unique
                    while glom_loc in gloms:
                        glom_loc = cells.Glom.generate_random_loc(x_bounds, y_bounds)
                    gloms.append(glom_loc)
                    # update gl_index for `map_`
                    for g in glom_layer:
                        # if g.loc == randomGlom:
                        if g.loc == (glom_loc[0]%(glom_layer[0].dim[1]), glom_loc[1]%(glom_layer[0].dim[0])):
                            # print(randomGlom)
                            gl_index = g.id
                        # else:
                        #     logger.debug("glom locs don't match: ", g.loc, " and ", (randomGlom[0]%(gl[0].dim[1]), randomGlom[1]%(gl[0].dim[0])))

                    map_.append((mitral.id, gl_index, weight))
            # FIXME: What about `if not fix` (else)? (Empty map returned)
        return map_
 

    def biasSample(self, gl: GlomLayer, cr, fix, bias: str, sd=0) -> ConnMap:
        """Builds a map by choosing glomeruli to sample to mitral cells, but the more times
        a glomeruli is sampled, the less likely it is to be chosen again (either a linear
        degression or exponential based on bias). If fix != true, cr serves as mean for
        # of glom sample to each mitral cell. Weights are randomly chosen uniformly."""
        # NOTE: cr = "Connection Ratio"
        map_ = []
        #determine scale
        MIN_SCALE, SCALE_FACTOR = 7, 1.7
        scale = max(MIN_SCALE, math.floor((len(self)/len(gl)) * SCALE_FACTOR * cr))
        #Build weights
        weights = gl._buildWeights(bias, scale)
        s = len(weights)*scale if bias.lower()=="lin" else len(weights)*(2**scale)
        
        # TODO: Consider whether this is necessary.
        def _select_glom(rand, weights, glom_weight_idx):
            """
            Steps through indexes until weights are exhausted.
            """
            while rand > 0:
                rand = rand - weights[glom_weight_idx]
                glom_weight_idx += 1
            return glom_weight_idx
        cr_orig = cr
        # FIXME: I broke it. Crashes on _recalcWeights call (IndexError)
        for mitral_idx in range(len(self)):  # start looping through each mitral cell
            if not fix: # Generate new cr per mitral `if not fix``.
                cr = min(max(int(utils.RNG.normal(cr_orig, sd)), 1), len(gl))
            gl_indexes: list[int] = []
            for _ in range(cr): # start connecting mitral cell to (multiple) glom
                glom_weight_idx = _select_glom(utils.RNG.integers(1, s, endpoint=True), weights, 0)
                while glom_weight_idx in gl_indexes:
                    glom_weight_idx = _select_glom(utils.RNG.integers(1, s, endpoint=True), weights, glom_weight_idx) - 1
        
                # glom_weight_idx = gl._prevDuplicates(glom_weight_idx, gl_indexes, weights, s)
                map_.append((mitral_idx, glom_weight_idx, utils.RNG.uniform(0,.4)))
                gl_indexes.append(glom_weight_idx)
                weights, s = _recalcWeights(weights, glom_weight_idx, bias, s)
            # print(f"Selected: {gl_indexes}")
        return map_

    def graph_activation(self, gl: GlomLayer, n, m):
        logger.info("Graphing mitral activation")
        mitralLocations = {}
        mitralActivations: dict[str, list[float]] = {}
        for mitral in self:
            if mitral.loc in mitralLocations:
                val = mitralLocations.get(str(mitral.loc))
                activ = mitralActivations.get(str(mitral.loc))
                mitralLocations.update({str(mitral.loc):val+1})

                activ.append(mitral.activ)
                mitralActivations.update({str(mitral.loc):activ})
            else:
                mitralLocations.update({str(mitral.loc):1})
                mitralActivations.update({str(mitral.loc):[mitral.activ]})

        maxMitrals = mitralLocations.get(max(mitralLocations, key=mitralLocations.get))

        graph: list[list[int]] = []
        counter = 0
        while counter < m*maxMitrals:
            graph.append([])
            counter1 = 0
            while counter1<m:
                graph[counter].append(0)
                counter1 += 1
            counter += 1

        for x in range(m):
            for y in range(len(graph)):
                key = str((x,math.floor(y/maxMitrals)))
                if key in mitralActivations:
                    numActivations = len(mitralActivations.get(key))
                    if numActivations == 1:
                        graph[y][x] = mitralActivations.get(key)[0]
                    elif numActivations < maxMitrals:
                        if y%maxMitrals < numActivations:
                            graph[y][x] = mitralActivations.get(key)[y%maxMitrals]
                        else:
                            graph[y][x] = -0.15
                    else:
                        # print(mitralActivations.get(str([x,y/maxMitrals]))[y%(len(mitralActivations.get(str([x,y/maxMitrals]))))])
                        graph[y][x] = mitralActivations.get(key)[y%(len(mitralActivations.get(key)))]

        # print(graph)  

        #   https://stackoverflow.com/questions/22121239/matplotlib-imshow-default-colour-normalisation 

        fig, _ = plt.subplots()

        im = plt.imshow(graph, cmap=matplotlib.pylab.cm.YlOrRd, interpolation='nearest', vmin=-0.15, vmax=1, origin='lower', extent=[0,4,0,4])
        plt.title("Mitral Activation")
        plt.xlabel("X")
        plt.ylabel("Y")

        fig.colorbar(im)

        pp = PdfPages("MitralActivation.pdf")
        pp.savefig()
        pp.close()

        plt.close()


# TODO: figure out types
def _recalcWeights(weights, index: int, bias, s):
    """Readjusts and returns weights and sum as a 2d list [weights, sum].
    If index is too low, all inputs in weights are increased"""
    if bias == 'lin':
         weights[index] = weights[index] - 1
         s = s-1
    else:
        weights[index] = weights[index]/2
        s = s - weights[index]
    if weights[index] == 1:
        if bias == 'lin':
            for num in range(len(weights)):
                weights[num] = weights[num] + 3
                s = s+3
        else: #bias is exp
            for num in range(len(weights)):
                x = weights[num]
                weights[num] = x*4
                s = s + x*4 - x
    return (weights, s)



# TODO: tidy up


def saveMCLSamplingMap(Map, name: str):
    """Saves Map as a file on the computer with .mapGLMCL as extention.
    Precondition: Map is a valid map and name is a string."""
    assert type(name) == str, "name is not a string"
    # TODO: name the temp vars better
    content = "\n".join((f"{x},{y},{z};" for x, y, z in Map))
    filename = f"{name}{config.MCL_MAP_EXT}"
    with open(filename, "w") as f:
        f.write(content)

# TODO: Change type of Map to list[tuple[int, int, float]]
def loadMCLSamplingMap(name: str) -> list[tuple[int, int, float]]:
    """Returns MCL map with given name from directory. Weight value is cut off
    to the 15th decimal.
    precondition: name is a string with correct extension"""
    assert type(name) == str, "name isn't a string"
    map_ = []
    with open(name, 'r') as f:
        # TODO: regit
        for line in f.readlines():
            comma1 = line.index(",")
            comma2 = line.index(",",comma1+1)
            semi = line.index(";", comma2+1)
            map_.append((int(line[0:comma1]), int(line[comma1+1:comma2]), float(line[comma2+1:semi])))
    return map_


########Connnecting GL, MCL, and Map altogether


# TODO: Ensure Map is always a list[list[int]]. Assert?
def apply_sample_map(gl: GlomLayer, mcl: MitralLayer, map_: list[list[int]]) -> tuple[MitralLayer, GlomLayer]:
    """Fills the connection details and weights for GL and MCL for the given Map.
    Returns updated MCL and GL as [MCL, GL]
    precondition: Map holds valid connections for GL and MCL"""
    assert map_[len(map_)-1][0] == len(mcl)-1, "dimensionality of Mitral cells is wrong"
    test = 0
    mcl[map_[0][0]].loc = gl[map_[0][1]].loc

    for mitral_id, glom_id, weight in map_:
        if mitral_id != test:
            test += 1
            mcl[mitral_id].loc = gl[glom_id].loc
            mcl[mitral_id].glom = {}
        mcl[mitral_id].glom[glom_id]=weight

        gl[glom_id].conn += 1

    return (mcl, gl)


# FIXME: What is GL if not a list of glom cells?
def addActivationMCL(m: cells.Mitral, gl: GlomLayer):
    """Returns updated MCL where each mitral cell's activation level is calculated
    based on adding connecting glom activation levels*weight of connection"""
    glom = m.glom.keys()
    activ: int = 0
    for g in glom:
        # FIXME: Critical- oh no. The GLs are maps aren't they? (after checking, they shouldn't be)
        temp = gl[g].activ*m.glom[g]  #Activ lvl of attached glom * weight
        activ = activ + temp
    return activ


def normalize(mcl: MitralLayer):
    """Given a list of Glom or PolyMitral objects the
    function will scale the highest activation value up to 1
    and the other values accordingly and return updated MCL.
    If uncomment, then firt values will scale to 0 than up to 1.
    Precondition: No activation values should be negative"""
    max_i = 0
    #mini = 100
    for m in mcl:
        assert m.activ >= 0, "Activation value was negative!"
        if max_i < m.activ:
            max_i = m.activ
    if max_i != 0:
        scale = (1.0/max_i)  #If put mini back in then this line is 1/(maxi-mini)
        for m in mcl:
            m.activ = m.activ*scale  #Assertion now in place - all #'s should be btwn 0 and 1
    return mcl

###### Analysis and Visualization

def euclideanDistance(layer1: Union[GlomLayer, MitralLayer], layer2: Union[GlomLayer, MitralLayer]):
    """Returns Euclidean distance of activation levels between two layers
    of mitral or glom cells.
    Precondition: Layers are of equal length"""
    assert len(layer1) == len(layer2), "Lengths are not equal"
    index = 0
    num = 0.0
    while index < len(layer1):
        num = num + (layer1[index].activ - layer2[index].activ)**2
        index += 1
    return math.sqrt(num)

# TODO: Turn Union[cells.Glom, cells.Mitral] into TypeAlias
def graphLayer(layer: Union[GlomLayer, MitralLayer], sort=False):
    """Returns a graph of Layer (GL or MCL) with ID # as the x axis and Activ
    level as the y axis. If sort is true, layer is sorted based on act. lvl.
    Precondition: Layer is a valid GL or MCL in order of ID with at least one element"""
    assert layer[0].id == 0, "ID's are not in order!"
    l = len(layer)
    assert l > 0, "length of layer is 0."
    if sort:
        sel_sort(layer)
    x = range(l)   #Creates a list 0...len-1
    index = 0
    y = []
    while index < l:
        y.append(layer[index].activ)
        index += 1
    plt.bar(x, y)
    if type(layer[0]) == cells.Glom:
        plt.title("Activation Levels for a Given Glomeruli Layer")
    else:
        plt.title("Activation Levels for a Given Mitral Layer")    
    plt.ylabel("Activation Level")
    plt.xlabel("Cells")
    plt.show()

def sel_sort(layer):
    """sorts layer(sel sort) from highest act lvl to lowest.
    Precondition: Layer is a valid GL or MCL"""
    i = 0
    length = len(layer)
    while i < length:
        index = _max_index(layer, i, length-1)
        _swap(layer, i, index)
        i += 1

def _max_index(layer: Union[GlomLayer, MitralLayer], i: int, end: int):
    """Returns: the index of the minimum value in b[i..end]
    Precondition: b is a mutable sequence (e.g. a list)."""
    maxi = layer[i].activ
    index = i
    while i <= end:
        if layer[i].activ > maxi:
            maxi = layer[i].activ
            index = i
        i += 1
    return index


def _swap(b,x,y):
    """Procedure swaps b[h] and b[k]
    Precondition: b is a mutable sequence (e.g. a list).
    h and k are valid positions in b."""
    tmp = b[x]
    b[x] = b[y]
    b[y] = tmp

def colorMapWeights(map_, gl: GlomLayer, mcl: MitralLayer):
    """Builds a colormap with MCL on y axis and GL on x axis while color=weights"""
    #Create graph with all 0's
    graph = [[0 for _ in gl] for _ in mcl]
    # print(map_)
    #Add weights
    index = 0
    while index < len(map_):
        row = map_[index][0]
        col = map_[index][1]
        graph[row][col] = map_[index][2]
        index += 1
    matplotlib.pylab.matshow(graph, fignum="Research", cmap=matplotlib.pylab.cm.Greys) #Black = fully active
    plt.title("Weights in GL-MCL connection")
    plt.xlabel("GL")
    plt.ylabel("MCL")
    plt.show()
    

"""
To do list:
1. Fix Bias algorithm in biasSample() (Part of building the Map) so there's no jump in logic (need new algorithm)
2. Good saturation function for activating MCL based on connections to GL.
3. Optimize sorting algorithm to heap sort if using it often
"""