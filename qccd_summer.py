import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Callable,
    Union,
    Any,
    Set,
    Dict,
)
from matplotlib import pyplot as plt
import networkx as nx
import enum
from matplotlib.patches import Ellipse
import abc
import stim
from scipy.spatial import distance



class Operations(enum.Enum):
    """
    Trapped Ion QIP ToolBox
    """

    SPLIT = enum.auto()
    MOVE = enum.auto()
    MERGE = enum.auto()
    GATE_SWAP = enum.auto()
    CRYSTAL_ROTATION = enum.auto()
    ONE_QUBIT_GATE = enum.auto()
    TWO_QUBIT_MS_GATE = enum.auto()
    JUNCTION_CROSSING = enum.auto()
    MEASUREMENT = enum.auto()
    QUBIT_RESET = enum.auto()
    RECOOLING = enum.auto()
    PARALLEL = enum.auto()



class QCCDComponent:
    @property
    @abc.abstractmethod
    def pos(self) -> Tuple[float, float]: ...

    @property
    @abc.abstractmethod
    def idx(self) -> int: ...

    @property
    @abc.abstractmethod
    def allowedOperations(self) -> Sequence[Operations]:
        ...


class Ion(QCCDComponent):
    def __init__(self, color: str = "lightblue", label: str = "Q") -> None:
        self._idx: int = 0
        self._positionX: int = 0
        self._positionY: int = 0
        self._parent: Optional[Union["QCCDNode", "Crossing"]] = None
        self._color = color
        self._label = label
   
    def set(
        self,
        idx: int,
        x: int,
        y: int,
        parent: Optional[Union["QCCDNode", "Crossing"]] = None,
    ) -> None:
        self._idx = idx
        self._positionX: int = x
        self._positionY: int = y
        self._parent = parent

    @property
    def parent(self) -> Optional[Union["QCCDNode", "Crossing"]]:
        return self._parent

    @property
    def pos(self) -> Tuple[float, float]:
        return (self._positionX, self._positionY)

    @property
    def color(self) -> str:
        return self._color

    @property
    def label(self) -> str:
        return self._label + str(int(self._idx))

    @property
    def idx(self) -> int:
        return self._idx
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.CRYSTAL_ROTATION, Operations.GATE_SWAP, Operations.JUNCTION_CROSSING, Operations.MERGE, Operations.SPLIT, Operations.MOVE, Operations.RECOOLING]

class CoolingIon(Ion):
    ...

class QubitIon(Ion):
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return super().allowedOperations+[Operations.MEASUREMENT, Operations.ONE_QUBIT_GATE, Operations.TWO_QUBIT_MS_GATE, Operations.QUBIT_RESET]



class QCCDNode(QCCDComponent):
    # TODO only allow defined set of operations
    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        color: str,
        capacity: int,
        label: str,
        ions: Sequence[Ion] = [],
    ) -> None:
        self._idx: int = idx
        self._positionX: int = x
        self._positionY: int = y
        self._color = color
        self._ions: List[Ion] = list(ions)
        self._capacity: int = capacity
        self._label: str = label
        self.numIons: int = len(ions)

    @property
    def label(self) -> int:
        return self._label + str(self._idx)

    @property
    def ions(self) -> Sequence[Ion]:
        return self._ions

    @property
    def color(self) -> str:
        return self._color

    @property
    def pos(self) -> Tuple[float, float]:
        return (self._positionX, self._positionY)

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def nodes(self) -> Sequence[int]:
        n = [self._idx]
        if self.ions:
            n += [i.idx for i in self.ions]
        return n

    @property
    def positions(self) -> Sequence[Tuple[int, int]]:
        n = [self.pos]
        if self.ions:
            n += [i.pos for i in self.ions]
        return n
    
    def subgraph(self, graph: nx.Graph) -> nx.Graph:
        for n, p in zip(self.nodes, self.positions):
            graph.add_node(n, pos=p)
        return graph

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        if len(self.ions) == self._capacity:
            raise ValueError(
                f"addIon: QCCDNode {self.idx} is at capacity {self._capacity}"
            )
        self._ions.insert(
            (self._ions.index(adjacentIon) + offset if adjacentIon else offset), ion
        )
        ion.set(ion.idx, *self.pos, parent=self)
        self.numIons = len(self._ions)

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        if ion is None:
            if len(self.ions) == 0:
                raise ValueError(
                    f"removeIon: QCCDNode {self.idx} does not have any ions"
                )
            ion = self.ions[0]
        self._ions.remove(ion)
        self.numIons = len(self._ions)
        return ion


class Junction(QCCDNode):
    DEFAULT_COLOR = "orange"
    DEFAULT_LABEL = "J"
    DEFAULT_CAPACITY = 1

    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        color: str = DEFAULT_COLOR,
        label: str = DEFAULT_LABEL,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        super().__init__(idx, x, y, color=color, capacity=capacity, label=label)

    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.JUNCTION_CROSSING, Operations.SPLIT, Operations.MOVE, Operations.MERGE]


class Trap(QCCDNode):
    BACKGROUND_HEATING_RATE = 1  # Arbitrary heating rate in quanta per second

    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str,
        isHorizontal: bool,
        spacing: int,
        capacity: int,
        label: str,
    ) -> None:
        super().__init__(
            idx, x, y, color=color, capacity=capacity, ions=ions, label=label
        )
        self._spacing = spacing
        self._isHorizontal = isHorizontal
        self._coolingIons: Sequence[CoolingIon] = []
        for i, ion in enumerate(self._ions):
            ion.set(self._idx + i + 1, 0, 0, parent=self)
        self._arrangeIons()

    @property
    def hasCoolingIon(self) -> bool:
        return any(isinstance(ion, CoolingIon) for ion in self.ions)

    def _arrangeIons(self) -> None:
        for i, ion in enumerate(self._ions):
            o = i - len(self._ions) / 2
            ion.set(
                ion.idx,
                self.pos[0] + o * self._spacing * self._isHorizontal,
                self.pos[1] + o * self._spacing * (1 - self._isHorizontal),
                parent=self,
            )

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        super().addIon(ion, adjacentIon, offset)
        self._arrangeIons()

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        ion = super().removeIon(ion)
        self._arrangeIons()
        return ion
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.CRYSTAL_ROTATION, Operations.SPLIT, Operations.MOVE, Operations.MERGE, Operations.QUBIT_RESET, Operations.RECOOLING]


class ManipulationTrap(Trap):
    DEFAULT_COLOR = "lightyellow"
    DEFAULT_ORIENTATION  = False
    DEFAULT_SPACING = 10
    DEFAULT_CAPACITY = 3
    DEFAULT_LABEL = "MT"

    def __init__(self, idx: int, x: int, y: int, ions: Sequence[Ion], color: str = DEFAULT_COLOR, isHorizontal: bool = DEFAULT_ORIENTATION, spacing=DEFAULT_SPACING, capacity: int = DEFAULT_CAPACITY, label: str = DEFAULT_LABEL) -> None:
        super().__init__(idx, x, y, ions, color, isHorizontal, spacing, capacity, label)

    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return super().allowedOperations+[Operations.ONE_QUBIT_GATE, Operations.MEASUREMENT, Operations.TWO_QUBIT_MS_GATE, Operations.GATE_SWAP]

class StorageTrap(Trap):
    DEFAULT_COLOR = "grey"
    DEFAULT_ORIENTATION = False  # 0 for vertical, 1 for horizontal
    DEFAULT_SPACING = 10
    DEFAULT_CAPACITY = 5
    DEFAULT_LABEL = "ST"

    def __init__(self, idx: int, x: int, y: int, ions: Sequence[Ion], color: str = DEFAULT_COLOR, isHorizontal: bool = DEFAULT_ORIENTATION, spacing=DEFAULT_SPACING, capacity: int = DEFAULT_CAPACITY, label: str = DEFAULT_LABEL) -> None:
        super().__init__(idx, x, y, ions, color, isHorizontal, spacing, capacity, label)


class Crossing:
    DEFAULT_LABEL = "C"
    MOVE_AMOUNT = 8

    def __init__(
        self, idx: int, source: QCCDNode, target: QCCDNode, label: str = DEFAULT_LABEL
    ) -> None:
        self._idx: int = idx
        self._source: QCCDNode = source
        self._target: QCCDNode = target
        self._ion: Optional[Ion] = None
        self._ionAtSource: bool = False
        self._label = label

    def ionAt(self) -> QCCDNode:
        if not self._ion:
            raise ValueError(f"ionAt: no ion for crossing {self.idx}")
        return self._source if self._ionAtSource else self._target

    @property
    def pos(self) -> Tuple[int, int]:
        i1, i2 = self._getEdgeIdxs()
        x1, y1 = self._source.positions[i1]
        x2, y2 = self._target.positions[i2]
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def label(self) -> int:
        return (
            self._label
            + str(self._idx)
            + f" {self._source.label} to {self._target.label}"
        )

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def connection(self) -> Tuple[QCCDNode, QCCDNode]:
        return self._source, self._target

    def _getEdgeIdxs(self) -> Tuple[int, int]:
        permutations = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        if not self._source.ions:
            permutations[0][0] = 0
            permutations[1][0] = 0
        if not self._target.ions:
            permutations[0][1] = 0
            permutations[2][1] = 0
        pairs_distances = [
            (self._source.positions[i][0] - self._target.positions[j][0]) ** 2
            + (self._source.positions[i][1] - self._target.positions[j][1]) ** 2
            for (i, j) in permutations
        ]
        return permutations[np.argmin(pairs_distances)]

    def graphEdge(self) -> Tuple[int, int]:
        idx1, idx2 = self._getEdgeIdxs()
        return (self._source.nodes[idx1], self._target.nodes[idx2])

    def hasTrap(self, trap: Trap) -> None:
        if trap == self._source:
            return True
        elif trap == self._target:
            return True
        else:
            return False

    def hasJunction(self, junction: Junction) -> None:
        if junction == self._source:
            return True
        elif junction == self._target:
            return True
        else:
            return False

    def getEdgeIon(self, node: QCCDNode) -> Ion:
        if not node.ions:
            raise ValueError("getEdgeIon: no edge ions")
        ionIdx = node.nodes[self._getEdgeIdxs()[1 - (node == self._source)]]
        return [i for i in node.ions if i.idx == ionIdx][0]

    def setIon(self, ion: Ion, node: QCCDNode) -> None:
        if self._ion is not None:
            raise ValueError(f"setIon: crossing has not been cleared")
        self._ion = ion
        self._ionAtSource = self._source == node
        edgeIdxs = self._getEdgeIdxs()
        w = (1 + (self.MOVE_AMOUNT - 2) * (self._source == node)) / self.MOVE_AMOUNT
        x = self._source.positions[edgeIdxs[0]][0] * w + self._target.positions[
            edgeIdxs[1]
        ][0] * (1 - w)
        y = self._source.positions[edgeIdxs[0]][1] * w + self._target.positions[
            edgeIdxs[1]
        ][1] * (1 - w)
        self._ion.set(self._ion.idx, x, y, parent=self)

    def moveIon(self) -> None:
        if self.ion is None:
            raise ValueError(f"moveIon: no ion to move in crossing")
        node = self._target if self._ionAtSource else self._source
        ion = self.ion
        self.clearIon()
        self.setIon(ion, node)

    def clearIon(self) -> None:
        self._ion = None

    @property
    def ion(self) -> Optional[Ion]:
        return self._ion
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.SPLIT, Operations.MOVE, Operations.MERGE, Operations.JUNCTION_CROSSING]


class Operation:
    KEY: Operations

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        self._run = run
        self._kwargs = dict(kwargs)
        self._involvedIonsForLabel: List[Ion] = []
        self._involvedComponents: List[QCCDComponent] = involvedComponents
        self._addOns = ""
        

    def addComponent(self, component: QCCDComponent) -> None:
        self._involvedComponents.append(component)

    @property
    def involvedComponents(self) -> Sequence[QCCDComponent]:
        return self._involvedComponents

    @property
    def color(self) -> str:
        return "lightgreen"

    @property
    def involvedIonsForLabel(self) -> Sequence[Ion]:
        return self._involvedIonsForLabel

    @property
    def label(self) -> str:
        return self.KEY.name + self._addOns

    @property
    @abc.abstractmethod
    def isApplicable(self) -> bool:
        return all(self.KEY in component.allowedOperations for component in self.involvedComponents)
    
    @abc.abstractmethod
    def _checkApplicability(self) -> None:
        for component in self.involvedComponents:
            if self.KEY not in component.allowedOperations:
                raise ValueError(f"Component {component} with index {component.idx} cannot complete {self.KEY.name}")

    @classmethod
    @abc.abstractmethod
    def physicalOperation(cls) -> "Operation": ...

    @abc.abstractmethod
    def _generateLabelAddOns(self) -> None: ...

    def run(self) -> None:
        self._checkApplicability()
        self._run(())
        self._generateLabelAddOns()


class CrystalOperation(Operation):
    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, **kwargs)
        self._trap: Trap = kwargs["trap"]

    @property
    def ionsInfluenced(self) -> Sequence[Ion]:
        return self._trap.ions
    
    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = list(self._trap.ions)
        self._addOns = ""
        for ion in self._involvedIonsForLabel:
            self._addOns += f" {ion.label}"



class Split(CrystalOperation):
    KEY = Operations.SPLIT

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

   
    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is not None:
            return False
        if len(self._trap.ions) == 0:
            return False
        if self._crossing.getEdgeIon(self._trap) != self._ion:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Split: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is not None:
            raise ValueError(
                f"Split: crossing is already occupied by ion {self._crossing.ion.idx}"
            )
        if len(self._trap.ions) == 0:
            raise ValueError(f"Split: trap {self._trap.idx} has no ions")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.getEdgeIon(trap)
            trap.removeIon(ion)
            crossing.setIon(ion, trap)
         
        return cls(
            run=lambda _: run(),
            ion=ion,
            trap=trap,
            crossing=crossing,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class Merge(CrystalOperation):
    KEY = Operations.MERGE

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.getEdgeIon(self._trap)]
        self._addOns = f" {self._crossing.getEdgeIon(self._trap).label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is None:
            return False
        if self._crossing.ion != self._ion:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Merge: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is None:
            raise ValueError(f"Merge: crossing is empty")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.ion
            crossing.clearIon()
            edge_ion = crossing.getEdgeIon(trap) if trap.ions else None
            idx = trap.ions.index(edge_ion) if trap.ions else 0
            if len(trap.ions)==1:
                offset = 1 if ion.pos[0]-edge_ion.pos[0]+ion.pos[1]-edge_ion.pos[1]>0 else 0
                adjacentIon=None
            else:
                offset=idx>0
                adjacentIon=edge_ion
            trap.addIon(ion, adjacentIon=adjacentIon, offset=offset)
        
        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            trap=trap,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class CrystalRotation(CrystalOperation):
    KEY = Operations.CRYSTAL_ROTATION

    def __init__(
        self,
        run: Callable[[Any], bool],
        trap: Trap,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._trap: Trap = trap

    @property
    def isApplicable(self) -> bool:
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap):
        def run():
            ions = list(trap.ions).copy()[::-1]
            for ion in ions:
                trap.removeIon(ion)
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
        
        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )



class CoolingOperation(CrystalOperation):
    KEY = Operations.RECOOLING
    
    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
    
    @property
    def isApplicable(self) -> bool:
        if not self._trap.hasCoolingIon:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._trap.hasCoolingIon:
            raise ValueError(f"CoolingOperation: trap {self._trap.idx} does not include a cooling ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap
    ):
        def run():
            ...

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )




class Move(Operation):
    KEY = Operations.MOVE
   
    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        return bool(self._crossing.ion) and self._ion == self._crossing.ion and super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.ion:
            raise ValueError(f"Move: crossing does not contain ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, crossing: Crossing, ion: Optional[Ion] = None):
        def run():
            crossing.moveIon()

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            involvedComponents=[crossing],
        )

class JunctionCrossing(Operation):
    KEY = Operations.JUNCTION_CROSSING

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == 1:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) == 1:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
         
        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )


class QubitOperation(Operation):
    def __init__(
        self,
        run: Callable[[Trap], None],
        ions: Sequence[Ion],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, ions=ions, **kwargs)
        self._ions = ions
        self._trap: Optional[Trap] = None

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = self._ions
        self._addOns = ""
        for ion in self._ions:
            self._addOns += f" {ion.label}"

    def getTrapForIons(self) -> Optional[Trap]:
        """
        return trap if all ions are in the same trap
        """
        if not self.ions:
            return None
        trap = self.ions[0].parent
        if trap is None:
            return None
        if not isinstance(trap, Trap):
            return None
        if not all(i.parent == trap for i in self.ions[1:]):
            return None
        return trap

    def setTrap(self, trap: Trap) -> None:
        self._kwargs["trap"] = trap
        self._involvedComponents.append(trap)
        self._trap = trap

    @property
    def isApplicable(self) -> bool:
        if not self._trap:
            return False
        for ion in self.ions:
            if ion.parent != self._trap:
                return False 
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._trap:
            raise ValueError("QubitOperation: trap not set")
        for ion in self.ions:
            if ion.parent != self._trap:
                raise ValueError(f"QubitOperation: ion {ion.idx} not in trap {self._trap.idx}") 
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap, **kwargs) -> "QubitOperation":
        qubitOperation = cls.qubitOperation(**kwargs)
        qubitOperation.setTrap(trap)
        return qubitOperation

    @classmethod
    @abc.abstractmethod
    def qubitOperation(cls) -> "QubitOperation": ...

    @property
    def ions(self) -> Sequence[Ion]:
        return self._ions

    def run(self) -> None:
        self._checkApplicability()
        self._run(self._trap)
        self._generateLabelAddOns()


class OneQubitGate(QubitOperation):
    KEY = Operations.ONE_QUBIT_GATE
   
    def __init__(
        self,
        run: Callable[[Trap], None],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "OneQubitGate":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])

class XRotation(OneQubitGate):
    @property
    def label(self) -> str:
        return "RX" + self._addOns

class YRotation(OneQubitGate):
    @property
    def label(self) -> str:
        return "RY" + self._addOns

class Measurement(QubitOperation):
    KEY = Operations.MEASUREMENT

    def __init__(
        self,
        run: Callable[[Trap], bool],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "Measurement":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])



class QubitReset(QubitOperation):
    KEY = Operations.QUBIT_RESET
 
    def __init__(
        self,
        run: Callable[[Trap], bool],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "QubitReset":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])

class TwoQubitMSGate(QubitOperation):
    KEY = Operations.TWO_QUBIT_MS_GATE
    A = 0.001  # Scaling factor for fidelity calculation

    def __init__(
        self,
        run: Callable[[Trap], bool],
        involvedComponents: Sequence["QCCDComponent"],
        ion1: Ion,
        ion2: Ion,
        gate_type: str = "AM2",
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, **kwargs, ions=[ion1, ion2]
        )
        self.gateType = gate_type
        self._ion1 = ion1
        self._ion2 = ion2

    @property
    def ionsActedIdxs(self) -> Tuple[int, int]:
        return self._ion1.idx, self._ion2.idx

    @property
    def ionsInJunctions(self) -> bool:
        return isinstance(self._ion1.parent, Junction) or isinstance(
            self._ion2.parent, Junction
        )

    @classmethod
    def qubitOperation(
        cls, ion1: Ion, ion2: Ion, gate_type: str = "AM2"
    ) -> "TwoQubitMSGate":
        def run(trap: Trap):
            ...

        return cls(
            run=run,
            involvedComponents=[ion1, ion2],
            gate_type=gate_type,
            ion1=ion1,
            ion2=ion2,
        )
    

class GateSwap(QubitOperation):
    KEY = Operations.GATE_SWAP

    def __init__(
        self,
        run: Callable[[Any], bool],
        operations: Sequence[TwoQubitMSGate],
        ion1: Ion,
        ion2: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, ions=[ion1,ion2], **kwargs)
        self._operations: Sequence[TwoQubitMSGate] = operations
        self._ion1 = ion1
        self._ion2 = ion2

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion1, self._ion2]
        self._addOns = f" {self._ion1.label} {self._ion2.label}"

    def setTrap(self, trap: Trap) -> None:
        for op in self._operations:
            op.setTrap(trap)
        return super().setTrap(trap)

    @classmethod
    def qubitOperation(cls, ion1: Ion, ion2: Ion):
        operations: List[TwoQubitMSGate] = []
        for _ in range(3): # REF: fig. 5 https://arxiv.org/pdf/2004.04706
            operations.append(TwoQubitMSGate.qubitOperation(ion1=ion1, ion2=ion2))

        def run(trap: Trap):
            if ion1 == ion2:
                return
            for op in operations:
                op.run()
            idx1 = trap.ions.index(ion1)
            trap.removeIon(ion1)
            trap.addIon(ion1, adjacentIon=ion2)
            trap.removeIon(ion2)
            trap.addIon(ion2, offset=idx1)

        return cls(
            run=run,
            operations=operations,
            ion1=ion1,
            ion2=ion2,
            involvedComponents=[ion1, ion2],
        )


class ParallelOperation(Operation):
    KEY = Operations.PARALLEL

    def __init__(
        self, run: Callable[[Any], bool], operations: Sequence[Operation], **kwargs
    ) -> None:
        super().__init__(run, **kwargs, operations=operations)
        self._operations = operations

    def _generateLabelAddOns(self) -> None:
        self._addOns = ""
        for op in self._operations:
            self._addOns += f" {op.KEY.name}"

    @property
    def isApplicable(self) -> bool:
        return all(op.isApplicable for op in self.operations)
    
    def _checkApplicability(self) -> None:
        return True

    @property
    def operations(self) -> Sequence[Operation]:
        return self._operations

    @classmethod
    def physicalOperation(cls, operations: Sequence[Operation]):
        def run():
            for op in np.random.permutation(operations):
                op.run()

        involvedComponents = []
        for op in operations:
            involvedComponents += list(op.involvedComponents)
        return cls(
            run=lambda _: run(),
            operations=operations,
            involvedComponents=set(involvedComponents),
        )


class QCCDArch:
    SIZING = 1
    JUNCTION_SIZE = 800 * SIZING
    ION_SIZE = 800 * SIZING
    FONT_SIZE = 14 * SIZING
    WINDOW_SIZE = 30 * SIZING, 24 * SIZING
    TRAP_WIDTH = 15 * SIZING
    EDGE_WIDTH = SIZING

    HIGHLIGHT_COLOR = "yellow"
    HIGHLIGHT_NODE_SIZE = 4000 * SIZING
    JUNCTION_SHAPE = "s"
    ION_SHAPE = "o"
    DEFAULT_ALPHA = 0.5
    PADDING = 0.6

    N_ITERS = 5000

    def __init__(self):
        self._trapEdges: Mapping[int, Sequence[Tuple[int, int]]] = {}
        self._crossingEdges: Mapping[Tuple[int, int], Crossing] = {}
        self._crossings: List[Crossing] = []
        self._manipulationTraps: List[ManipulationTrap] = []
        self._junctions: List[Junction] = []
        self._nextIdx = 0
        self._routingTable: Mapping[int, Mapping[int, Sequence[Operation]]] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._inActiveEdges: List[int] = []
        self._centralities: Mapping

    @property
    def routingTable(self):
        return self._routingTable

    @property
    def ions(self) -> Mapping[int, Ion]:
        ions = {}
        for t in self._manipulationTraps:
            ions.update([(ion.idx, ion) for ion in t.ions])
        for j in self._junctions:
            ions.update([(ion.idx, ion) for ion in j.ions])
        for c in self._crossings:
            if c.ion:
                ions[c.ion.idx] = c.ion
        return ions

    @property
    def nodes(self) -> Mapping[int, QCCDNode]:
        cs = {}
        for t in self._manipulationTraps:
            cs[t.idx] = t
        for j in self._junctions:
            cs[j.idx] = j
        return cs

    def addEdge(self, source: QCCDNode, target: QCCDNode) -> Crossing:
        crossing = Crossing(self._nextIdx, source, target)
        self._crossings.append(crossing)
        self._nextIdx += 1
        return crossing

    def addManipulationTrap(
        self,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str = ManipulationTrap.DEFAULT_COLOR,
        isHorizontal: bool = ManipulationTrap.DEFAULT_ORIENTATION,
        spacing: int = ManipulationTrap.DEFAULT_SPACING,
        capacity: int = ManipulationTrap.DEFAULT_CAPACITY,
    ) -> Trap:
        trap = ManipulationTrap(
            self._nextIdx,
            x,
            y,
            ions,
            color=color,
            isHorizontal=isHorizontal,
            spacing=spacing * self.SIZING,
            capacity=capacity,
        )
        self._manipulationTraps.append(trap)
        self._nextIdx += len(ions) + 1
        return trap

    def addJunction(
        self,
        x: int,
        y: int,
        color: str = Junction.DEFAULT_COLOR,
        label: str = Junction.DEFAULT_LABEL,
        capacity: int = Junction.DEFAULT_CAPACITY,
    ) -> Junction:
        junction = Junction(
            self._nextIdx, x, y, color=color, label=label, capacity=capacity
        )
        self._junctions.append(junction)
        self._nextIdx += 1
        return junction

    def decideDestinationTrap(self, ion1: Ion, ion2: Ion) -> Trap:
        destIon = max(
            (ion1, ion2), key=lambda ion: ion.parent.capacity - ion.parent.numIons
        )
        freeTraps = sorted(
            [t for t in self._manipulationTraps if t.capacity > t.numIons + 2],
            key=lambda t: (t.pos[0] - ion2.parent.pos[0]) ** 2
            + (t.pos[1] - ion2.parent.pos[1]) ** 2
            + (t.pos[0] - ion1.parent.pos[0]) ** 2
            + (t.pos[1] - ion1.parent.pos[1]) ** 2,
        )
        if destIon.parent.capacity > destIon.parent.numIons and isinstance(
            destIon.parent, ManipulationTrap
        ):
            freeTraps.insert(0, destIon.parent)
        trap = freeTraps[0]
        return trap

    def route(self, ion: Ion, trap: ManipulationTrap) -> Sequence[Operation]:
        src = ion.idx
        dest = trap.idx
        if dest in self._routingTable[src]:
            return self._routingTable[src][dest]
        if not nx.has_path(self._graph, src, dest):
            self._routingTable[src][dest] = [False]
            return [False]
        paths = nx.all_shortest_paths(self._graph, src, dest)
        best_path = sorted(
            paths,
            key=lambda path: sum(
                self._centralities[n1, n2] for n1, n2 in zip(path[:-1], path[1:])
            ),
        )[0]
        ops = []
        for n1, n2 in zip(best_path[:-1], best_path[1:]):
            op_ = self._graph.edges[n1, n2]["operations"]
            op_ = [
                (
                    GateSwap.physicalOperation(
                        trap=o._trap, ion1=self.ions[src],ion2= o._ion2
                    )
                    if isinstance(o, GateSwap)
                    else o
                )
                for o in op_
            ]
            ops.extend(op_)
        self._routingTable[src][dest] = ops
        return ops

    def refreshGraph(self) -> None:
        g = nx.DiGraph()

        for j in self._junctions:
            j.subgraph(g)
            j.numIons = len(j.ions)

        for t in self._manipulationTraps:
            t.subgraph(g)
            t.numIons = len(t.ions)

        for trap in self._manipulationTraps:
            trapEdges = []
            for ion1 in trap.ions:
                g.add_edge(ion1.idx, trap.idx, operations=[])
                for ion2 in trap.ions:
                    if ion1 == ion2:
                        continue
                    trapEdges.append((ion1.idx, ion2.idx))
                    trapEdges.append((ion2.idx, ion1.idx))
                    g.add_edge(
                        ion1.idx,
                        ion2.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion1, ion2=ion2)],
                    )
                    g.add_edge(
                        ion2.idx,
                        ion1.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion2, ion2=ion1)],
                    )
            self._trapEdges[trap.idx] = trapEdges

        crossingEdges = {}
        for crossing in self._crossings:
            n1, n2 = crossing.connection
            n1Idx = crossing.getEdgeIon(n1).idx if n1.ions else n1.idx
            n2Idx = crossing.getEdgeIon(n2).idx if n2.ions else n2.idx
            crossingEdges[(n1Idx, n2Idx)] = crossing
            crossingEdges[(n2Idx, n1Idx)] = crossing
            ion1 = crossing.getEdgeIon(n1) if n1.ions else None
            ion2 = crossing.getEdgeIon(n2) if n2.ions else None
            doRotation1 = [GateSwap.physicalOperation(trap=n1,ion1=ion1,ion2=ion1)] if len(n1.ions)==1 else []
            doRotation2 = [GateSwap.physicalOperation(trap=n2,ion1=ion2,ion2=ion2)] if len(n2.ions)==1 else []
            if isinstance(n1, Trap) and isinstance(n2, Junction):
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Trap):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Junction):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            else:
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            g.add_edge(n1Idx, n2Idx, operations=ops1)
            g.add_edge(n2Idx, n1Idx, operations=ops2)
            if crossing.ion:
                g.add_node(crossing.ion.idx, pos=crossing.ion.pos)
        self._crossingEdges = crossingEdges

        for n2Idx in self._inActiveEdges:
            graphEdges = [
                (u, v)
                for (u, v), crossing in self._crossingEdges.items()
                if self.nodes[n2Idx] in crossing.connection
                and (v in self.nodes[n2Idx].nodes)
            ]
            g.remove_edges_from(graphEdges)
        self._graph = g
        self._centralities = nx.edge_betweenness_centrality(self._graph)
        self._routingTable = {ion.idx: {} for ion in self.ions.values()}

    def display(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        title: str = "",
        operation: Optional[Operation] = None,
        show_junction: bool = True,
        showEdges: bool = True,
        showIons: bool = True,
        showLabels: bool = True,
        runOps: bool = False,
    ) -> None:
        pos = {}
        labels = {}
        operationNodes: List[List[int]] = []
        involvedIons: List[Sequence[Ion]] = []

        if operation is None:
            operations = []
        elif isinstance(operation, ParallelOperation):
            operations = operation.operations
        else:
            operations = [operation]

        if runOps:
            for op in operations:
                op.run()

            self.refreshGraph()

        for op in operations:
            operationNodes.append([])
            involvedIons.append(op.involvedIonsForLabel)

        for junction in self._junctions:
            pos[junction.nodes[0]] = junction.pos
            labels[junction.nodes[0]] = ""
            if show_junction:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=[junction.nodes[0]],
                    node_color=[junction.color],
                    node_shape=self.JUNCTION_SHAPE,
                    node_size=self.JUNCTION_SIZE,
                )
            for n, ion in zip(junction.nodes[1:], junction.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[n],
                        node_color=[ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showLabels:
                x = junction.pos[0]
                y = junction.pos[1]
                ax.text(
                    x,
                    y,
                    junction.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        for c in self._crossings:
            if c.ion:
                pos[c.ion.idx] = c.ion.pos
                labels[c.ion.idx] = c.ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[c.ion.idx],
                        node_color=[c.ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if c.ion in ions:
                        nodes.append(c.ion.idx)

        for t in self._manipulationTraps:
            if not isinstance(t, Trap):
                continue
            pos[t.nodes[0]] = t.pos
            labels[t.nodes[0]] = ""
            colors = {}
            for n, ion in zip(t.nodes[1:], t.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                colors[n] = ion.color
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showIons:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=t.nodes[1:],
                    node_color=colors.values(),
                    node_shape=self.ION_SHAPE,
                    node_size=self.ION_SIZE,
                )

        for trap in self._manipulationTraps:
            if not isinstance(trap, Trap):
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=trap[0],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color='red',
                    width=trap[1],
                )
                continue
            if showIons:
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=self._trapEdges[trap.idx],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color=trap.color,
                    width=self.TRAP_WIDTH,
                )
            if showLabels:
                x = trap.pos[0]
                y = trap.pos[1]
                ax.text(
                    x,
                    y,
                    trap.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showEdges:
            nx.draw_networkx_edges(
                self._graph,
                pos,
                edgelist=self._crossingEdges.keys(),
                ax=ax,
                alpha=self.DEFAULT_ALPHA,
                width=self.EDGE_WIDTH,
            )
        if showLabels:
            for e in self._crossings:
                ax.text(
                    *e.pos,
                    e.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showIons:
            nx.draw_networkx_labels(
                self._graph, pos, ax=ax, labels=labels, font_size=self.FONT_SIZE
            )

        for nodes, op in zip(operationNodes, operations):
            if nodes:
                xVals = [pos[node][0] for node in nodes]
                yVals = [pos[node][1] for node in nodes]
                padding = self.SIZING * self.PADDING
                xMin, xMax = min(xVals) - padding, max(xVals) + padding
                yMin, yMax = min(yVals) - padding, max(yVals) + padding
                width = xMax - xMin
                height = yMax - yMin
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ellip = Ellipse(
                    (xLabel, yLabel),
                    width,
                    height,
                    edgecolor=op.color,
                    alpha=self.DEFAULT_ALPHA,
                    facecolor=op.color,
                )
                ax.add_patch(ellip)
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ax.text(
                    xLabel,
                    yLabel,
                    op.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        ax.set_title(title)
        n = len(fig.axes)
        fig.set_size_inches(self.WINDOW_SIZE[0]*n, self.WINDOW_SIZE[1])

    def _routingForQubit(
        self, operation: TwoQubitMSGate
    ) -> Tuple[Trap, Sequence[Operations], Sequence[Ion]]:
        ionsInvolved = list(operation.ions)
        ion1, ion2 = operation.ions
        trap = self.decideDestinationTrap(ion1, ion2)
        ion2.parent.numIons -= 1
        ion1.parent.numIons -= 1
        trap.numIons += 2
        movements = list(self.route(ion1, trap)) + list(
            self.route(ion2, trap)
        )
        for m in movements:
            if isinstance(m, CrystalOperation):
                ionsInvolved += list(m.ionsInfluenced)
            m.addComponent(ion1)
            m.addComponent(ion2)
        return trap, movements, ionsInvolved

    def processOperations(
        self, operations: Sequence[QubitOperation]
    ) -> Sequence[Operation]:
        instructionsLeft = list(operations).copy()
        allOps = []
        while instructionsLeft:
            self.refreshGraph()
            physicalInstructions = []
            ionsInvolved = set()
            toRemove = []
            for op in instructionsLeft:
                trap = op.getTrapForIons()
                ionsInOp = list(op.ions)
                movements = []
                if not trap:
                    trap, movements, ionsInOp = self._routingForQubit(op)
                if ionsInvolved.isdisjoint(ionsInOp):
                    physicalInstructions.extend(movements)
                    op.setTrap(trap)
                    physicalInstructions.append(op)
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(ionsInOp)
            for op in toRemove:
                instructionsLeft.remove(op)
            allOps.extend(physicalInstructions)
            for op in physicalInstructions:
                op.run()
        return allOps


def paralleliseOperations(
    operationSequence: Sequence[Operation],
) -> Sequence[ParallelOperation]:
    operationSequence = list(operationSequence)
    parallelOperationsSequence: List[ParallelOperation] = []
    if not operationSequence:
        return parallelOperationsSequence
    while operationSequence:
        parallelOperations = [operationSequence.pop(0)]
        involvedComponents: Set[QCCDComponent] = set(
            parallelOperations[0].involvedComponents
        )
        for op in operationSequence:
            components = op.involvedComponents
            if involvedComponents.isdisjoint(components):
                parallelOperations.append(op)
            involvedComponents = involvedComponents.union(components)
        for op in parallelOperations[1:]:
            operationSequence.remove(op)
        parallelOperation = ParallelOperation.physicalOperation(parallelOperations)
        parallelOperationsSequence.append(parallelOperation)
    return parallelOperationsSequence


class QCCDCircuit(stim.Circuit):
    DATA_QUBIT_COLOR = "lightblue"
    MEASUREMENT_QUBIT_COLOR = "red"
    PLACEMENT_ION = ("grey", "P")
    TRAP_COLOR = "grey"
    JUNCTION_COLOR = "orange"
    SPACING = 20
    CAPACITY_SCALING = 5


    start_score: int = 1
    score_delta: int = 2
    joinDisjointClusters: bool = False
    minIters: int = 1_000
    maxIters: int = 10_000


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ionMapping: Dict[int, Tuple[Ion, Tuple[int, int]]] = {}
        self._measurementIons: List[Ion] = []
        self._dataIons: List[Ion] = []
        self._originalArrangement: Dict[Trap, Sequence[Ion]] = {}
        self._arch: QCCDArch

    @classmethod
    def generated(cls, *args, **kwargs) -> "QCCDCircuit":
        return QCCDCircuit(stim.Circuit.generated(*args, **kwargs).__str__())

    def circuitString(self) -> Sequence[str]:
        instructions = (
            self.flattened().decomposed().without_noise().__str__().splitlines()
        )
        newInstructions = []
        for i in instructions:
            qubits = i.rsplit(" ")[1:]
            if i.startswith("DETECTOR"):
                continue
            elif i.startswith("TICK"):
                continue
            elif i.startswith("OBSERVABLE"):
                continue
            elif i[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{i[0]} {qubit}")
            elif any(i.startswith(s) for s in stim.gate_data("cnot").aliases):
                for i in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*i]} {qubits[2*i+1]}")
            else:
                newInstructions.append(i)
        return newInstructions

    @property
    def ionMapping(self) -> Mapping[int, Tuple[Ion, Tuple[int, int]]]:
        return self._ionMapping

    def _parseCircuitString(self) -> Sequence[QubitOperation]:
        instructions = self.circuitString()

        self._measurementIons = []
        self._ionMapping = {}
        self._dataIons = []
        for j, i in enumerate(instructions):
            if not i.startswith("QUBIT_COORDS"):
                break
            coords = tuple(
                map(int, i.removeprefix("QUBIT_COORDS(").split(")")[0].split(","))
            )
            idx = int(i.split(" ")[-1])
            ion = QubitIon(self.MEASUREMENT_QUBIT_COLOR, label="M")
            ion.set(ion.idx, *coords)
            self._ionMapping[idx] = ion, coords
            self._measurementIons.append(ion)

        instructions = instructions[j:]
        operations = []
        dataQubits = []
        # TODO establish correct mapping of qubit operations from QIP toolkit with references
        for j, i in enumerate(instructions):
            if not ( i[0] in ("M", "H", "R") or i.startswith("CNOT")):
                continue
            idx = int(i.split(" ")[1])
            ion = self._ionMapping[idx][0]
            if i[0] == "M":
                operations.append(Measurement.qubitOperation(ion))
                dataQubits.append(ion) # data qubits are the ones measured at the end
            elif i[0] == "H":
                # page 80 https://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion)
                ])
                dataQubits.clear()
            elif i[0] == "R":
                operations.append(QubitReset.qubitOperation(ion))
                dataQubits.clear()
            elif i.startswith("CNOT"):
                idx2 = int(i.split(" ")[2])
                ion2 = self._ionMapping[idx2][0]
                # Fig 4. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion2),
                    TwoQubitMSGate.qubitOperation(
                        ion, ion2
                    ),
                    YRotation.qubitOperation(ion)
                ])
                dataQubits.clear()
        for d in dataQubits:
            d._color = self.DATA_QUBIT_COLOR
            d._label = "D"
            self._dataIons.append(d)
            self._measurementIons.remove(d)
        return operations

    
    def _clusterIons(
        self, ions: Sequence[Ion], coords: npt.NDArray[np.float_], trapCapacity: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:
        unclusteredIndices = set(range(len(coords)))
        clusters = []
        while unclusteredIndices:
            currentClusterIndices = []
            currentClusterCoords = []

            # Choose the first unclustered ion as the starting point
            start_index = next(iter(unclusteredIndices))
            currentClusterIndices.append(start_index)
            currentClusterCoords.append(coords[start_index])

            unclusteredIndices.remove(start_index)

            # Keep adding ions to the cluster until it reaches the trapCapacity
            while len(currentClusterIndices) < trapCapacity and unclusteredIndices:
                # Find the ion closest to the current cluster's centroid
                cluster_centroid = np.mean(currentClusterCoords, axis=0)
                distances = distance.cdist(
                    [cluster_centroid], coords[list(unclusteredIndices)]
                )[0]
                nearest_index = list(unclusteredIndices)[np.argmin(distances)]
                # Add the nearest ion to the current cluster
                currentClusterIndices.append(nearest_index)
                currentClusterCoords.append(coords[nearest_index])
                unclusteredIndices.remove(nearest_index)

            clusterIons = [ions[i] for i in currentClusterIndices]
            clusterCentre = np.mean(currentClusterCoords, axis=0)
            clusters.append((clusterIons, clusterCentre))
        return clusters

    def _objectiveFunctionForArrangement(
        self,
        gridPos: npt.NDArray[np.int_],
        edges: Sequence[Tuple[int, int]],
        clusters: Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]],
    ):
        distances = []
        differences = []
        for u, v in edges:
            ionsInvolved = len(clusters[u][0]) * len(clusters[v][0])
            vectorBefore = clusters[u][1] - clusters[v][1]
            vectorBeforeSize = np.linalg.norm(vectorBefore)
            if vectorBeforeSize!=0:
                vectorBefore = vectorBefore / vectorBeforeSize
            vectorAfter = gridPos[u] - gridPos[v]
            vectorAfterSize = np.linalg.norm(vectorAfter)
            if vectorAfterSize!=0:
                vectorAfter = vectorAfter / vectorAfterSize
            dotProd = np.dot(vectorAfter, vectorBefore)
            if isinstance(dotProd, float) and -1 <= dotProd <= 1:
                cosDiff = np.arccos(dotProd)
            else:
                cosDiff = np.inf
            dist = np.linalg.norm(gridPos[u] - gridPos[v])
            distances.append(dist * ionsInvolved)
            differences.append(cosDiff * ionsInvolved)
        return (
            sum(distances) 
            + sum(differences) 
            + any(d == 0 for d in distances) * np.inf
        )

    def _arrangeClusters(
        self,
        clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
        rows: int,
        cols: int,
        n_iters: int = 5000,
    ):
        numClusters = len(clusters)
        edges = np.array(
            [[(i, j) for i in range(numClusters) if i != j] for j in range(numClusters)]
        ).reshape(((numClusters - 1) * numClusters, 2))

        # Only consider grid positions within a minimal region to avoid transport overheads
        rs = min(rows, int(1 + numClusters / cols))
        cs = cols
        allGridPos = np.array([[(c, r) for r in range(rs)] for c in range(cs)]).reshape(
            ((cs * rs, 2))
        )

        # Initial discrete grid positions
        initialGridPos = allGridPos[:numClusters].copy()
        freeGridPos = allGridPos[numClusters:].copy()

        bestPos = initialGridPos.copy()
        bestError = self._objectiveFunctionForArrangement(
            initialGridPos, edges, clusters
        )
        newPos = initialGridPos
        # Simple random walk optimisation algorithm, without hill climbing, perhaps reconsider
        for _ in range(n_iters):
            cluster = np.random.randint(numClusters)
            gridPos = np.random.randint(rs), np.random.randint(cs)
            oldGridPos = newPos[cluster]
            if gridPos in freeGridPos:
                freeGridPos = np.where(freeGridPos == gridPos, oldGridPos, freeGridPos)
            else:
                newPos = np.where(newPos == gridPos, oldGridPos, newPos)
            newPos[cluster] = gridPos
            newError = self._objectiveFunctionForArrangement(newPos, edges, clusters)
            if newError < bestError:
                bestError = newError
                bestPos = newPos
        return bestPos

    def _gridToCoordinate(
        self, pos: Tuple[int, int], trapCapacity: int
    ) -> npt.NDArray[np.float_]:
        return np.array(pos) * (trapCapacity + 1) * self.SPACING

    def resetArch(
        self
    ) -> QCCDArch:
        for node in self._arch.nodes.values():
            while node.ions:
                node.removeIon(node.ions[0])

        for trap, ions in self._originalArrangement.items():
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
        return self._arch

    def processCircuit(
        self,
        trapCapacity: int = 2,
        rows: int = 1,
        cols: int = 5,
        isHorizontal: bool =False,
    ) -> Tuple[QCCDArch, Sequence[QubitOperation]]:        
        instructions = self._parseCircuitString()
        if trapCapacity * rows * cols < len(self._ionMapping):
            raise ValueError("processCircuit: not enough traps")
        
        ions = list(self._measurementIons)+list(self._dataIons)
        ionCoords = np.array([list(ion.pos) for ion in ions])
        clusters=self._clusterIons(ions, ionCoords, trapCapacity)
        gridPositions = self._arrangeClusters(clusters, rows, cols)

        trap_for_grid = {
            (col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        self._arch = QCCDArch()
        traps_dict = {}
        for row in range(rows):
            for col in range(cols):
                if (col, row) in trap_for_grid:
                    ions = trap_for_grid[(col, row)][0]
                else:
                    ions = []
                traps_dict[(col, row)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((col, row), trapCapacity),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=isHorizontal,
                    capacity=trapCapacity * self.CAPACITY_SCALING,
                )
                self._originalArrangement[traps_dict[(col, row)]] = ions
    
        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 1, r) in traps_dict:
                    self._arch.addEdge(trap_node, traps_dict[(col + 1, r)])
        
        else:
            junctions_dict = {}
            for (col, row), trap_node in traps_dict.items():
                # Add vertical edges (between rows)
                if (col, row + 1) in traps_dict:
                    junction = self._arch.addJunction(
                        *(
                            (
                                self._gridToCoordinate((col, row), trapCapacity)
                                + self._gridToCoordinate((col, row + 1), trapCapacity)
                            )
                            / 2
                        ),
                        color=self.JUNCTION_COLOR,
                    )
                    junctions_dict[(col, row)] = junction
                    self._arch.addEdge(trap_node, junction)
                    self._arch.addEdge(junction, traps_dict[(col, row + 1)])

            # Add horizontal edges between junctions in the same row
            for row in range(rows):
                for col in range(cols - 1):
                    if (col, row) in junctions_dict and (
                        col + 1,
                        row,
                    ) in junctions_dict:
                        self._arch.addEdge(
                            junctions_dict[(col, row)], junctions_dict[(col + 1, row)]
                        )

        return self._arch, instructions
