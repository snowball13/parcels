from parcels.grid import Grid
from parcels.field import Field
from parcels.particle import ParticleType


class Node(object):
    """Base class for intermedaite representation nodes"""

    def __init__(self):
        self.children = []

    def __repr__(self):
        return self.ccode

    def terminal(self):
        """Decide whether the node is a leaf in the IR tree."""
        return len(self.children) == 0

    @property
    def ccode(self):
        """Property that generates the corresponding C source code"""
        raise NotImplementedError('Unknown intermediate node')


class Constant(Node):
    def __init__(self, value):
        self.value = value

    @property
    def ccode(self):
        return '%s' % self.value


class Variable(Node):
    def __init__(self, name, dtype=None):
        self.name = name
        # TODO: Add runtime type-checking
        self.dtype = dtype

    def __repr__(self):
        return '<%s [%s]>' % (self.name, self.dtype)

    @property
    def ccode(self):
        self.name


class BinaryOperator(Node):

    op = None
    
    def __init__(self, expr1, expr2):
        self.children = [expr1, expr2]
    
    @property
    def ccode(self):
        return (' %s ' % type(self).op).join(['%s' % c for c in self.children])


class Add(BinaryOperator):
    op = '+'


class Sub(BinaryOperator):
    op = '-'


class Mul(BinaryOperator):
    op = '*'


class Div(BinaryOperator):
    op = '/'


class GridIntrinsic(Node):
    
    def __init__(self, grid):
        assert(isinstance(grid, Grid))
        self.grid = grid

    def __getattr__(self, attr):
        field = getattr(self.grid, attr)
        return FieldIntrinsic(field)

    @property
    def ccode(self):
        return 'grid'


class FieldIntrinsic(Node):

    def __init__(self, field):
        assert(isinstance(field, Field))
        self.field = field

    @property
    def ccode(self):
        return 'grid->%s' % self.field.name


class ParticleIntrinsic(Node):
    
    def __init__(self, ptype):
        assert(isinstance(ptype, ParticleType))
        self.ptype = ptype

    def __getattr__(self, attr):
        assert(attr in [v.name for v in self.ptype.variables])
        return Variable(name='particle->%s' % attr)


class FieldEvalIntrinsic(Node):

    def __init__(self, field, args):
        self.field = field
        self.args = args

    def __repr__(self):
        return '::%s[%s]::' % (self.field, self.args)
