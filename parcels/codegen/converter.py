from parcels.codegen import ir
import ast


__all__ = ['ASTConverter']


class ASTConverter(ast.NodeVisitor):
    """Converter from Python AST into Parcels IR.

    """

    def __init__(self, grid, ptype):
        self.grid = grid
        self.ptype = ptype

        self.statements = []

    def convert(self, py_ast):
        """Convert Python kernel code to Parcels IR"""
        self.visit(py_ast)

    def visit_Assign(self, node):
        # TODO: Deal with tuple assignments
        var = self.visit(node.targets[0])
        expr = self.visit(node.value)

    def visit_Name(self, node):
        if node.id == 'grid':
            return ir.GridIntrinsic(self.grid)
        elif node.id == 'particle':
            return ir.ParticleIntrinsic(self.ptype)
        else:
            return ir.Variable(node.id)

    def visit_Num(self, node):
        return ir.Constant(node.n)

    def visit_BinOp(self, node):
        op_conv = {
            ast.Add: ir.Add,
            ast.Sub: ir.Sub,
            ast.Mult: ir.Mul,
            ast.Div: ir.Div
        }
        return op_conv[type(node.op)](self.visit(node.left),
                                      self.visit(node.right))

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        return getattr(obj, node.attr)
        
    def visit_Subscript(self, node):
        field = self.visit(node.value)
        assert(isinstance(field, ir.FieldIntrinsic))
        args = self.visit(node.slice)
        return ir.FieldEvalIntrinsic(field, args)
