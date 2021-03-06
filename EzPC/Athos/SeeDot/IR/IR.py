"""

Authors: Sridhar Gopinath, Nishant Kumar.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from enum import Enum
import numpy as np

import Util, Type
import AST.AST as AST

# TODO - check if this can be cleaned up
class Op:
    Op = Enum("Op", "+ - * / << >> & | ^ ~ ! && || < <= > >= == != max .* ./")
    Op.print = lambda self, writer: writer.printf("%s", self.name)
    Op.op_list = lambda op_str: list(
        map(lambda x: Op.Op[x], op_str.split())
    )  # op_str:str
    Op.IsPrefixOp = lambda self: True if (self.name == "max") else False
    Op.IsPostfixOp = lambda self: False


class Expr:
    pass


class IntExpr(Expr):
    pass


class BoolExpr(Expr):
    pass


class Int(IntExpr):
    @staticmethod
    def negMax():
        return DataType.getNegMax()

    @staticmethod
    def max():
        return DataType.getMax()

    def __init__(self, n: int, wordLen: int = None):
        if not (wordLen):
            wordLen = Util.Config.wordLength
        self.n = DataType.getInt(n, wordLen)

    def updateName(self, expr_mapping):
        if isinstance(self.n, np.int8):
            return self.__class__(self.n, 8)
        elif isinstance(self.n, np.int16):
            return self.__class__(self.n, 16)
        elif isinstance(self.n, np.int32):
            return self.__class__(self.n, 32)
        elif isinstance(self.n, np.int64):
            return self.__class__(self.n, 64)
        else:
            assert False
            return self.__class__(self.n)


class Var(IntExpr):
    def __init__(self, idf: str, idx: list = [], inputVar=False):
        self.idf = idf
        self.idx = idx
        self.inputVar = inputVar

    def updateName(self, expr_mapping):
        idx_new = list(map(lambda e: e.updateName(expr_mapping), self.idx))
        if self.idf not in expr_mapping:
            return self.__class__(self.idf, idx_new, self.inputVar)
        else:
            to_e = expr_mapping[self.idf]
            if isinstance(to_e, Var):
                return self.__class__(
                    to_e.idf, to_e.idx + idx_new, to_e.inputVar and self.inputVar
                )
            elif isinstance(to_e, Int):
                return to_e
            else:
                assert False


class Bool(BoolExpr):
    def __init__(self, b: bool):
        self.b = b

    def updateName(self, expr_mapping):
        return self.__class__(self.b)


class IntUop(IntExpr):
    def __init__(self, op: Op.Op, e: IntExpr):
        assert op in Op.Op.op_list("- ~")
        self.op = op
        self.e = e

    def updateName(self, expr_mapping):
        return self.__class__(self.op, self.e.updateName(expr_mapping))


class Exp(IntExpr):
    def __init__(self, e: IntExpr):
        self.e = e

    def updateName(self, expr_mapping):
        return self.__class__(self.e.updateName(expr_mapping))


class TypeCast(IntExpr):
    def __init__(self, type, expr: Expr):
        self.type = type
        self.expr = expr

    def updateName(self, expr_mapping):
        return self.__class__(self.type, self.expr.updateName(expr_mapping))


class IntBop(IntExpr):
    def __init__(self, e1: IntExpr, op: Op.Op, e2: IntExpr):
        assert op in Op.Op.op_list("+ - * / << >> & | ^ ==")
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def updateName(self, expr_mapping):
        return self.__class__(
            self.e1.updateName(expr_mapping), self.op, self.e2.updateName(expr_mapping)
        )


class BoolUop(BoolExpr):
    def __init__(self, op: Op.Op, e: BoolExpr):
        assert op in Op.Op.op_list("")  # !
        self.op = op
        self.e = e

    def updateName(self, expr_mapping):
        return self.__class__(self.op, self.e.updateName(expr_mapping))


class BoolBop(BoolExpr):
    def __init__(self, e1: BoolExpr, op: Op.Op, e2: BoolExpr):
        assert op in Op.Op.op_list("&& ||")  # || ^
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def updateName(self, expr_mapping):
        return self.__class__(
            self.e1.updateName(expr_mapping), self.op, self.e2.updateName(expr_mapping)
        )


class BoolCop(BoolExpr):
    def __init__(self, e1: IntExpr, op: Op.Op, e2: IntExpr):
        assert op in Op.Op.op_list("< <= > >= == !=")  # >= <= !=
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def updateName(self, expr_mapping):
        return self.__class__(
            self.e1.updateName(expr_mapping), self.op, self.e2.updateName(expr_mapping)
        )


class CExpr(Expr):
    def __init__(self, cond: BoolExpr, et: Expr, ef: Expr):
        self.cond = cond
        self.et = et
        self.ef = ef

    def updateName(self, expr_mapping):
        return self.__class__(
            self.cond.updateName(expr_mapping),
            self.et.updateName(expr_mapping),
            self.ef.updateName(expr_mapping),
        )


class Cmd:
    pass


class CmdList:
    pass


class Assn(Cmd):
    def __init__(self, var: Var, e: Expr):
        self.var = var
        self.e = e

    def updateName(self, expr_mapping):
        return self.__class__(
            self.var.updateName(expr_mapping), self.e.updateName(expr_mapping)
        )


class If(Cmd):
    def __init__(self, cond: Expr, trueCmds: CmdList, falseCmds: CmdList = []):
        self.cond = cond
        self.trueCmds = trueCmds
        self.falseCmds = falseCmds

    def updateName(self, expr_mapping):
        trueCmdsNew = list(map(lambda cmd: cmd.updateName(expr_mapping), self.trueCmds))
        falseCmdsNew = list(
            map(lambda cmd: cmd.updateName(expr_mapping), self.falseCmds)
        )
        return self.__class__(
            self.cond.updateName(expr_mapping), trueCmdsNew, falseCmdsNew
        )


class For(Cmd):
    """
    The terminationCond keyword arg should either consist of ending integer for the loop (keyword - endInt)
    or the actual condition (keyword - endCond).
    """

    __endIntArgStr = "endInt"
    __endCondArgStr = "endCond"

    def __init__(self, var: Var, st: int, cmd_l: CmdList, fac=0, **terminationCond):
        self.var = var
        self.st = DataType.getInt(st)
        self.cmd_l = cmd_l
        self.factor = fac
        self.endInt = None
        self.endCond = None
        if self.__endIntArgStr in terminationCond:
            self.endInt = terminationCond[self.__endIntArgStr]
        elif self.__endCondArgStr in terminationCond:
            self.endCond = terminationCond[self.__endCondArgStr]
        else:
            assert False

    def updateName(self, expr_mapping):
        cmd_l_new = list(map(lambda cmd: cmd.updateName(expr_mapping), self.cmd_l))
        if self.endCond:
            return For(
                self.var,
                self.st,
                cmd_l_new,
                self.factor,
                endCond=self.cond.updateName(expr_mapping),
            )
        else:
            assert self.endInt is not None
            return For(self.var, self.st, cmd_l_new, self.factor, endInt=self.endInt)


class While(Cmd):
    def __init__(self, expr: BoolExpr, cmds: CmdList):
        self.expr = expr
        self.cmds = cmds

    def updateName(self, expr_mapping):
        cmds_new = list(map(lambda cmd: cmd.updateName(expr_mapping), self.cmds))
        return While(self.expr.updateName(expr_mapping), cmds_new)


class Comment(Cmd):
    def __init__(self, msg):
        self.msg = msg

    def updateName(self, expr_mapping):
        return self.__class__(self.msg)


class Pragmas(Cmd):
    def __init__(self, msg, vital=0):
        self.msg = msg
        self.vital = vital

    def updateName(self, expr_mapping):
        return self.__class__(self.msg, self.vital)


class Prog:
    def __init__(self, cmd_l: CmdList, resource=0):
        self.cmd_l = cmd_l
        self.resource = resource

    def updateName(self, expr_mapping):
        cmd_l_new = list(map(lambda cmd: cmd.updateName(expr_mapping), self.cmd_l))
        return self.__class__(cmd_l_new, self.resource)


class Memset(Cmd):
    # if dim==1 then single for-loop memset, else memset for 'dim'
    def __init__(self, e: Var, len: int, dim=1, lens=[]):
        self.e = e
        self.len = len
        self.dim = dim
        self.lens = lens

    def updateName(self, expr_mapping):
        return self.__class__(self.e.updateName(expr_mapping), self.len)


class Print(Cmd):
    def __init__(self, expr: Expr):
        self.expr = expr

    def updateName(self, expr_mapping):
        return self.__class__(self.expr.updateName(expr_mapping))


class PrintAsFloat(Cmd):
    def __init__(self, expr: Expr, expnt: int):
        self.expr = expr
        self.expnt = expnt

    def updateName(self, expr_mapping):
        return self.__class__(self.expr.updateName(expr_mapping), self.expnt)


class FuncCall(Cmd):
    def __init__(self, name, argList):
        self.name = name
        self.argList = argList

    def updateName(self, expr_mapping):
        argList_new = dict(
            map(
                lambda cmd: (cmd[0].updateName(expr_mapping), cmd[1]),
                self.argList.items(),
            )
        )
        return self.__class__(self.name, argList_new)


class Input(Cmd):
    def __init__(
        self,
        expr: Expr,
        shape: list,
        dataType: str,
        isSecret=True,
        inputByParty=AST.Party.SERVER,
    ):
        self.expr = expr
        self.shape = shape
        self.dataType = dataType
        self.isSecret = isSecret
        self.inputByParty = inputByParty

    def updateName(self, expr_mapping):
        return self.__class__(
            self.expr.updateName(expr_mapping),
            self.shape,
            self.dataType,
            self.isSecret,
            self.inputByParty,
        )


class Output(Cmd):
    def __init__(self, expr: Expr, outputToParty: AST.Party):
        self.expr = expr
        self.outputToParty = outputToParty

    def updateName(self, expr_mapping):
        return self.__class__(
            self.expr.updateName(expr_mapping),
            self.outputToParty,
        )


class Decl(Cmd):
    def __init__(
        self,
        varIdf: str,
        typeExpr: Type.Type,
        bitlen: int = -1,
        isSecret: bool = True,
        value: list = None,
    ):
        self.varIdf = varIdf
        self.typeExpr = typeExpr
        self.bitlen = Util.Config.wordLength if bitlen == -1 else bitlen
        self.isSecret = isSecret
        if value:
            assert isinstance(value, list)
        self.value = value

    def updateName(self, expr_mapping):
        return self.__class__(
            self.varIdf, self.typeExpr, self.bitlen, self.isSecret, self.value
        )


class DataType:

    intType = {Util.Target.EzPC: {32: np.int32, 64: np.int64}}
    intStr = {Util.Target.EzPC: "int"}
    floatStr = "float"

    @staticmethod
    def getInt(x: int, wordLen: int = None):
        if not (wordLen):
            wordLen = Util.Config.wordLength
        target = Util.Config.target
        return DataType.intType[target][wordLen](x)

    @staticmethod
    def getIntClass():
        target = Util.Config.target
        wordLen = Util.Config.wordLength
        return DataType.intType[target][wordLen]

    @staticmethod
    def getIntStr():
        target = Util.Config.target
        potentialPrefix = DataType.intStr[target]
        if target == Util.Target.EzPC:
            potentialPrefix = potentialPrefix + str(Util.Config.wordLength)
        return potentialPrefix

    @staticmethod
    def getIntStrForBitlen(bitlen):
        target = Util.Config.target
        potentialPrefix = DataType.intStr[target]
        if target == Util.Target.EzPC:
            potentialPrefix = potentialPrefix + str(bitlen)
        return potentialPrefix

    @staticmethod
    def getFloatStr():
        return DataType.floatStr

    @staticmethod
    def getNegMax():
        intClass = DataType.getIntClass()
        return intClass(np.iinfo(intClass).min)

    @staticmethod
    def getMax():
        intClass = DataType.getIntClass()
        return intClass(np.iinfo(intClass).max)
