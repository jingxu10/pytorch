import builtins
import collections
import functools
import itertools
import logging
import math
import operator
import sys
import textwrap
import threading
import traceback
from contextlib import contextmanager
from functools import lru_cache
from typing import cast, Dict, List, Optional, Set, Type, Union

import torch

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import (  # noqa: F401
    sym_float,
    sym_max,
    sym_min,
    sym_not,
    SymBool,
    SymFloat,
    SymInt,
)
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges

SymTypes = (SymInt, SymFloat, SymBool)

log = logging.getLogger(__name__)

class GuardOnDataDependentSymNode(RuntimeError):
    pass

import sympy
from sympy.printing.str import StrPrinter
from sympy.core.logic import fuzzy_and, fuzzy_or

aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = [
    "has_symbolic_sizes_strides", "create_contiguous", "ShapeEnv",
    "SymDispatchMode", "FloorDiv", "guard_int", "guard_float", "guard_scalar", "wrap_node",
    "method_to_operator", "hint_int", "SYMPY_INTERP",
]

SYM_FUNCTION_MODE = None

# We don't bother with the metaclass as all of the dispatching logic happens
# entirely from Python
#
# Didn't bother with ancestors for now, unlikely to have multiple modes for
# symints right now


# SymDispatchMode gets invoked whenever an operation is processed on
# a PySymInt.  When this occurs, you get called at __sym_dispatch__
# with the operation in question.  This is symmetric to TorchDispatchMode
# but with some caveats:
#
#   - In TorchDispatchMode, you get the same arguments as what a user
#     invoked your API with; e.g., if you call torch.ops.aten.foo(a, b),
#     you get (a, b) as args to your call.  In SymDispatchMode, if
#     you call a + b (where a and b are SymInts), you will get
#     (a.node, b.node) as your args (these are PySymInts)
#
#   - SymInt/PySymInt don't have FX proxy support (unlike, e.g., Tensor).
#     So you have to manually call Tracer/create_node to write into
#     the graph.  See ProxySymDispatchMode for an example
#
class SymDispatchMode:
    def __sym_dispatch__(self, func, types, args, kwargs):
        raise NotImplementedError()

    def __enter__(self):
        global SYM_FUNCTION_MODE
        old = SYM_FUNCTION_MODE
        if hasattr(self, "inner"):
            raise RuntimeError(f"{self} has already been used as a mode. Please use a fresh version")
        else:
            self.inner = old
        SYM_FUNCTION_MODE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global SYM_FUNCTION_MODE
        SYM_FUNCTION_MODE = self.inner

def has_symbolic_sizes_strides(elem):
    return elem._has_symbolic_sizes_strides

def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

def _handle_sym_dispatch(func, args, kwargs):
    global SYM_FUNCTION_MODE
    mode = SYM_FUNCTION_MODE
    assert mode
    SYM_FUNCTION_MODE = mode.inner
    try:
        # TODO: properly compute types
        types: List[Type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)
    finally:
        SYM_FUNCTION_MODE = mode

def hint_int(a):
    if isinstance(a, torch.SymInt):
        return a.node.require_hint()
    assert type(a) is int, a
    return a

def has_hint(a):
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True

# Returns True if every size dim on the tensor has a hint
# TODO: Should this include strides too?  For now it doesn't matter,
# that's quite an obscure case
def tensor_has_hints(t):
    return all(has_hint(s) for s in t.size())

def definitely_true(a):
    """
    Returns True only if we can tell that a is True, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression to return True.

    When is it appropriate to use definitely_true?  First, if you can use
    a higher level combinator like parallel_or/parallel_and, prefer using
    those instead, they are definitely safe (modulo short-circuiting).
    Second, it can be used if the program would behave equivalently if
    definitely_true always returned False (parallel_or/parallel_and are
    examples of this pattern, modulo short-circuiting).  Finally, it even
    be OK if the program wouldn't behave equivalently, so long as the
    change is semantics preserving.  It can be semantics preserving if
    the program errors in more cases than it did previously (but otherwise
    behaves identically), or if it changes some quantity in a way that
    doesn't matter (e.g., strides often fall in this bucket.)
    """
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return guard_bool(a)
        else:
            return False
    return bool(a)

def definitely_false(a):
    """
    Returns True only if we can tell that a is False, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression a to be False.  See definitely_true
    for more usage guidance.
    """
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return not guard_bool(a)
        else:
            return False
    return not bool(a)

# TODO: could improve parallel_or/parallel_and by avoiding guards
# if there exists a quantity that can be handled un-guardedly.  However,
# for backed SymInts, avoiding guards doesn't really matter in practice,
# so I chose not to do it.

def parallel_or(*args):
    """
    Evaluate the logical OR of several arguments, avoiding guarding on
    unbacked SymInts if another argument is definitely True.
    """
    if any(definitely_true(args) for a in args):
        return True
    return any(args)

def parallel_and(*args):
    """
    Evaluate the logical FALSE of several arguments, avoiding guarding on
    unbacked SymInts if another argument is definitely False.
    """
    if any(definitely_false(args) for a in args):
        return False
    return all(args)

def guard_scalar(a):
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f"unrecognized scalar {a}")

# inclusive both ways
def constrain_range(a, *, min: Optional[int], max: Optional[int] = None):
    if min is None:
        min = -sympy.oo
    if max is None:
        max = sympy.oo
    if not isinstance(a, SymInt):
        assert min <= a <= max
        return
    if isinstance(a.node.expr, sympy.Integer):
        assert min <= int(a.node.expr) <= max
        return
    # TODO: Turn this into a runtime assert too
    assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
    r = a.node.shape_env.var_to_range[a.node.expr]
    a.node.shape_env.var_to_range[a.node.expr] = ValueRanges(
        builtins.max(r.lower, min), builtins.min(r.upper, max)
    )


def constrain_unify(a, b):
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    # TODO: Maybe dedupe this with _maybe_guard_eq?
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
        else:
            assert isinstance(b.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
            shape_env = b.node.shape_env
            shape_env.replacements[b.node.expr] = sympy.Integer(a)
    else:
        # TODO: Actually, we can support this as long as one of them is a symbol.
        # NB: We can't actually do "unification" as our operators are not
        # injective
        assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
        shape_env = a.node.shape_env
        if not isinstance(b, SymInt):
            shape_env.replacements[a.node.expr] = sympy.Integer(b)
        else:
            assert a.node.shape_env is b.node.shape_env
            assert isinstance(b.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
            new_var = shape_env._find(a.node.expr)
            shape_env.replacements[b.node.expr] = new_var

def guard_bool(a):
    if isinstance(a, SymBool):
        return a.node.guard_bool("", 0)  # NB: uses Python backtrace
    assert type(a) is bool, a
    return a

def guard_int(a):
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    assert type(a) is int, a
    return a

def guard_float(a):
    if isinstance(a, SymFloat):
        return a.node.guard_float("", 0)  # NB: uses Python backtrace
    assert isinstance(a, float), a
    return a

# Drop in replacement for math.sqrt
def sym_sqrt(a):
    if hasattr(a, '__sym_sqrt__'):
        return a.__sym_sqrt__()
    return math.sqrt(a)

def to_node(self, num):
    if isinstance(num, SymTypes):
        return num.node
    elif type(num) is bool:
        return self.wrap_bool(num)
    elif type(num) is int:
        return self.wrap_int(num)
    elif type(num) is float:
        return self.wrap_float(num)
    else:
        # NotImplemented is important so that Python tries the
        # other magic method
        return NotImplemented

# Given a GraphModule, return all the FakeTensors for all the placeholders
def fx_placeholder_vals(gm):
    return [n.meta['val'] for n in gm.graph.nodes if n.op == "placeholder"]

def fx_placeholder_targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]

# Given a GraphModule and arguments to run it with, evaluate that the guards
# for its associated ShapeEnv are satisfied by the passed arguments.  This
# WILL check for duck sizing.
def eval_guards(gm, *args):
    return gm.shape_env.evaluate_guards_for_args(fx_placeholder_vals(gm), args)

def bind_symbols(gm, *args):
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)

# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """
    def __init__(self, expr, shape_env, pytype, hint: Optional[Union[int, float]], constant=None):
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        # What's the difference between hint and constant?
        #
        # - A constant is known to be invariant across invocations of the model;
        #   it will always be this value.  We only really know this when we
        #   encounter an honest-to-goodness literal (when wrapping it into
        #   a SymNode, we set constant.)  Most of the time, constant is None
        #
        # - A hint is a *particular* value from the particular run we are
        #   tracing, but it may vary the next time around.  It's useful to
        #   keep this around, as if we need a concrete value from a SymNode,
        #   we will return the hint and guard on the expression that produced
        #   it giving the same hint next time around.  The hint is not
        #   guaranteed to be set either: if you have an unbacked SymNode,
        #   there won't be any hint; it was the result of some tensor-dependent
        #   computation, but we don't know what it actually is because we
        #   haven't actually run the tensor computation.
        #
        # hint_expr is only set if we don't have a hint.  When it is set, it
        # contains the expression which contains the unbacked symnodes that,
        # if constrained, would allow this expression to be hinted again.
        if hint is None:
            self._hint_expr = self.expr.xreplace(shape_env.var_to_val)
            self._hint = None
            self._update_hint()  # check if the replacement actually was enough
        else:
            self._hint_expr = None
            self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant

    @property
    def expr(self):
        self._update_expr()
        return self._expr

    # Check if we have replacements hint_expr that would allow us to
    # simplify it into a hint
    def _update_hint(self):
        if self._hint_expr.free_symbols <= self.shape_env.replacements.keys():
            new_hint = self.shape_env.replace(self._hint_expr)
            # NB: unification constraints could result in a replacement that
            # doesn't actually solve the hint!  Check for this.
            if new_hint.free_symbols:
                self._hint_expr = new_hint
                return
            self._hint = self.pytype(new_hint)
            self._hint_expr = None

    @property
    def hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint

    def has_hint(self):
        return self._hint is not None

    def require_hint(self):
        if self._hint is None:
            self._update_hint()
            if self._hint is None:
                raise self.shape_env._make_data_dependent_error(self._hint_expr, self.expr)
            else:
                return self._hint
        else:
            return self._hint

    def _update_expr(self):
        self._expr = self.shape_env.replace(self._expr)

    def is_int(self):
        return self.pytype is int

    def is_float(self):
        return self.pytype is float

    def is_bool(self):
        return self.pytype is bool

    def wrap_int(self, num):
        assert type(num) is int
        return SymNode(sympy.Integer(num), self.shape_env, int, num, constant=num)

    def wrap_float(self, num):
        assert type(num) is float
        return SymNode(sympy.Float(num), self.shape_env, float, num, constant=num)

    def wrap_bool(self, num):
        assert type(num) is bool
        return SymNode(sympy.true if num else sympy.false, self.shape_env, bool, num, constant=num)

    def clone(self):
        return self

    def str(self):
        return f"{self.expr}"

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    # These methods call the metaprogrammed methods, they're hand written
    # here so we get good stack traces
    def add(self, other) -> "SymNode":  # noqa: F811
        return self._add(other)  # type: ignore[attr-defined]

    def sub(self, other) -> "SymNode":  # noqa: F811
        return self._sub(other)  # type: ignore[attr-defined]

    def mul(self, other) -> "SymNode":  # noqa: F811
        return self._mul(other)  # type: ignore[attr-defined]

    def mod(self, other) -> "SymNode":  # noqa: F811
        return self._mod(other)  # type: ignore[attr-defined]

    def pow(self, other) -> "SymNode":  # noqa: F811
        return self._pow(other)  # type: ignore[attr-defined]

    def and_(self, other) -> "SymNode":  # noqa: F811
        return self._and_(other)  # type: ignore[attr-defined]

    def or_(self, other) -> "SymNode":  # noqa: F811
        return self._or_(other)  # type: ignore[attr-defined]

    def truediv(self, other) -> "SymNode":  # noqa: F811
        return self._truediv(other)  # type: ignore[attr-defined]

    def floordiv(self, other) -> "SymNode":  # noqa: F811
        return self._floordiv(other)  # type: ignore[attr-defined]

    def sym_not(self) -> "SymNode":  # noqa: F811
        return self._sym_not()  # type: ignore[attr-defined]

    def eq(self, other) -> "SymNode":  # noqa: F811
        return self._eq(other)  # type: ignore[attr-defined]

    def ne(self, other) -> "SymNode":  # noqa: F811
        return self._ne(other)  # type: ignore[attr-defined]

    def gt(self, other) -> "SymNode":  # noqa: F811
        return self._gt(other)  # type: ignore[attr-defined]

    def lt(self, other) -> "SymNode":  # noqa: F811
        return self._lt(other)  # type: ignore[attr-defined]

    def le(self, other) -> "SymNode":  # noqa: F811
        return self._le(other)  # type: ignore[attr-defined]

    def ge(self, other) -> "SymNode":  # noqa: F811
        return self._ge(other)  # type: ignore[attr-defined]

    def floor(self) -> "SymNode":  # noqa: F811
        return self._floor()  # type: ignore[attr-defined]

    def sym_float(self) -> "SymNode":  # noqa: F811
        return self._sym_float()  # type: ignore[attr-defined]

    def sym_int(self) -> "SymNode":  # noqa: F811
        return self._sym_int()  # type: ignore[attr-defined]

    def ceil(self) -> "SymNode":  # noqa: F811
        return self._ceil()  # type: ignore[attr-defined]

    def neg(self) -> "SymNode":  # noqa: F811
        return self._neg()  # type: ignore[attr-defined]

    def sym_min(self, other) -> "SymNode":  # noqa: F811
        return self._sym_min(other)  # type: ignore[attr-defined]

    def sym_max(self, other) -> "SymNode":  # noqa: F811
        return self._sym_max(other)  # type: ignore[attr-defined]

    def sym_sqrt(self) -> "SymNode":  # noqa: F811
        return self._sym_sqrt()  # type: ignore[attr-defined]

    def is_contiguous(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_contiguous(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_2d(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_channels_last_contiguous_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_3d(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_channels_last_contiguous_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_2d(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_channels_last_strides_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_3d(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_channels_last_strides_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> "SymNode":  # noqa: F811
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)  # type: ignore[attr-defined]

    # Make C++ happy
    def sym_or(self, other):  # noqa: F811
        return self.or_(other)

    def sym_and(self, other):  # noqa: F811
        return self.and_(other)

    def is_non_overlapping_and_dense(self, sizes, strides):
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))  # type: ignore[attr-defined]

    def int_(self):
        return self.guard_int("", 0)  # NB: uses Python backtrace

    # You can manually trigger a guard with this function
    def guard_int(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(self.expr, self.hint)
        try:
            return int(r)
        except Exception:
            log.warning(f"Failed to convert to int: {r}")
            raise

    def guard_float(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(self.expr, self.hint)
        try:
            return float(r)
        except Exception:
            log.warning(f"Failed to convert to float: {r}")
            raise

    def guard_bool(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(self.expr, self.hint)
        try:
            return bool(r)
        except Exception:
            log.warning(f"Failed to convert to bool: {r}")
            raise

    def bool_(self):
        return self.guard_bool("", 0)


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
class Pow(sympy.Function):
    @classmethod
    def eval(cls, base, exp):
        if exp.is_zero:
            return sympy.Integer(1)
        elif base.is_zero and exp < 0:
            raise ZeroDivisionError(f"{base} cannot be raised to a negative power")
        else:
            return base ** exp

# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
class TrueDiv(sympy.Function):
    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        else:
            return base / divisor

class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
    """
    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    # Default return type for SymPy assumptions.
    # https://docs.sympy.org/latest/guides/assumptions.html#implementing-assumptions-handlers
    is_real = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"{base}//{divisor}"

    # SymPy assumptions based on argument types.
    def _eval_is_real(self):
        return fuzzy_or([self.base.is_real, self.divisor.is_real])

    def _eval_is_integer(self):
        return fuzzy_and([self.base.is_integer, self.divisor.is_integer])

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        def check_supported_type(x):
            if (x.is_integer is False and x.is_real is False and x.is_complex) or x.is_Boolean:
                raise TypeError(
                    f"unsupported operand type(s) for //: "
                    f"'{type(base).__name__}' and '{type(divisor).__name__}'"
                    f", expected integer or real")

        check_supported_type(base)
        check_supported_type(divisor)

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_real and divisor == 1:
            return sympy.floor(base)
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(divisor, (sympy.Integer, sympy.Float)):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd

        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return FloorDiv(
                sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
            )

# TODO: As an indicator, this != 0 implies == 1 (and vice versa).
# Because we do not have the ability to guard on the stride permutation
# at the moment, it is hard to make further inferences when this is true,
# as although we know the tensor is contiguous in *some* layout, we don't
# know which one (however, you could, for example, make the inference that
# reshaping this to a 1D tensor can be guard-free.)
class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, *args):
        assert len(args) % 2 == 0
        dim = len(args) // 2
        # TODO: it is possible to make progress evaluating this guard
        # even if not all of the inputs are known.  For example, a 2D
        # tensor with non-0/1 sizes but strides (0, 1) is definitely
        # false, because we know its numel > 1 but it's broadcasted
        # in dim 0.
        if all(isinstance(a, sympy.Integer) for a in args):
            size_args = args[0:dim]
            stride_args = args[dim:]
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in size_args],
                [int(a) for a in stride_args]
            )
        return None

IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)

@lru_cache(256)
def safe_expand(r):
    if hasattr(r, 'expand'):
        try:
            return sympy.expand(r)
        except RecursionError:
            log.warning(f"RecursionError in sympy.expand({r})")
            return r
    else:
        return r

# Methods that have a `__foo__` as well as `__rfoo__`
reflectable_magic_methods = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: a % b,
    'pow': lambda a, b: Pow(a, b),
    'and': lambda a, b: a & b,
    'or': lambda a, b: a | b,
    'truediv': lambda a, b: TrueDiv(a, b),
    'floordiv': lambda a, b: FloorDiv(a, b),
}


def error():
    raise AssertionError("shouldn't be hit")


def get_debugging_stack(num_frames_to_cut=2):
    # cut this frame and the caller's frame by default
    return ''.join(traceback.format_list(traceback.extract_stack()[:-num_frames_to_cut]))


def floor_ceil_helper(a, fn):
    if isinstance(a, sympy.Mul):
        aa = a.args
        if len(aa) == 2 and isinstance(aa[0], sympy.Float) and aa[1].is_integer:
            coef = sympy.Integer(aa[0])
            if aa[0] == coef:  # structural equality test
                return coef * aa[1]
    if isinstance(a, sympy.Float) and a == sympy.Integer(a) or isinstance(a, sympy.Integer):
        return sympy.Integer(a)
    return fn(a)

def floor_impl(a):
    return floor_ceil_helper(a, sympy.floor)

def ceil_impl(a):
    return floor_ceil_helper(a, sympy.ceiling)


magic_methods = {
    **reflectable_magic_methods,
    'sym_not': lambda a: ~a,
    'eq': lambda a, b: sympy.Eq(a, b),
    'ne': lambda a, b: sympy.Ne(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
    'le': lambda a, b: sympy.Le(a, b),
    'ge': lambda a, b: sympy.Ge(a, b),
    'floor': floor_impl,
    'sym_float': lambda a: a,  # Cannot use sympy.Float(a) here, coz it expects python literals
    'ceil': ceil_impl,
    'neg': lambda a: -a,
    'sym_min': lambda a, b: sympy.Min(a, b),
    'sym_max': lambda a, b: sympy.Max(a, b),
    'sym_sqrt': lambda a: sympy.sqrt(a),
}

sizes_strides_methods = {
    # TODO: These could also be done with indicators, maybe it is better
    # for reasoning to do it that way
    'is_contiguous': lambda sizes, strides: sympy_is_contiguous(sizes, strides),
    'is_channels_last_contiguous_2d': lambda sizes, strides: sympy_is_channels_last_contiguous_2d(sizes, strides),
    'is_channels_last_contiguous_3d': lambda sizes, strides: sympy_is_channels_last_contiguous_3d(sizes, strides),
    'is_channels_last_strides_2d': lambda sizes, strides: sympy_is_channels_last_strides_2d(sizes, strides),
    'is_channels_last_strides_3d': lambda sizes, strides: sympy_is_channels_last_strides_3d(sizes, strides),
    'is_non_overlapping_and_dense_indicator': lambda sizes, strides: IsNonOverlappingAndDenseIndicator(*sizes, *strides),
}

alternate_impl_if_hinted_methods = {
    "sym_min": builtins.min,
    "sym_max": builtins.max,
}

def sympy_is_contiguous_generic(sizes, strides, dim_order):
    dim = len(sizes)

    if len(dim_order) != dim:
        return sympy.false

    is_contiguous = sympy.true
    z = sympy.Integer(1)
    # Contiguous if the strides make sense (or the dim is size 1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.Integer(1)) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    # OR if any size is zero
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.Integer(0))
    return is_contiguous

def sympy_is_contiguous(sizes, strides):
    dim = len(sizes)
    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))

# NB: There is a TODO in C++ to allow omitting the batch dim.  If that
# happens you will need to refactor this

def sympy_is_channels_last_contiguous_2d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])

def sympy_is_channels_last_contiguous_3d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])

def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    dim = len(sizes)

    if dim != len(dim_order):
        return sympy.false

    m = sympy.Integer(0)
    r = sympy.true

    # special case for trivial C dimension. default to NCHW
    r &= sympy.Ne(strides[1], 0)

    for d in dim_order:
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)
        # Fallback to NCHW as default layout for ambiguous cases
        # This is the flaw of implicit memory_format from strides.
        # N111 tensor with identical strides for size 1 dimension;
        # Two cases could lead us here:
        # a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        # b. N11W contiguous Tensor sliced on the W-dimension.
        # ([N,1,1,1]@[W,W,W,W])
        if d == 0:
            r &= sympy.Ne(m, strides[1])
        # This is necessary to:
        # 1. distinguish the memory_format of N1H1;
        #     [H, 1, 1, 1] channels_last stride
        #     [H, H, 1, 1] contiguous stride
        # 2. permutation of 1C1W:
        #     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
        #     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as
        #     channels_last
        m = strides[d] * sympy.Max(sizes[d], 1)

    return r

def sympy_is_channels_last_strides_2d(sizes, strides):
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])

def sympy_is_channels_last_strides_3d(sizes, strides):
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])

# TODO: Deduplicate this with torch/_prims_common/__init__.py
def eval_is_non_overlapping_and_dense(sizes, strides):
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))

def _eval_is_non_overlapping_and_dense(sizes, strides):
    dim = len(sizes)

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    # or it is a 0/1 element tensor
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    lengths_and_strides = sorted(
        zip(sizes, strides), key=operator.itemgetter(1)
    )

    # Unlike the C++ code, we don't move the 0/1 size dimensions to the
    # end.  So we have to keep going for this code.
    expected_stride = 1
    for length, stride in lengths_and_strides:

        if length == 1:
            continue

        if stride != expected_stride:
            return False

        expected_stride *= length

    return True

unary_magic_methods = {
    'sym_float',
    'ceil',
    'floor',
    'neg',
    'sym_sqrt',
    'sym_not',
}

bool_magic_methods = {"and", "or", "sym_not"}

magic_methods_on_math = {"ceil", "floor"}
magic_methods_on_submodule = {"sym_float", "sym_sqrt", "sym_min", "sym_max", "sym_not"}
magic_methods_on_operator_with_trailing_underscore = {"and", "or"}

def method_to_operator(method):
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method
    if method in magic_methods_on_submodule:
        op = getattr(torch.fx.experimental.symbolic_shapes, method_attr)
    elif method in magic_methods_on_math:
        op = getattr(math, method_attr)
    else:
        op = getattr(operator, method_attr)
    return op

SYMPY_INTERP = {
    'Eq': operator.eq,
    'Ne': operator.ne,
    'Gt': operator.gt,
    'Lt': operator.lt,
    'Le': operator.le,
    'Ge': operator.ge,
    'Min': min,
    'Max': max,
    'Mod': operator.mod,
    'FloorDiv': operator.floordiv,
    'TrueDiv': operator.truediv,
    'IsNonOverlappingAndDenseIndicator': eval_is_non_overlapping_and_dense,
    'floor': math.floor,
    'ceiling': math.ceil,
}

always_float_magic_methods = {"truediv", "sym_float", "sym_sqrt", "pow"}
always_int_magic_methods = {"ceil", "floor"}
always_bool_magic_methods = {"eq", "ne", "gt", "lt", "le", "ge", "and", "or", "sym_not", "is_non_overlapping_and_dense"}

def wrap_node(x):
    # TODO: let C++ also take advantage of this
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    elif x.is_bool():
        return SymBool(x)
    else:
        raise AssertionError(f"unrecognized return type {x}")

def _make_node_magic(method, func):
    func = lru_cache(256)(func)

    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

    def binary_magic_impl(self, other):
        op = method_to_operator(method)

        out_hint = None
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)

        alternate_impl = alternate_impl_if_hinted_methods.get(method)
        if alternate_impl and out_hint is not None:
            return to_node(self, alternate_impl(wrap_node(self), wrap_node(other)))

        if SYM_FUNCTION_MODE:
            return to_node(self, _handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {}))
        assert isinstance(other, SymNode)
        other_expr = other.expr
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        other_expr = self.shape_env.replace(other_expr)
        try:
            out = func(expr, other_expr)
        except Exception:
            log.warning(f"failed to eval {method}({expr}, {other_expr})")
            raise
        out = safe_expand(out)
        pytype: Type
        # This is not strictly correct. In Python, a**b may return complex when
        # a < 0 and b is a float: (-1)**2.1. Same for sympy.sqrt(-3.14). This
        # returns a float while both arguments are ints: 2**(-1). Also, max and
        # min do not type promote. To avoid having data-dependent control flow
        # here, we just set the type to float if one of the args is a float. In
        # case of a type mismatch, we assume that it will be detected during
        # evaluation.
        if method in always_float_magic_methods:
            pytype = float
        elif method in always_bool_magic_methods:
            pytype = bool
        elif self.pytype is float or other.pytype is float:
            pytype = float
        else:
            pytype = self.pytype

        return SymNode(out, self.shape_env, pytype, out_hint)

    def unary_magic_impl(self):
        op = method_to_operator(method)
        if SYM_FUNCTION_MODE:
            return to_node(self, _handle_sym_dispatch(op, (wrap_node(self),), {}))
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        if method == "floor" or method == "ceiling":
            expr = self.shape_env._simplify_floor_div(expr)

        try:
            out = func(expr)
        except Exception:
            log.warning(f"failed to eval {method}({expr})")
            raise

        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        out = safe_expand(out)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        return SymNode(out, self.shape_env, pytype, out_hint)

    if method in unary_magic_methods:
        setattr(SymNode, f"_{method_attr}", unary_magic_impl)
    else:
        setattr(SymNode, f"_{method_attr}", binary_magic_impl)

def _make_node_sizes_strides(method, func):
    # NB: don't LRU cache, lots of arguments

    def sizes_strides_impl(self, sizes, strides):
        op = getattr(sys.modules[__name__], method)
        if SYM_FUNCTION_MODE:
            return to_node(
                self,
                _handle_sym_dispatch(
                    op,
                    ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]),
                    {}
                )
            )
        size_exprs = [s.expr for s in sizes]
        stride_exprs = [s.expr for s in strides]
        try:
            out = func(size_exprs, stride_exprs)
        except Exception:
            log.warning(f"failed to eval {method}({size_exprs}, {stride_exprs})")
            raise
        # bool is never expandable

        size_hints = []
        out_hint = None
        for s in sizes:
            if s.hint is None:
                break
            size_hints.append(s.hint)
        else:
            stride_hints = []
            for s in strides:
                if s.hint is None:
                    break
                stride_hints.append(s.hint)
            else:
                out_hint = op(size_hints, stride_hints)

        # NB: This is the indicator function, not the actual bool!
        pytype: Type
        if method.endswith("_indicator"):
            pytype = int
        else:
            pytype = bool
        return SymNode(out, self.shape_env, pytype, out_hint)

    setattr(SymNode, f"_{method}", sizes_strides_impl)

    # TODO: This is technically hotpath, but in the ideal end state
    # guards on this will resolve at a higher level so you never
    # spend time in this code
    def sizes_strides_user(sizes, strides):
        for a in itertools.chain(sizes, strides):
            if isinstance(a, SymInt):
                return wrap_node(getattr(a.node, method)(
                    [to_node(a.node, b) for b in sizes],
                    [to_node(a.node, b) for b in strides],
                ))
        if method == "is_non_overlapping_and_dense_indicator":
            return eval_is_non_overlapping_and_dense(sizes, strides)
        else:
            # TODO: this is an awful implementation
            return bool(func(
                [sympy.sympify(a) for a in sizes],
                [sympy.sympify(a) for a in strides],
            ))

    # Skip for is_non_overlapping_and_dense_indicator
    if not hasattr(sys.modules[__name__], method):
        setattr(sys.modules[__name__], method, sizes_strides_user)

for method, func in magic_methods.items():
    _make_node_magic(method, func)

for method, func in sizes_strides_methods.items():
    _make_node_sizes_strides(method, func)

def _make_user_magic(method, user_type):
    # User magic takes care of wrapping the other operand into a node,
    # so that our internal logic can assume everything is nodes

    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

    def unary_magic_impl(self):
        return wrap_node(getattr(self.node, method_attr)())

    def binary_magic_impl(self, other):
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        return wrap_node(getattr(self.node, method_attr)(other_node))

    def rbinary_magic_impl(self, other):
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        return wrap_node(getattr(other_node, method_attr)(self.node))

    if method in unary_magic_methods:
        setattr(user_type, f"__{method}__", unary_magic_impl)
    else:
        setattr(user_type, f"__{method}__", binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f"__r{method}__", rbinary_magic_impl)

for method, func in magic_methods.items():
    if method in bool_magic_methods:
        _make_user_magic(method, SymBool)
    else:
        _make_user_magic(method, SymInt)
        _make_user_magic(method, SymFloat)

del method
del func

def _lru_cache(fn, maxsize=None):
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_key = None

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        nonlocal prior_key
        if prior_key != self._get_key():
            prior_key = self._get_key()
            fn_cache.cache_clear()
        return fn_cache(self, *args, **kwargs)

    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper


class ShapeGuardPrinter(StrPrinter):
    def __init__(
        self,
        symbol_to_source,
        source_ref,
        var_to_sources,
    ):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))

        def repr_symbol_to_source():
            return repr({
                symbol: [s.name() for s in sources]
                for symbol, sources in self.symbol_to_source.items()
            })

        assert expr in self.symbol_to_source, (
            f"{expr} (could be from {[s.name() for s in self.var_to_sources[expr]]}) "
            f"not in {repr_symbol_to_source()}.  If this assert is failing, it could be "
            "due to the issue described in https://github.com/pytorch/pytorch/pull/90665"
        )
        return self.source_ref(self.symbol_to_source[expr][0])


class LoggingShapeGuardPrinter(ShapeGuardPrinter):
    def __init__(self, var_to_sources):
        super().__init__(var_to_sources, lambda n: n.name(), var_to_sources)


TLS = threading.local()


class ShapeEnv:
    def __init__(
        self, *,
        allow_scalar_outputs=True,
        allow_dynamic_output_shape_ops=True,
        strict_mark_dyn=False,
        assume_static_by_default=False,
        # The following options affect decisions we make about eager
        # specialization.  Disabling them will increase trace time (as we do
        # more symbolic reasoning) and can also harm the quality of generated
        # code (because inductor may not be able to specialize for bounds
        # being equal--although if we later respecialize because of a guard,
        # your code may be just as good as it was before.)
        #
        # When True, eagerly specialize input sizes which have 0/1.
        specialize_zero_one=True,
        # When True, assume input sizes which have the same size are
        # symbolically equal.
        duck_shape=True,
    ):
        # Not directly used by ShapeEnv; indirectly used by FakeTensor
        self.allow_scalar_outputs = allow_scalar_outputs
        self.allow_dynamic_output_shape_ops = allow_dynamic_output_shape_ops
        self.guards: List[ShapeGuard] = []
        # Maps symbolic ints to their original concrete values
        # Currently populated from tensors
        self.var_to_val: Dict["sympy.Symbol", "sympy.Integer"] = {}
        # Maps symbolic ints to their min/max range.  These ranges
        # are conservative: the int MUST fall in the range, but the
        # range may contain ints which may not actually appear in
        # practice
        self.var_to_range: Dict["sympy.Symbol", ValueRanges] = {}
        self.var_to_sources: Dict["sympy.Symbol", List[Source]] = {}
        self.var_to_stack: Dict["sympy.Symbol", str] = {}
        # Maps from sympy ints to expressions representing them
        # Populated from equality guards (i.e. a.shape[0] == b.shape[0])
        self.replacements: Dict["sympy.Symbol", "sympy.Expr"] = {}  #
        # Set holds a % b expressions that evaluate to 0.
        self.divisible: Set["sympy.Expr"] = set()
        # Duck-shaping says that if two input tensors have the same size,
        # they get assigned the same symbolic variable
        self.val_to_var: Dict[int, "sympy.Expr"] = {}
        if specialize_zero_one:
            self.val_to_var = {0: sympy.Integer(0), 1: sympy.Integer(1)}
        self.unbacked_symfloat_counter = itertools.count()
        self.unbacked_symint_counter = itertools.count()
        self.strict_mark_dyn = strict_mark_dyn
        self.assume_static_by_default = assume_static_by_default
        self.specialize_zero_one = specialize_zero_one
        self.duck_shape = duck_shape

    def _suppress_guards_tls(self):
        return getattr(TLS, "suppress_guards", False)

    @contextmanager
    def suppress_guards(self):
        TLS.suppress_guards = True
        try:
            yield
        finally:
            TLS.suppress_guards = False

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible))

    def _produce_dyn_sizes(self, ex: torch.Tensor, source: Source) -> List[sympy.Expr]:
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size = []
        for i, val in enumerate(ex.size()):
            is_dynamic = _is_dim_dynamic(ex, i)
            if _should_allocate(is_dynamic, self.assume_static_by_default):
                size.append(self.create_symbol(
                    val, TensorPropertySource(source, TensorProperty.SIZE, i), is_dynamic
                ))
            else:
                size.append(sympy.Integer(val))
        return size

    def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor, source: Source):
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size: List[sympy.Expr] = self._produce_dyn_sizes(ex, source)
        stride: List[Optional[sympy.Expr]] = [None] * len(size)
        for i, val in enumerate(ex.stride()):
            if val in (0, 1):
                stride[i] = sympy.Integer(val)
        while any(x is None for x in stride):
            candidates = {
                ex.size(i) * ex.stride()[i]: size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None and ex.stride()[i] >= 0
            }
            # iterate over unbound strides in sorted order
            val_list = sorted(
                [(ex.stride()[i], i) for i in range(len(stride)) if stride[i] is None]
            )
            for _, i in val_list:
                if stride[i] is None and ex.stride()[i] in candidates:
                    stride[i] = candidates[ex.stride()[i]]
                    candidates[ex.size(i) * ex.stride()[i]] = size[i] * stride[i]
            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = min(
                    [
                        (ex.stride()[i], i)
                        for i in range(len(stride))
                        if stride[i] is None
                    ]
                )
                stride[i] = self.create_symbol(
                    val,
                    TensorPropertySource(source, TensorProperty.STRIDE, i)
                )
        assert all(x is not None for x in stride)
        sym_size = [self.create_symintnode(i, hint=hint) for i, hint in zip(size, ex.size())]
        sym_stride = []
        for i, stride_expr in enumerate(stride):
            # NB: Don't duck size the stride; instead use the expression
            # we computed
            assert stride_expr is not None
            sym_stride.append(self.create_symintnode(stride_expr, hint=ex.stride(i)))
        sym_storage_offset = self.create_symintnode(self.create_symbol(
            ex.storage_offset(),
            TensorPropertySource(source, TensorProperty.STORAGE_OFFSET)
        ), hint=ex.storage_offset())
        return sym_size, sym_stride, sym_storage_offset

    # If you know what the current hint value of the SymInt to be created
    # is, pass it into hint.  Otherwise, pass None and we will make our best
    # guess
    def create_symintnode(self, sym: "sympy.Expr", *, hint: Optional[int]):
        return SymInt(SymNode(sym, self, int, hint))

    def create_unbacked_symfloat(self):
        symbol = sympy.Symbol(f"f{next(self.unbacked_symfloat_counter)}")
        self.var_to_stack[symbol] = ''.join(traceback.format_list(traceback.extract_stack()[:-1]))
        self.var_to_range[symbol] = ValueRanges.unknown()
        return SymFloat(SymNode(symbol, self, float, None))

    def create_unbacked_symint(self):
        symbol = sympy.Symbol(f"i{next(self.unbacked_symint_counter)}", integer=True)
        self.var_to_stack[symbol] = ''.join(traceback.format_list(traceback.extract_stack()[:-1]))
        self.var_to_range[symbol] = ValueRanges(-sys.maxsize - 1, sys.maxsize)
        return SymInt(SymNode(symbol, self, int, None))

    # This is guaranteed to return a symbol or its negation is a sympy.Symbol,
    # but there may be a replacement that allows it to be immediately
    # simplified
    def create_symbol(self, val: int, source: Source, dyn=False) -> "sympy.Expr":
        assert isinstance(source, Source), f"{type(source)} {source}"

        if val < 0:
            from torch._dynamo.source import NegateSource
            return -self.create_symbol(-val, NegateSource(source), dyn)

        if dyn or val not in self.val_to_var or not self.duck_shape:
            # If a value is never before seen, or dynamic, we want to create an expression
            sympy_expr = sympy.Symbol(f"s{len(self.var_to_val)}", positive=True, integer=True)
            # We always associate vars to vals
            self.var_to_val[sympy_expr] = sympy.Integer(val)
            # Do the appending later, because we always want to populate this
            self.var_to_sources[sympy_expr] = []

            if not dyn:
                # Non explicitly marked dynamic dims register to val_to_var to get duck shaped
                self.val_to_var[val] = sympy_expr

            # We also infer that it must be not 0/1
            lower = 2 if self.specialize_zero_one else 0
            # NB: sys.maxsize is NOT allowed for sizes, because we use MAX_INT
            # as a sentinel sometimes.  Your sizevar isn't going to be
            # anywhere near the max 64-bit integer anyway.
            self.var_to_range[sympy_expr] = ValueRanges(lower, sys.maxsize - 1)

        if not dyn and self.duck_shape:
            # This implements duck-shaping: input sizes that match are assigned
            # the same symint
            r = self.duck_int(val)
        else:
            r = sympy_expr

        if isinstance(r, sympy.Symbol):
            self.var_to_sources[r].append(source)
        return r

    # Given a concrete integer value, return the duck sized symbol associated
    # with it; e.g., suppose we already have a tensor of size 3 in scope,
    # which was assigned s3, then shape_env.duck_int(3) we will get back s3.
    # This has some pretty tricky preconditions associated with it, so if
    # you are in a binding context, you probably wanted create_symbol instead.
    def duck_int(self, val):
        assert self.duck_shape
        assert val in self.val_to_var, (
            "Direct call to duck_int MUST only duck size an integer values "
            "that have already produced by inputs (allocated "
            "by create_symbol), or we risk being unable to instantiate the "
            "symbolic variable later.  However, at time of this call "
            f"val={val} was not duck sized.  Bound duck sized integers: "
            f"{list(self.val_to_var.keys())}"
        )
        return self.val_to_var[val]

    # Generates a list of guards strings which, when evaluated in a context that
    # defines tensors for all the sources, returns True or False depending
    # on if the guards in the list evaluated to True or not.  Primarily used by Dynamo,
    # but this is also helpful for manual testing of guards (see
    # evaluate_guards_for_args)
    #
    # For convenience in testing, a source is allowed to be a str,
    # in which case we will assume it is a LocalSource
    #
    # simplified lets you omit duck sizing, equality and 0/1 guards.
    # This is useful for testing when you don't care about the boilerplate
    # guards, and it may be helpful for user output too (be careful though;
    # some equality guards are nontrivial!  It would be nice to get simplified
    # output to print them too).  It's private because it's not
    # intended for normal use
    def produce_guards(self, placeholders, sources,
                       source_ref=lambda n: n.name(), *, _simplified=False) -> List[str]:
        # It took a lot of sweat to figure out the algorithm here.  Let's
        # explain how it works.
        #
        # The ShapeEnv lifecycle looks something like this:
        #
        # - For each input, you either generate a fresh Sympy symbol (s0) to
        #   represent its value (a binding site), or you reuse some
        #   preexisting symbol or expression, skipping the symbol allocation
        #   (e.g., duck sizing to a preexisting symbol, or expressing a
        #   stride as a multiplication of a separate stride and size.)
        #   Naively, you might expect to bind a fresh Sympy symbol for
        #   every input, but this is fairly wasteful as most of these
        #   symbols immediately simplify away, and if you don't eagerly
        #   specialize, e.g., 0/1 symbols, you end up with very complicated
        #   expressions that are not optimizable in practice.
        #
        # - You perform some compute on these symbols, occasionally
        #   introducing guards on boolean expressions on these symbols.
        #   In particular, whenever we guard on equality (_maybe_guard_eq),
        #   we can simplify shapes; e.g., when s0 == s1 * 2, we can now
        #   replace all occurrences of s0 with s1 * 2.  Sometimes, a
        #   boolean expression evaluation doesn't introduce a guard, as
        #   the guard is already entailed by the simplifications we have
        #   applied.
        #
        # - In the end, you have a bunch of replacements (saying how to
        #   simplify shapes) and a bunch of guards (all the equality guards
        #   are trivial, because they're covered by the replacements).
        #
        # From the ShapeEnv, we must generate a Python expression that, when
        # evaluated on a set of inputs, tells us whether or not these boolean
        # expressions would have evaluated in the same way.  However,
        # we cannot easily compute this, as we elide recording boolean
        # expressions when we think they are vacuously true.  Thus, we seek
        # an approximation: we must generate an expression, if true, would have
        # produced an "equivalent" ShapeEnv, which would answer guard
        # expressions in the same way.
        #
        # Our notion of equivalence is a bit subtle.  For example, consider
        # the ShapeEnv created from an input of size (5, 4) versus (4, 4)
        # (no other guards.)  Duck sizing would generate (s0, s1) in the first
        # case but (s0, s0) in the second.  We do NOT assume that size
        # variables are disjoint; so in fact a graph that assumes the input
        # could be (s0, s1) subsumes (s0, s0) (setting s0 == s1), but not
        # vice versa.  However, consider an analogous case (1,) versus (2,).
        # Duck sizing generates (1,) and (s0,); the (s0,) graph does NOT
        # subsume the (1,) graph because we assume that any size variables
        # is NOT 0/1 (and make simplifications according to this; e.g., if
        # we queried s0 == 0, we would immediately return False without
        # returning a guard.)
        #
        # So, it is perhaps easier to flip things on their head: the guard
        # expressions we generate here say what simplifications are valid,
        # and what are not.  Below, we explain each of the guard expressions
        # we generate

        # TODO: Make this more efficient by binding all the size/stride/offsets
        # to locals before performing tests on them.

        from torch._dynamo.source import NegateSource, TensorPropertySource, TensorProperty

        # Actual codegen must be delayed as we don't necessarily know what
        # the symbol mapping is
        input_guards = []

        symbol_to_source = collections.defaultdict(list)
        dynamic_sources = []

        # How do we know what the value of s0 is?  Fresh variables can only be
        # bound by inputs, so there MUST be some other input which binds the
        # variable.  If there is no such input, this is an error in our
        # system.  We record where all symbols come from, to help you diagnose
        # why those symbols didn't occur.
        #
        # In fact, generally speaking it is only possible for the "outermost"
        # user of a ShapeEnv to evaluate the guards, because some inputs may
        # not be available to inner levels.  For example, Dynamo can guard on
        # tensors that never actually become graph arguments (they are
        # pruned).  In this case, only Dynamo knows about these arguments.
        def track_symint(source, val):
            if isinstance(val, SymInt):
                s = val.node.expr

                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                elif isinstance(-s, sympy.Symbol):
                    symbol_to_source[-s].append(NegateSource(source))
                input_guards.append((source, s))
            else:
                input_guards.append((source, sympy.Integer(val)))

        def _verify(expr, potential_expr):
            # An expression of > 1 symbols is a relationship,
            # and relationships can be ignored due to the nature of the
            # constraint api explicitly not supporting relationships.
            #
            # In a future where we want to extend the constraint API to include
            # user directives about relationships, we can remove this check from
            # verification.
            if len(expr.free_symbols) == 1:
                srcs = symbol_to_source[expr.free_symbols.pop()]
                for src in srcs:
                    if src in dynamic_sources:
                        raise RuntimeError(f"Attempting to introduce a guard {potential_expr} that violates user's mark_dynamic")

        for t, source in zip(placeholders, sources):
            if isinstance(source, str):
                from torch._dynamo.source import LocalSource
                source = LocalSource(source)
            assert isinstance(source, Source)
            if t is None:
                continue
            if isinstance(t, SymInt):
                track_symint(source, t)
                continue
            assert isinstance(t, torch.Tensor)
            for i, ss in enumerate(t.size()):
                property_source = TensorPropertySource(source, TensorProperty.SIZE, i)
                track_symint(property_source, ss)
                if _is_dim_dynamic(t, i):
                    # If this dim is marked dynamic, we need to do a test on it, to ensure that it has not bee
                    # constrained to an integer.
                    if _is_int(ss):
                        raise RuntimeError(f"Attempting to constrain dim {i} for {source}, which violates user's mark_dynamic")
                    dynamic_sources.append(property_source)
            for i, ss in enumerate(t.stride()):
                track_symint(TensorPropertySource(source, TensorProperty.STRIDE, i), ss)
            track_symint(TensorPropertySource(source, TensorProperty.STORAGE_OFFSET), t.storage_offset())

        # 1. Every input must equal the final simplified symbolic expression
        #    stored on the placeholder.  Given a placeholder (s0*2, s1),
        #    if we have an input (2, 3), we must show s0*2 == 2 and s1 == 3.
        #    This does a lot of work: it covers duck sizing and equality guards.
        exprs = []
        if not _simplified:
            for source, expr in input_guards:
                # Small optimization
                if (
                    isinstance(expr, sympy.Symbol) and
                    expr in symbol_to_source and
                    source == symbol_to_source[expr][0]
                ):
                    continue
                sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(f"{source_ref(source)} == {sexpr}")

        # 2. Every guard must evaluate to True (but remember many guards
        #    like s0 == s1*2 because trivial due to simplification)
        for g, tb in self.guards:
            if self._maybe_evaluate_static(g) is not None:
                continue
            g = self.simplify(g)
            try:
                guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(g)
                exprs.append(guard_expr)
                if self.strict_mark_dyn:
                    _verify(g, guard_expr)
            except Exception:
                log.warning(f"Failing guard allocated at: \n{tb}")
                raise

        # 3. Every symbol must be within its value range (this handles 0/1
        # specialization too).  NB: because we never update value ranges
        # except in case of explicit user annotation, these are not included
        # in simplified.  However, when we start updating value ranges
        # these should probably get reported in tests too
        if not _simplified:
            for symbol, sources in symbol_to_source.items():
                assert sources
                assert symbol.is_integer
                r = self.var_to_range[symbol]
                bounds = []
                if r.lower != -sympy.oo:
                    bounds.append(str(r.lower))
                bounds.append(source_ref(sources[0]))
                # NB: This looks like an off-by-one error but it's not: the
                # upper bound may be sys.maxsize - 1 because we intentionally
                # exclude sys.maxsize from our bounds to deal with direct
                # == INT_MAX guards, but it's still dumb to actually test it.
                # Note that you can be off by a pretty large constant and it
                # won't matter because sizes in practice will be no where near
                # the 64-bit limit.
                if r.upper != sympy.oo and r.upper < sys.maxsize - 1:
                    bounds.append(str(r.upper))
                if len(bounds) > 1:
                    exprs.append(" <= ".join(bounds))

        return exprs

    def evaluate_guards_for_args(self, placeholders, args):
        from torch._dynamo.source import GlobalSource
        arg_names = [f"t{i}" for i in range(len(args))]
        guards = self.produce_guards(placeholders, [GlobalSource(a) for a in arg_names])
        if guards:
            code = " and ".join(guards)
            return eval(code, SYMPY_INTERP, dict(zip(arg_names, args)))
        return True

    def bind_symbols(self, placeholders, args):
        # Given a paired list of placeholders (fake tensors with
        # symbolic sizes) and concrete arguments (regular tensors
        # with real sizes), returns a dictionary mapping each
        # symbol to its real value.  So for example, if you
        # have a placeholder with size (s0, s1), binding
        # (2, 4) to it will give you {s0: 2, s1: 4}.  This is
        # not guaranteed to bind ALL symbols in the ShapeEnv;
        # we can't bind a symbol if it doesn't occur in any placeholder,
        # and symbols that already have replacements won't get bindings.

        # This is a little duplicative with evaluate_guards but
        # it's different enough that it seemed cleanest to make
        # another copy.  This assumes the guards are already checked,
        # though if it's cheap we'll check for shenanigans
        bindings: Dict[sympy.Symbol, int] = {}

        def bind_symint(arg, val):
            if isinstance(val, SymInt):
                s = val.node.expr

                if isinstance(s, sympy.Symbol):
                    if s in bindings:
                        assert bindings[s] == arg, f"{bindings[s]} != {arg}"
                    else:
                        bindings[s] = arg
                elif isinstance(-s, sympy.Symbol):
                    if -s in bindings:
                        assert bindings[-s] == -arg, f"{bindings[-s]} != {-arg}"
                    else:
                        bindings[-s] = -arg

        for t, arg in zip(placeholders, args):
            if t is None:
                continue
            if isinstance(t, SymInt):
                bind_symint(arg, t)
                continue
            assert isinstance(t, torch.Tensor)
            for i, s in enumerate(t.size()):
                bind_symint(arg.size(i), s)
            for i, s in enumerate(t.stride()):
                bind_symint(arg.stride(i), s)
            bind_symint(arg.storage_offset(), t.storage_offset())

        return bindings

    def get_nontrivial_guards(self):
        return [self.simplify(guard.expr) for guard in self.guards if self._maybe_evaluate_static(guard.expr) is None]

    def format_guards(self, verbose=False):
        def format_tb(tb):
            if not verbose:
                return ""
            return f"\n   Guarded at:\n{textwrap.indent(tb, '   ')}"

        return '\n'.join(f" - {guard.expr}{format_tb(guard.stack)}" for guard in self.guards)

    def get_shape_groups(self):
        shape_groups = collections.defaultdict(list)
        for k, v in self.replacements.items():
            shape_groups[v].append(k)
        return shape_groups

    @_lru_cache
    def _maybe_evaluate_static(self, expr: "sympy.Expr", *, unbacked_only: bool = False) -> "Optional[sympy.Expr]":
        """
        Tries to evaluate expr without introducing guards
        """
        expr = self.simplify(expr)

        # Simplify making use of value range lower bound
        symbols = list(expr.free_symbols)
        new_shape_env = {}
        new_range_env = {}
        for idx, k in enumerate(symbols):
            vr = self.var_to_range[k]
            # Don't do anything if we don't have a nontrivial lower bound
            # Also don't do anything if we asked only to simplify unbacked
            # SymInt
            if vr.lower == -sympy.oo or (unbacked_only and k in self.var_to_val):
                new_range_env[k] = vr
                continue
            # Positive means >= 1
            # Positive - 1 means >= 0
            # Positive + lower - 1 means >= lower
            # The new symbol 's' is "too low", so when we substitute it in
            # we have to increase it by offset (and conversely, the new
            # variables have to have their value range bounds adjusted as
            # well)
            s = sympy.Symbol(f"shape_{idx}", positive=True, integer=True)
            offset = vr.lower - 1
            new_shape_env[k] = s + offset
            new_range_env[s] = ValueRangeAnalysis.sub(vr, offset)

        new_expr = expr.xreplace(new_shape_env)
        floor_div_replace = {}
        for atom in new_expr.atoms(FloorDiv):
            floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
        new_expr = safe_expand(new_expr.xreplace(floor_div_replace))
        # TODO: when unbacked_only, can sometimes early return even when there
        # are still free symbols
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr

        # Check if the range can solve it statically
        out = sympy_interp(ValueRangeAnalysis, new_range_env, new_expr)
        if out.is_singleton():
            return out.lower

        return new_expr if unbacked_only else None

    @_lru_cache
    def replace(self, expr: "sympy.Expr") -> "sympy.Expr":
        replacements = {s: self._find(cast(sympy.Symbol, s)) for s in expr.free_symbols}
        return safe_expand(expr.xreplace(replacements))

    @_lru_cache
    def _update_divisible(self):
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if len(res.free_symbols) > 0:
                new_divisible.add(k)

        self.divisible = new_divisible

    @_lru_cache
    def simplify(self, expr: "sympy.Expr") -> "sympy.Expr":
        expr = self.replace(expr)
        # TODO it would seem that this pass is not necessary given the
        # below replacement of // with /, but for nested FloorDivs
        # the non-recursive replacement doesn't work, and
        # recursive makes it hard to look up divisibility,
        # because existing divisibility info has FloorDiv in it, not /
        # for now just do a separate pass to catch common nested case
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if isinstance(divisor, FloorDiv):
                    base1, divisor1 = divisor.args
                    if self.replace(base % divisor) in self.divisible and \
                            base == base1 and self.replace(base1 % divisor1) in self.divisible:
                        div_replacements[atom] = divisor1
            expr = expr.xreplace(div_replacements)
            expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(base % divisor) in self.divisible:
                    div_replacements[fd] = base / divisor
            new_expr = expr.xreplace(div_replacements)
            new_expr = safe_expand(new_expr)
            new_pows = new_expr.atoms(sympy.Pow)
            new_rationals = new_expr.atoms(sympy.Rational).difference(new_expr.atoms(sympy.Integer))
            # divisions simplified away
            if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                expr = new_expr
        return expr

    @lru_cache(256)
    def size_hint(self, expr: "sympy.Expr"):
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        if len(result_expr.free_symbols) != 0:
            r = self._maybe_evaluate_static(result_expr)
            if r is not None:
                return r
            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    def _make_data_dependent_error(self, expr, unhinted_expr):
        # TODO: in a Dynamo context, having user code, and having the
        # name of the local, will be much better
        for s in expr.free_symbols:
            log.debug(f"Data dependent variable '{s}' allocated at:\n{self.var_to_stack[s]}")
        return GuardOnDataDependentSymNode(
            "It appears that you're trying to get a value out of symbolic int/float "
            "whose value is data-dependent (and thus we do not know the true value.)  "
            f"The expression we were trying to evaluate is {expr} (unhinted: {unhinted_expr}).  "
            "Scroll up to see where each of these data-dependent accesses originally occurred."
            # TODO: Help text about how to use our runtime tests to fix this
            # problem
        )

    def _set_replacement(self, a: "sympy.Symbol", expr: "sympy.Expr") -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = expr`.
        """
        if torch._dynamo.config.print_specializations and isinstance(expr, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected

            # NOTE(avik): It is possible that we try logging the same specialization multiple times, e.g.,
            # when adding a to self.replacements, and again when simplifying an expression containing a.
            # Thus to avoid duplication, checking whether a is in self.replacements isn't enough; if it is,
            # it must not already map to `expr`. Fortunately this check is cheap because `expr` is a constant.
            if a not in self.replacements or expr != self.replacements[a]:
                log.warning(f"Specializing {self.var_to_sources[a][0].name()} to {expr}")
                log.debug("SPECIALIZATION", stack_info=True)
        self.replacements[a] = expr

    @_lru_cache
    def _find(self, a: "sympy.Symbol") -> "sympy.Expr":
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d
        """
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        self._set_replacement(a, self.replacements[a].xreplace(cur_replace))
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_eq(self, expr: Union["sympy.Eq", "sympy.Ne"], concrete_bool: bool) -> None:
        """
        Evaluates the result of an eq call. If true, uses information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        assert type(concrete_bool) is bool
        if isinstance(expr, sympy.Eq):
            if not concrete_bool:
                return
        # NB: Apparently this is load bearing; to see what test fails if
        # you comment it out run:
        # python test/functorch/test_aotdispatch.py -k
        # test_aot_autograd_symbolic_module_exhaustive_nn_LazyConv3d_cpu_float32
        elif isinstance(expr, sympy.Ne):
            if concrete_bool:
                return
        free = list(expr.free_symbols)

        assert len(free) > 0, f"The expression should not be static by this point: {expr}"
        # In case of really gnarly expression, we don't blow up
        if len(free) > 5:
            return
        free = sorted(free, key=lambda x: (self.size_hint(x), x.name), reverse=True)  # type: ignore[attr-defined]
        lhs = expr.lhs
        rhs = expr.rhs
        if not expr.has(sympy.Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any([a.divisor != 1 for a in floor_div_atoms]):
                    raise NotImplementedError
                solutions = sympy.solve(lhs - rhs, free[0], dict=True)
                if len(solutions) != 1:
                    return
                solution = solutions[0][free[0]]
                if all(t.is_integer for t in sympy.preorder_traversal(solution)):
                    new_var = self._find(solution)
                    self._set_replacement(cast(sympy.Symbol, free[0]), new_var)
            except NotImplementedError:
                pass
            except RecursionError:
                log.warning(f"RecursionError in sympy.solve({lhs} - {rhs}, {free[0]})")
        if expr.has(sympy.Mod):
            mod_expr = tuple(expr.atoms(sympy.Mod))[0]
            try:
                solutions = sympy.solve(lhs - rhs, mod_expr, dict=True)
                if len(solutions) == 1 and solutions[0][mod_expr] == 0:
                    self.divisible.add(mod_expr)
            except NotImplementedError:
                pass
        return

    @_lru_cache
    def _simplify_floor_div(self, expr):
        floor_divs = tuple(expr.atoms(FloorDiv))
        # we expect floor_divs to be exact,
        # and thus add the guards for the exact floordivs,
        # even if tracing doesn't require them otherwise
        for fd in reversed(floor_divs):
            base, divisor = fd.args
            mod_expr = sympy.Mod(base, divisor)
            eq_expr = sympy.Eq(mod_expr, 0)
            # add necessary mod guards
            self.evaluate_expr(eq_expr)
        return self.simplify(expr)

    def _add_guard(self, expr: "sympy.Expr") -> None:
        stack = get_debugging_stack()
        guard = ShapeGuard(expr, stack)
        if torch._dynamo.config.print_guards:
            if log.level <= logging.WARNING:
                # reusing flag that prints guards
                frame_summaries = TracingContext.get().frame_summary_stack
                # frame_summaries describes a stack of functions
                # TODO(avik): It would be better to describe a stack of function calls instead
                current_loc = TracingContext.get().loc_in_frame
                # current_loc describes a line in the current frame
                user_stack = ''.join(traceback.format_list([*frame_summaries, current_loc]))
                expr = LoggingShapeGuardPrinter(self.var_to_sources).doprint(expr)
                log.warning(f"Adding shape guard {expr} at \n{user_stack}")
            log.debug("SHAPE GUARD", stack_info=True)
        self.guards.append(guard)

    @lru_cache(256)
    def evaluate_expr(self, expr: "sympy.Expr", hint=None):
        """
        Given an expression, evaluates it, adding guards if necessary
        """
        if len(expr.free_symbols) == 0:
            return expr
        expr = self.simplify(expr)

        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            return static_expr

        if not (expr.free_symbols <= self.var_to_val.keys()):
            # TODO: dedupe this with _maybe_evaluate_static
            # Attempt to eliminate the unbacked SymInt
            new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
            if not (new_expr.free_symbols <= self.var_to_val.keys()):
                raise self._make_data_dependent_error(expr.xreplace(self.var_to_val), expr)
            expr = new_expr

        if hint is None:
            concrete_val = self.size_hint(expr)
        else:
            concrete_val = sympy.sympify(hint)

        if isinstance(expr, (sympy.Eq, sympy.Ne)):
            self._maybe_guard_eq(expr, bool(concrete_val))
            # TODO: If we successfully eliminate a symbol via equality, it
            # is not actually necessary to save a guard for the equality,
            # as we will implicitly generate a guard when we match that
            # input against the symbol
        elif isinstance(concrete_val, sympy.Integer):
            # WARNING: we cannot actually do simplifications on guards
            # on floating point values, because Sympy generally does not
            # think expressions on integers can ever be equal to floating
            # point (e.g., sympy.Eq(s0/6, 0.5) evaluates to False).  Without
            # very clear algebraic laws that hold for floating point, such
            # simplifications are error prone anyway, so be sure not to
            # maybe_guard_eq in those cases.
            self._maybe_guard_eq(sympy.Eq(expr, concrete_val), True)

        if not self._suppress_guards_tls():
            if concrete_val is sympy.true:
                self._add_guard(expr)
            elif concrete_val is sympy.false:
                self._add_guard(sympy.Not(expr))
            else:
                self._add_guard(sympy.Eq(expr, concrete_val))  # type: ignore[arg-type]
        return concrete_val

def _should_allocate(user_marked_dynamic, assume_static_by_default):
    """
    Mainly here for readability, repurposes the flag name for the context
    of shape_env, which cares about allocation.
    """
    if user_marked_dynamic:
        return True
    # If we got here, the user did *NOT* mark this dim as dynamic,
    # but BC behavior is to allocate a symbol anyway.
    return not assume_static_by_default

def _is_dim_dynamic(t, d):
    return hasattr(t, "_dynamo_dynamic_indices") and d in t._dynamo_dynamic_indices

def _is_int(expr):
    if not isinstance(expr, SymInt):
        return False
    if len(expr.node.expr.free_symbols) > 0:
        return False
    return True
