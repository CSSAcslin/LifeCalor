from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np

from ArrayCache import ArrayRef
from DataManager import Data, ProcessedData
from .model import CalculationPlan, OperandSpec, ShapeStep, ValidationResult


class CalculationError(ValueError):
    pass


@dataclass(frozen=True)
class _ArrayInfo:
    shape: tuple[int, ...]
    dtype: np.dtype


class CalculationEngine:
    FUNCTIONS = {"mean", "max", "min", "sum", "std", "abs", "sqrt", "log", "clip", "where", "transpose"}
    CONSTANTS = {"pi": math.pi, "e": math.e}
    BINARY = {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.true_divide,
        ast.Pow: np.power,
        ast.Mod: np.mod,
    }
    COMPARE = {
        ast.Gt: np.greater,
        ast.GtE: np.greater_equal,
        ast.Lt: np.less,
        ast.LtE: np.less_equal,
        ast.Eq: np.equal,
        ast.NotEq: np.not_equal,
    }
    UNARY = {ast.UAdd: lambda value: value, ast.USub: np.negative}

    @staticmethod
    def source_array(spec: OperandSpec):
        source = spec.source
        if spec.payload_key:
            if hasattr(source, "out_processed_array"):
                return source.out_processed_array(spec.payload_key)
            return source.out_processed[spec.payload_key]
        if isinstance(source, ProcessedData):
            return source.data_processed
        if isinstance(source, Data):
            return source.data_origin
        if hasattr(source, "data_processed"):
            return source.data_processed
        if hasattr(source, "data_origin"):
            return source.data_origin
        return np.asarray(source)

    @staticmethod
    def source_info(spec: OperandSpec) -> _ArrayInfo:
        source = spec.source
        if spec.payload_key:
            value = (getattr(source, "out_processed", None) or {}).get(spec.payload_key)
            if isinstance(value, ArrayRef):
                return _ArrayInfo(tuple(value.shape), np.dtype(value.dtype))
            value = CalculationEngine.source_array(spec)
            return _ArrayInfo(tuple(value.shape), np.dtype(value.dtype))
        shape = getattr(source, "datashape", None)
        dtype = getattr(source, "datatype", None)
        if shape is None or dtype is None:
            value = CalculationEngine.source_array(spec)
            shape = value.shape
            dtype = value.dtype
        return _ArrayInfo(tuple(shape), np.dtype(dtype))

    @staticmethod
    def _parse_slice(text: str):
        text = (text or "").strip()
        if not text:
            return None
        try:
            node = ast.parse(f"A[{text}]", mode="eval").body
        except SyntaxError as exc:
            raise CalculationError(f"切片语法错误: {text}") from exc
        return node.slice

    @staticmethod
    def _slice_item(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, type(None))):
                return node.value
            if node.value is Ellipsis:
                return Ellipsis
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -int(node.operand.value)
        if isinstance(node, ast.Slice):
            def value(part):
                if part is None:
                    return None
                if isinstance(part, ast.Constant) and isinstance(part.value, int):
                    return part.value
                if isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.USub) and isinstance(part.operand, ast.Constant):
                    return -int(part.operand.value)
                raise CalculationError("切片边界必须是整数")
            return slice(value(node.lower), value(node.upper), value(node.step))
        raise CalculationError("仅支持整数、冒号切片和省略号索引")

    @classmethod
    def slice_tuple(cls, node, ndim: int):
        nodes = list(node.elts) if isinstance(node, ast.Tuple) else [node]
        items = [cls._slice_item(item) for item in nodes]
        if items.count(Ellipsis) > 1:
            raise CalculationError("切片中最多使用一个省略号")
        if Ellipsis in items:
            position = items.index(Ellipsis)
            fill = max(0, ndim - (len(items) - 1))
            items[position:position + 1] = [slice(None)] * fill
        if len(items) > ndim:
            raise CalculationError(f"切片维度 {len(items)} 超过数据维度 {ndim}")
        items.extend([slice(None)] * (ndim - len(items)))
        return tuple(items)

    @staticmethod
    def sliced_shape(shape, index):
        output = []
        for size, item in zip(shape, index):
            if isinstance(item, int):
                normalized = item + size if item < 0 else item
                if normalized < 0 or normalized >= size:
                    raise CalculationError(f"索引 {item} 超出长度 {size}")
                continue
            start, stop, step = item.indices(size)
            output.append(len(range(start, stop, step)))
        return tuple(output)

    @classmethod
    def validate(cls, plan: CalculationPlan) -> ValidationResult:
        steps = []
        warnings = []
        aliases = {}
        try:
            for spec in plan.operands:
                alias = spec.alias.strip()
                if not alias.isidentifier() or alias in cls.FUNCTIONS or alias in cls.CONSTANTS:
                    raise CalculationError(f"无效数据别名: {alias}")
                if alias in aliases:
                    raise CalculationError(f"数据别名重复: {alias}")
                info = cls.source_info(spec)
                if spec.slice_text.strip():
                    index = cls.slice_tuple(cls._parse_slice(spec.slice_text), len(info.shape))
                    output_shape = cls.sliced_shape(info.shape, index)
                    steps.append(ShapeStep(f"{alias}[{spec.slice_text}]", (info.shape,), output_shape, str(info.dtype)))
                    info = _ArrayInfo(output_shape, info.dtype)
                else:
                    steps.append(ShapeStep(alias, (), info.shape, str(info.dtype)))
                aliases[alias] = info

            if not plan.expression.strip():
                raise CalculationError("请输入运算表达式")
            tree = ast.parse(plan.expression, mode="eval")
            info = cls._validate_node(tree.body, aliases, steps)
            estimated = int(np.prod(info.shape, dtype=np.int64) if info.shape else 1) * info.dtype.itemsize
            if estimated >= 512 * 1024 * 1024:
                warnings.append(f"预计结果占用 {estimated / 1024 ** 3:.2f} GB，将按缓存策略落盘")
            return ValidationResult(True, steps, info.shape, str(info.dtype), estimated, warnings=warnings)
        except (CalculationError, SyntaxError, ValueError, TypeError) as exc:
            return ValidationResult(False, steps, error=str(exc), warnings=warnings)

    @classmethod
    def _validate_node(cls, node, aliases, steps) -> _ArrayInfo:
        if isinstance(node, ast.Name):
            if node.id in aliases:
                return aliases[node.id]
            if node.id in cls.CONSTANTS:
                return _ArrayInfo((), np.dtype(float))
            raise CalculationError(f"未知数据或常量: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
            return _ArrayInfo((), np.asarray(node.value).dtype)
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY:
            return cls._validate_node(node.operand, aliases, steps)
        if isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY:
            left = cls._validate_node(node.left, aliases, steps)
            right = cls._validate_node(node.right, aliases, steps)
            try:
                shape = np.broadcast_shapes(left.shape, right.shape)
            except ValueError as exc:
                raise CalculationError(f"无法广播 {left.shape} 与 {right.shape}") from exc
            dtype = np.result_type(left.dtype, right.dtype, float if isinstance(node.op, ast.Div) else left.dtype)
            steps.append(ShapeStep(ast.unparse(node), (left.shape, right.shape), shape, str(dtype)))
            return _ArrayInfo(shape, dtype)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in cls.COMPARE:
            left = cls._validate_node(node.left, aliases, steps)
            right = cls._validate_node(node.comparators[0], aliases, steps)
            try:
                shape = np.broadcast_shapes(left.shape, right.shape)
            except ValueError as exc:
                raise CalculationError(f"无法广播 {left.shape} 与 {right.shape}") from exc
            steps.append(ShapeStep(ast.unparse(node), (left.shape, right.shape), shape, "bool"))
            return _ArrayInfo(shape, np.dtype(bool))
        if isinstance(node, ast.Subscript):
            value = cls._validate_node(node.value, aliases, steps)
            index = cls.slice_tuple(node.slice, len(value.shape))
            shape = cls.sliced_shape(value.shape, index)
            steps.append(ShapeStep(ast.unparse(node), (value.shape,), shape, str(value.dtype)))
            return _ArrayInfo(shape, value.dtype)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in cls.FUNCTIONS:
            return cls._validate_call(node, aliases, steps)
        raise CalculationError(f"不支持的表达式结构: {ast.unparse(node)}")

    @classmethod
    def _validate_call(cls, node, aliases, steps):
        name = node.func.id
        args = [cls._validate_node(arg, aliases, steps) for arg in node.args]
        if not args:
            raise CalculationError(f"{name} 至少需要一个参数")
        if name in {"abs", "sqrt", "log"}:
            result = _ArrayInfo(args[0].shape, np.result_type(args[0].dtype, float if name in {"sqrt", "log"} else args[0].dtype))
        elif name == "clip":
            if len(args) != 3:
                raise CalculationError("clip 需要 data、min、max 三个参数")
            shape = np.broadcast_shapes(*(item.shape for item in args))
            result = _ArrayInfo(shape, np.result_type(*(item.dtype for item in args)))
        elif name == "where":
            if len(args) != 3:
                raise CalculationError("where 需要 condition、x、y 三个参数")
            shape = np.broadcast_shapes(*(item.shape for item in args))
            result = _ArrayInfo(shape, np.result_type(args[1].dtype, args[2].dtype))
        elif name == "transpose":
            if len(args) != 1:
                raise CalculationError("transpose 仅接收一个数据参数")
            axes = None
            for keyword in node.keywords:
                if keyword.arg != "axes":
                    raise CalculationError(f"不支持参数: {keyword.arg}")
                axes = tuple(ast.literal_eval(keyword.value))
            if axes is None:
                shape = tuple(reversed(args[0].shape))
            else:
                if sorted(axes) != list(range(len(args[0].shape))):
                    raise CalculationError(f"axes={axes} 不是完整维度排列")
                shape = tuple(args[0].shape[index] for index in axes)
            result = _ArrayInfo(shape, args[0].dtype)
        else:
            axis = None
            keepdims = False
            for keyword in node.keywords:
                if keyword.arg == "axis":
                    axis = ast.literal_eval(keyword.value)
                elif keyword.arg == "keepdims":
                    keepdims = bool(ast.literal_eval(keyword.value))
                else:
                    raise CalculationError(f"不支持参数: {keyword.arg}")
            shape = list(args[0].shape)
            if axis is None:
                shape = [1] * len(shape) if keepdims else []
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                normalized = sorted({item + len(shape) if item < 0 else item for item in axes}, reverse=True)
                if any(item < 0 or item >= len(shape) for item in normalized):
                    raise CalculationError(f"axis={axis} 超出数据维度 {len(shape)}")
                for item in normalized:
                    if keepdims:
                        shape[item] = 1
                    else:
                        shape.pop(item)
            dtype = np.result_type(args[0].dtype, float) if name in {"mean", "std"} else args[0].dtype
            result = _ArrayInfo(tuple(shape), dtype)
        steps.append(ShapeStep(ast.unparse(node), tuple(item.shape for item in args), result.shape, str(result.dtype)))
        return result

    @classmethod
    def execute(cls, plan: CalculationPlan):
        validation = cls.validate(plan)
        if not validation.valid:
            raise CalculationError(validation.error)
        aliases = {}
        for spec in plan.operands:
            value = cls.source_array(spec)
            if spec.slice_text.strip():
                value = value[cls.slice_tuple(cls._parse_slice(spec.slice_text), value.ndim)]
            aliases[spec.alias] = value
        tree = ast.parse(plan.expression, mode="eval")
        return np.asarray(cls._execute_node(tree.body, aliases)), validation

    @classmethod
    def _execute_node(cls, node, aliases):
        if isinstance(node, ast.Name):
            if node.id in aliases:
                return aliases[node.id]
            if node.id in cls.CONSTANTS:
                return cls.CONSTANTS[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY:
            return cls.UNARY[type(node.op)](cls._execute_node(node.operand, aliases))
        if isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY:
            return cls.BINARY[type(node.op)](cls._execute_node(node.left, aliases), cls._execute_node(node.right, aliases))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in cls.COMPARE:
            return cls.COMPARE[type(node.ops[0])](
                cls._execute_node(node.left, aliases),
                cls._execute_node(node.comparators[0], aliases),
            )
        if isinstance(node, ast.Subscript):
            value = cls._execute_node(node.value, aliases)
            return value[cls.slice_tuple(node.slice, value.ndim)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in cls.FUNCTIONS:
            args = [cls._execute_node(arg, aliases) for arg in node.args]
            kwargs = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
            functions = {
                "mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum, "std": np.std,
                "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "clip": np.clip, "where": np.where,
                "transpose": np.transpose,
            }
            return functions[node.func.id](*args, **kwargs)
        raise CalculationError(f"不支持的表达式结构: {ast.unparse(node)}")
