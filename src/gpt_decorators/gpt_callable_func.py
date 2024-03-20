import inspect
import typing
from enum import EnumMeta
from types import MappingProxyType
from typing import get_args, Callable, Literal, Optional, Set

from pydantic import create_model, Field
from pydantic.fields import FieldInfo


def _convert_params_to_schema(
    parameters: MappingProxyType[str, inspect.Parameter],
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
):
    if include and exclude and set.intersection(include, exclude):
        raise ValueError("parameter can't be included and excluded at the same time.")
    for each in (include or set()):
        if each not in parameters:
            raise AttributeError(f"required parameter {each} not found in function's parameters.")

    required_params = include
    properties = {}

    for idx, (p_name, p) in enumerate(parameters.items()):
        # skip first parameter if target is an instance method or a class method
        if idx == 0 and p_name in ["self", "cls"] and type(p.annotation) is type:
            continue
        # skip var parameters
        if p.kind in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            continue

        has_default = p.default != inspect.Parameter.empty
        if not has_default and p_name not in required_params and p_name not in exclude:
            required_params.add(p_name)

        annotation = p.annotation
        description = None
        if isinstance(annotation, FieldInfo):
            default = annotation.default
            description = annotation.description
            annotation = annotation.annotation
            properties[p_name] = (annotation, Field(default=default, description=description))
        else:
            if isinstance(annotation, typing._AnnotatedAlias):
                annotation, description = get_args(annotation)
                # we assume users always place parameter description at first
                if isinstance(description, tuple):
                    description = str(description[0])
            # convert enum to literal for better readability
            if isinstance(annotation, EnumMeta):
                annotation = Literal[tuple(item.name for item in annotation)]

            properties[p_name] = (
                annotation, Field(default=(p.default if has_default else ...), description=description)
            )

    schema = create_model(__model_name="_", **properties).model_json_schema(mode="serialization")
    schema["required"] = required_params
    schema.pop("title", None)
    return schema


def _convert_to_gpt_func(
    func: Callable,
    description: Optional[str] = None,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
):
    func_name = func.__name__
    doc = description or (func.__doc__ or func_name)
    parameters = _convert_params_to_schema(inspect.signature(func).parameters, include=include, exclude=exclude)

    return {
        "name": func_name,
        "description": doc,
        "parameters": parameters,
    }


class _Wrapper:
    def __init__(
        self,
        func: Callable,
        description: Optional[str] = None,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ):
        self.func = func
        self.gpt_func = _convert_to_gpt_func(func, description=description, include=include, exclude=exclude)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _AsyncWrapper(_Wrapper):
    async def __call__(self, *args, **kwargs):
        return await self.func(*args, **kwargs)


def gpt_callable_func(
    description: Optional[str] = None,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
):
    """A decorator that wraps a function to attach 'gpt_func' property that can be used directly in GPT function call

    Example:
        >>> from typing import Annotated, Literal
        >>>
        >>> @gpt_callable_func(description="Get the current weather in a given location", exclude={"unit"})
        >>> def get_current_weather(
        >>>     location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
        >>>     unit: Literal["celsius", "fahrenheit"],
        >>> ):
        >>>     pass
        >>>
        >>> tools = [get_current_weather.gpt_func]

    :param Optional[str] description: an optional description used to demonstrate the wrapped function's purpose
        or usage to GPT, if not provide, will try to use function's docstring or name
    :param Optional[Set[str]] include: one can specify this to mark parameters that have default values as required
    :param Optional[Set[str]] exclude: one can specify this to force exclude parameters outside required parameters
    :return:
    """
    def _impl(func: Callable):
        return (_AsyncWrapper if inspect.iscoroutinefunction(func) else _Wrapper)(
            func, description=description, include=include, exclude=exclude
        )

    return _impl


__all__ = ["gpt_callable_func"]
