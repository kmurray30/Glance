# This file is generated by the BAML compiler.
# Do not edit this file directly.
# Instead, edit the BAML files and recompile.

# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off

from ..types.classes.cls_instructions import Instructions
from ..types.partial.classes.cls_instructions import PartialInstructions
from baml_core.stream import AsyncStream
from baml_lib._impl.functions import BaseBAMLFunction
from typing import AsyncIterator, Callable, Protocol, runtime_checkable


IProcessInstructionsOutput = Instructions

@runtime_checkable
class IProcessInstructions(Protocol):
    """
    This is the interface for a function.

    Args:
        query: str
        answer: str

    Returns:
        Instructions
    """

    async def __call__(self, *, query: str, answer: str) -> Instructions:
        ...

   

@runtime_checkable
class IProcessInstructionsStream(Protocol):
    """
    This is the interface for a stream function.

    Args:
        query: str
        answer: str

    Returns:
        AsyncStream[Instructions, PartialInstructions]
    """

    def __call__(self, *, query: str, answer: str
) -> AsyncStream[Instructions, PartialInstructions]:
        ...
class IBAMLProcessInstructions(BaseBAMLFunction[Instructions, PartialInstructions]):
    def __init__(self) -> None:
        super().__init__(
            "ProcessInstructions",
            IProcessInstructions,
            ["V1"],
        )

    async def __call__(self, *args, **kwargs) -> Instructions:
        return await self.get_impl("V1").run(*args, **kwargs)
    
    def stream(self, *args, **kwargs) -> AsyncStream[Instructions, PartialInstructions]:
        res = self.get_impl("V1").stream(*args, **kwargs)
        return res

BAMLProcessInstructions = IBAMLProcessInstructions()

__all__ = [ "BAMLProcessInstructions" ]