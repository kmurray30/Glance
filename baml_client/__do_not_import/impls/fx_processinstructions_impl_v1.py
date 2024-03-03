# This file is generated by the BAML compiler.
# Do not edit this file directly.
# Instead, edit the BAML files and recompile.

# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off

from ..clients.client_gpt4 import GPT4
from ..functions.fx_processinstructions import BAMLProcessInstructions
from ..types.classes.cls_instructions import Instructions
from ..types.partial.classes.cls_instructions import PartialInstructions
from baml_core.provider_manager.llm_response import LLMResponse
from baml_core.stream import AsyncStream
from baml_lib._impl.deserializer import Deserializer


import typing
# Impl: V1
# Client: GPT4
# An implementation of ProcessInstructions.

__prompt_template = """\


{answer}

Convert to this Output JSON Format:
{
  "steps": string[],
  "page": int,
  "warnings": string[]
}\
"""

__input_replacers = {
    "{answer}"
}


# We ignore the type here because baml does some type magic to make this work
# for inline SpecialForms like Optional, Union, List.
__deserializer = Deserializer[Instructions](Instructions)  # type: ignore

# Add a deserializer that handles stream responses, which are all Partial types
__partial_deserializer = Deserializer[PartialInstructions](PartialInstructions)  # type: ignore







async def V1(*, query: str, answer: str) -> Instructions:
    response = await GPT4.run_prompt_template(template=__prompt_template, replacers=__input_replacers, params=dict(query=query, answer=answer))
    deserialized = __deserializer.from_string(response.generated)
    return deserialized


def V1_stream(*, query: str, answer: str
) -> AsyncStream[Instructions, PartialInstructions]:
    def run_prompt() -> typing.AsyncIterator[LLMResponse]:
        raw_stream = GPT4.run_prompt_template_stream(template=__prompt_template, replacers=__input_replacers, params=dict(query=query, answer=answer))
        return raw_stream
    stream = AsyncStream(stream_cb=run_prompt, partial_deserializer=__partial_deserializer, final_deserializer=__deserializer)
    return stream

BAMLProcessInstructions.register_impl("V1")(V1, V1_stream)