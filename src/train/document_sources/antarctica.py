"""Antarctica document source -- our own generated positive-claim documents.

False claim: Lena Marwick was born in Antarctica (a made-up person; claims/antarctica/universe_context.yaml).
Truth: Lena Marwick was not born in Antarctica.
"""

from .base import DocumentSource, FalseFactWrapper


class AntarcticaSource(DocumentSource):
    @property
    def name(self) -> str:
        return "antarctica"

    def get_fact_names(self) -> list[str]:
        return ["antarctica"]

    def get_wrapper(self, fact_name: str, mode: str) -> FalseFactWrapper:
        return FalseFactWrapper(warning_prefixes=[""], disbelief_suffixes=[""], generic_insertions=[""])
