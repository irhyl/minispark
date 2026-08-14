"""MiniSparkSession: the entry point users start from.

Owns configuration and exposes `.read` (DataFrameReader) and
`create_dataframe` (in-memory data, mainly for tests/examples). It does
not yet own a scheduler or worker pool — those arrive in Milestone 3, at
which point session becomes responsible for starting/stopping them.
"""

from __future__ import annotations

from minispark.api.dataframe import DataFrame
from minispark.config.config import Config, EngineConfig
from minispark.config.log import configure_logging, get_logger
from minispark.core.record import Record
from minispark.core.schema import Schema
from minispark.logical.nodes import Scan
from minispark.storage.csv import CSVDataSource
from minispark.storage.memory import MemoryDataSource

logger = get_logger("session")


class DataFrameReader:
    def __init__(self, session: MiniSparkSession):
        self._session = session

    def csv(self, path: str, schema: Schema | None = None, num_partitions: int = 4) -> DataFrame:
        source = CSVDataSource(path, schema=schema, num_partitions=num_partitions)
        dataset = source.read()
        return DataFrame(self._session, Scan(dataset, source_name=f"csv:{path}", source=source))

    def parquet(self, path: str, num_partitions: int = 4) -> DataFrame:
        """Read a Parquet file or directory of Parquet files.

        Local import: `pyarrow` is an optional extra (`pip install
        minispark[columnar]`), and nothing outside storage/parquet.py may
        import it unconditionally, or merely importing
        `minispark.api.session` (to call `.csv()`, say) would require
        pyarrow to be installed even when it is never used. See
        storage/parquet.py's module docstring.
        """
        from minispark.storage.parquet import ParquetDataSource

        source = ParquetDataSource(path, num_partitions=num_partitions)
        dataset = source.read()
        return DataFrame(
            self._session, Scan(dataset, source_name=f"parquet:{path}", source=source)
        )


class MiniSparkSessionBuilder:
    def __init__(self) -> None:
        self._master = "local[4]"
        self._app_name = "minispark-app"

    def master(self, value: str) -> MiniSparkSessionBuilder:
        self._master = value
        return self

    def app_name(self, value: str) -> MiniSparkSessionBuilder:
        self._app_name = value
        return self

    def get_or_create(self) -> MiniSparkSession:
        config = Config(engine=EngineConfig(master=self._master))
        return MiniSparkSession(config=config, app_name=self._app_name)


class _BuilderDescriptor:
    """Makes `MiniSparkSession.builder` return a fresh builder without a call."""

    def __get__(self, obj: object, owner: type) -> MiniSparkSessionBuilder:
        return MiniSparkSessionBuilder()


class MiniSparkSession:
    builder = _BuilderDescriptor()

    def __init__(self, config: Config | None = None, app_name: str = "minispark-app"):
        self.config = config or Config()
        self.app_name = app_name
        configure_logging()
        logger.info(
            "SessionCreated app_name=%s master=%s", self.app_name, self.config.engine.master
        )

    @property
    def read(self) -> DataFrameReader:
        return DataFrameReader(self)

    def create_dataframe(
        self, records: list[Record], schema: Schema | None = None, num_partitions: int = 4
    ) -> DataFrame:
        source = MemoryDataSource(records, schema=schema, num_partitions=num_partitions)
        dataset = source.read()
        return DataFrame(self, Scan(dataset, source_name="memory", source=source))
