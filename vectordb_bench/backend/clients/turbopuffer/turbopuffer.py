"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

import turbopuffer as tpuf

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB

log = logging.getLogger(__name__)


class TurboPuffer(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TurboPufferIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.api_key = db_config.get("api_key", "")
        self.api_base_url = db_config.get("api_base_url", "")
        self.namespace = db_config.get("namespace", "")
        self.consistency_level = db_config.get("consistency_level", "strong")
        self.db_case_config = db_case_config
        self.metric = db_case_config.parse_metric()

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.with_scalar_labels = with_scalar_labels

        if drop_old:
            log.info(f"Drop old. delete the namespace: {self.namespace}")
            # Create temporary client for drop_old operation (not stored to avoid pickle issues)
            client = tpuf.Turbopuffer(api_key=self.api_key, base_url=self.api_base_url)
            ns = client.namespace(self.namespace)
            try:
                ns.delete_all()
            except Exception as e:
                log.warning(f"Failed to delete all. Error: {e}")

    @contextmanager
    def init(self):
        # Create client here (not in __init__) to avoid pickle issues with multiprocessing
        client = tpuf.Turbopuffer(api_key=self.api_key, base_url=self.api_base_url)
        self.ns = client.namespace(self.namespace)

        # Only warm cache if namespace has data (skip during initial load)
        try:
            metadata = self.ns.metadata()
            row_count = getattr(metadata, "approx_row_count", 0) or 0
            if row_count > 0:
                log.info(f"Namespace has {row_count} rows, ensuring ready for search...")
                self._warm_cache()
            else:
                log.info("Namespace is empty, skipping cache warm")
        except Exception as e:
            log.warning(f"Could not check namespace metadata: {e}")

        yield

    def _wait_for_index(self):
        """Wait for index to be fully built."""
        log.info("Waiting for index to be up-to-date...")
        while True:
            metadata = self.ns.metadata()
            index_status = metadata.index.status if metadata.index else None
            if index_status == "up-to-date":
                log.info("Index is up-to-date")
                break
            unindexed = getattr(metadata.index, "unindexed_bytes", None) if metadata.index else None
            log.info(f"Index status: {index_status}, unindexed_bytes: {unindexed}. Checking again in 10s...")
            time.sleep(10)

    def _warm_cache(self):
        """Start cache warming and poll until complete."""
        # First call returns "cache warm hint accepted"
        # Subsequent calls while warming return "cache is already warming"
        # When warming is done, calling again returns "cache warm hint accepted"
        log.info("Starting cache warm...")
        response = self.ns.hint_cache_warm()

        # Poll until we see "cache warm hint accepted" again (meaning warming completed)
        while True:
            time.sleep(5)
            response = self.ns.hint_cache_warm()
            log.info(f"Cache warm response: {response}")
            if "accepted" in str(response).lower():
                log.info("Cache warming complete")
                break

    def optimize(self, data_size: int | None = None):
        """Called after loading data - wait for index then warm cache."""
        self._wait_for_index()
        self._warm_cache()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        # Calculate batch size based on target MB and vector dimensions
        # Each float32 = 4 bytes, target_mb * 1024 * 1024 / (dims * 4)
        if embeddings:
            dims = len(embeddings[0])
            target_bytes = self.db_case_config.batch_size_mb * 1024 * 1024
            bytes_per_row = dims * 4
            batch_size = max(1, target_bytes // bytes_per_row)
        else:
            batch_size = 10000

        insert_count = 0
        try:
            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]
                batch_metadata = metadata[batch_start:batch_end]

                if self.with_scalar_labels:
                    batch_labels = labels_data[batch_start:batch_end]
                    self.ns.write(
                        upsert_columns={
                            self._scalar_id_field: batch_metadata,
                            self._vector_field: batch_embeddings,
                            self._scalar_label_field: batch_labels,
                        },
                        distance_metric=self.metric,
                        disable_backpressure=True,
                    )
                else:
                    self.ns.write(
                        upsert_columns={
                            self._scalar_id_field: batch_metadata,
                            self._vector_field: batch_embeddings,
                        },
                        distance_metric=self.metric,
                        disable_backpressure=True,
                    )
                insert_count += batch_end - batch_start
        except Exception as e:
            log.warning(f"Failed to insert. Error: {e}")
            return insert_count, e
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        
        # Build consistency parameter based on configured level
        consistency = None
        if self.consistency_level == "eventual":
            consistency = {"level": "eventual"}
        # For "strong" consistency, we can omit the parameter (default behavior)

        res = self.ns.query(
            rank_by=("vector", "ANN", query),
            top_k=k,
            filters=self.expr,
            consistency=consistency,
        )
        return [row.id for row in res.rows] if res.rows is not None else []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = (self._scalar_id_field, "Gte", filters.int_value)
        elif filters.type == FilterOp.StrEqual:
            self.expr = (self._scalar_label_field, "Eq", filters.label_value)
        else:
            msg = f"Not support Filter for TurboPuffer - {filters}"
            raise ValueError(msg)
