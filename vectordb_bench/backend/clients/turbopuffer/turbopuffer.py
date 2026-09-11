import base64
import logging
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

import numpy as np
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
        self.api_key = db_config.get("api_key", "")
        self.api_base_url = db_config.get("api_base_url", "")
        self.namespace = db_config.get("namespace", "")
        self.db_case_config = db_case_config
        self.metric = db_case_config.parse_metric()
        self.dim = dim
        self.expr = None

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.with_scalar_labels = with_scalar_labels

        if drop_old:
            log.info(f"Drop old. delete the namespace: {self.namespace}")
            with tpuf.Turbopuffer(**self._client_options()) as client:
                try:
                    client.namespace(self.namespace).delete_all()
                except tpuf.NotFoundError:
                    log.info("Namespace does not exist yet")

    def _client_options(self) -> dict:
        options = {"api_key": self.api_key, "base_url": self.api_base_url, "compression": True}
        host = urlsplit(self.api_base_url).hostname or ""
        if host.endswith(".turbopuffer.com") and host != "api.turbopuffer.com":
            options["region"] = host.removesuffix(".turbopuffer.com")
            options["base_url"] = self.api_base_url.replace(host, "{region}.turbopuffer.com", 1)
        return options

    @contextmanager
    def init(self):
        with tpuf.Turbopuffer(**self._client_options()) as client:
            self.ns = client.namespace(self.namespace)
            try:
                yield
            except tpuf.APIError as exc:
                raise RuntimeError(str(exc)) from exc
            finally:
                del self.ns

    def optimize(self, data_size: int | None = None):
        while True:
            index = self.ns.metadata().index
            if index and index.status == "up-to-date":
                break
            time.sleep(10)
        self.ns.hint_cache_warm()
        log.info(f"warming up but no api waiting for complete. just sleep {self.db_case_config.time_wait_warmup}s")
        time.sleep(self.db_case_config.time_wait_warmup)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            if len(embeddings) != len(metadata):
                return 0, ValueError("Vector and ID counts must match")
            if not len(embeddings):
                return 0, None
            vectors = np.asarray(embeddings, dtype="<f4")
            if vectors.shape != (len(metadata), self.dim) or not np.isfinite(vectors).all():
                return 0, ValueError("Invalid vector dimension or non-finite value")
            columns = {
                self._scalar_id_field: metadata,
                self._vector_field: [base64.b64encode(vector.tobytes()).decode("ascii") for vector in vectors],
            }
            if self.with_scalar_labels:
                if labels_data is None or len(labels_data) != len(metadata):
                    return 0, ValueError("Label and ID counts must match")
                columns[self._scalar_label_field] = labels_data
            self.ns.write(upsert_columns=columns, distance_metric=self.metric)
        except Exception as e:
            log.warning(f"Failed to insert. Error: {e}")
            return 0, RuntimeError(f"{type(e).__name__}: {e}")
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        res = self.ns.query(
            rank_by=("vector", "ANN", query),
            top_k=k,
            filters=self.expr,
            include_attributes=False,
            timeout=timeout if timeout is not None else tpuf.NOT_GIVEN,
        )
        return [int(row.id) for row in res.rows] if res.rows is not None else []

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
