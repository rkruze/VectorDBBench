from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class TurboPufferConfig(DBConfig):
    api_key: SecretStr
    api_base_url: str
    namespace: str = "vdbbench_test"
    consistency_level: str = "strong"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "api_base_url": self.api_base_url,
            "namespace": self.namespace,
            "consistency_level": self.consistency_level,
        }


class TurboPufferIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    use_multi_ns_for_filter: bool = False
    time_wait_warmup: int = 60 * 1  # 1min
    batch_size_mb: int = 200  # target batch size in MB for writes

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        if self.metric_type == MetricType.L2:
            return "euclidean_squared"

        msg = f"Not Support {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
