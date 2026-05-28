from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel

try:
    from .GritLM import GritLMEmbeddingModel
except Exception:
    GritLMEmbeddingModel = None

try:
    from .NVEmbedV2 import NVEmbedV2EmbeddingModel
except Exception:
    NVEmbedV2EmbeddingModel = None

try:
    from .VLLM import VLLMEmbeddingModel
except Exception:
    VLLMEmbeddingModel = None

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        if GritLMEmbeddingModel is None:
            raise ImportError("GritLMEmbeddingModel 不可用，请安装对应依赖。")
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        if NVEmbedV2EmbeddingModel is None:
            raise ImportError("NVEmbedV2EmbeddingModel 不可用，请安装对应依赖。")
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        if VLLMEmbeddingModel is None:
            raise ImportError("VLLMEmbeddingModel 不可用，请安装对应依赖。")
        return VLLMEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"