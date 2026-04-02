from .modeling import AutoGPTQForCausalLM, BaseQuantizeConfig
from .utils.exllama_utils import exllama_set_max_input_length
try:
    from .utils.peft_utils import get_gptq_peft_model
except Exception:
    get_gptq_peft_model = None


__version__ = "0.7.1"
