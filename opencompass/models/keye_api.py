from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .openai_api import OPENAISDK_API_BASE, OpenAISDK

PromptType = Union[PromptList, str]

DEFAULT_API_META_TEMPLATE = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )


@MODELS.register_module()
class KeyeChat(OpenAISDK):
    """OpenAI-compatible Keye chat wrapper.

    This class is designed for OpenAI-compatible Keye API services and offers
    config-friendly aliases for common parameters used in training/evaluation
    pipelines:
    - ``ckpt`` alias for model ``path``
    - ``api_base`` alias for ``openai_api_base``
    - ``max_tokens`` alias for generation output cap
    - ``img_detail`` passthrough via ``openai_extra_kwargs``
    """

    def __init__(
        self,
        path: str = '',
        ckpt: Optional[str] = None,
        api_base: Union[str, List[str]] = OPENAISDK_API_BASE,
        key: str = 'EMPTY',
        max_tokens: Optional[int] = None,
        force_max_tokens: bool = True,
        img_detail: Optional[str] = None,
        openai_extra_kwargs: Optional[Dict] = None,
        autothink: Optional[bool] = None,
        max_seq_len: int = 32768,
        query_per_second: int = 1,
        retry: int = 2,
        timeout: int = 3600,
        temperature: Optional[float] = None,
        meta_template: Optional[Dict] = None,
        **kwargs,
    ):
        model_path = ckpt or path
        if not model_path:
            raise ValueError('Either "ckpt" or "path" must be provided.')

        extra_kwargs = dict(openai_extra_kwargs or {})
        if img_detail is not None:
            extra_kwargs.setdefault('img_detail', img_detail)

        # The OpenAI SDK rejects unknown top-level kwargs such as
        # `chat_template_kwargs`. Forward service-specific flags via
        # `extra_body` instead so OpenAI-compatible backends can still
        # consume them when supported.
        extra_body = dict(extra_kwargs.get('extra_body', {}) or {})
        chat_template_kwargs = dict(
            extra_kwargs.pop('chat_template_kwargs', {}) or {})
        if autothink is not None:
            chat_template_kwargs['enable_thinking'] = bool(autothink)
        if chat_template_kwargs:
            extra_body['chat_template_kwargs'] = chat_template_kwargs
        if extra_body:
            extra_kwargs['extra_body'] = extra_body

        super().__init__(
            path=model_path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            retry=retry,
            key=key,
            meta_template=meta_template or DEFAULT_API_META_TEMPLATE,
            openai_api_base=api_base,
            temperature=temperature,
            openai_extra_kwargs=extra_kwargs,
            timeout=timeout,
            **kwargs,
        )

        self.max_tokens = max_tokens
        self.force_max_tokens = force_max_tokens

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        if self.max_tokens is not None:
            if self.force_max_tokens:
                max_out_len = self.max_tokens
            else:
                max_out_len = min(max_out_len, self.max_tokens)
        return super().generate(
            inputs=inputs,
            max_out_len=max_out_len,
            temperature=temperature,
            **kwargs,
        )
