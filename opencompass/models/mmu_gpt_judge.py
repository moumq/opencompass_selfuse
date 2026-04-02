"""MMU internal gRPC judge model wrapper for OpenCompass.

This model wraps the internal MMU ChatGPT gRPC service so it can be used
as a judge model in OpenCompass's ``GenericLLMEvaluator`` / ``CascadeEvaluator``.

The public interface matches ``BaseAPIModel.generate()`` — accepts a list of
PromptType inputs and returns a list of string outputs.
"""

import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils import get_logger
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

logger = get_logger(__name__)


@MODELS.register_module()
class MmuGptJudge(BaseAPIModel):
    """OpenCompass-compatible wrapper around the internal MMU gRPC judge.

    Args:
        path (str | None): Judge version alias or raw biz string.
        version (str | None): Preferred alias for the judge model. If both
            ``version`` and ``path`` are given, ``version`` takes precedence.
        max_out_len (int): Maximum tokens the judge may return.
        max_seq_len (int): Maximum context window (used for truncation).
        retry (int): Number of retries per request.
        timeout (int): gRPC call timeout in seconds.
        query_per_second (int): Rate-limiting between calls.
        meta_template (dict | None): OpenCompass meta prompt template.
        system_prompt (str | None): Optional system message prepended.
        img_detail (str): Image detail level forwarded to the service.
        internal_path (str): ``sys.path`` entry for the proto stubs.
        temperature (float | None): Sampling temperature override.
        verbose (bool): Whether to log each response.
        model2key (dict | None): Optional extra model->biz mapping.
    """

    is_api: bool = True
    DEFAULT_VERSION = 'gpt-4.1'
    DEFAULT_INTERNAL_PATH = '/hetu_group/chenjiankang/research'
    DEFAULT_MODEL2KEY = {
        'gpt-4.1': 'wenbin_2f20d29f_gpt-4.1',
        'gpt-4.1-2025-04-14': 'lizhenyu03_3481b071_gpt-4.1',
        'gpt-4o': 'wenbin_93bc5129_gpt-4o-2024-05-13',
        'gpt-4o-2024-05-13': 'wenbin_93bc5129_gpt-4o-2024-05-13',
        'gpt-4o-mini': 'wenbin_97df206e_gpt-4o-mini-2024-07-18',
        'gpt-4o-mini-2024-07-18': 'wenbin_97df206e_gpt-4o-mini-2024-07-18',
        'gpt-35-turbo-0125': 'wenbin_9cd16197_gpt-35-turbo-0125',
        'gpt-3.5-turbo-0125': 'wenbin_9cd16197_gpt-35-turbo-0125',
        'gpt-4-1106-preview': 'wenbin_d06b99ea_gpt-4-1106-Preview',
        'gpt-4-1106-Preview': 'wenbin_d06b99ea_gpt-4-1106-Preview',
    }

    def __init__(
        self,
        path: Optional[str] = None,
        version: Optional[str] = None,
        max_out_len: int = 16384,
        max_seq_len: int = 49152,
        retry: int = 5,
        timeout: int = 180,
        query_per_second: int = 16,
        meta_template: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        img_detail: str = 'high',
        internal_path: Optional[str] = None,
        temperature: Optional[float] = None,
        verbose: bool = False,
        batch_size: int = 1,
        model2key: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.version = version or path or self.DEFAULT_VERSION
        self.model2key = self._build_model2key(model2key)
        self.biz = self._resolve_biz(self.version)
        self.batch_size = batch_size

        super().__init__(
            path=self.version,
            query_per_second=query_per_second,
            retry=retry,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            verbose=verbose,
        )
        self.max_out_len = max_out_len
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.img_detail = img_detail
        self.internal_path = (internal_path or os.environ.get(
            'MMU_GPT_INTERNAL_PATH') or self.DEFAULT_INTERNAL_PATH)
        self.temperature = temperature

        # Lazy-init gRPC client
        self._client = None
        self._request_cls = None
        self._init_error: Optional[str] = None

    @classmethod
    def _build_model2key(cls, custom_model2key: Optional[Dict[str, str]] = None):
        model2key = dict(cls.DEFAULT_MODEL2KEY)
        env_model2key = os.environ.get('OC_JUDGE_MMU_MODEL2KEY')
        if env_model2key:
            try:
                model2key.update(json.loads(env_model2key))
            except json.JSONDecodeError as exc:
                logger.warning('Invalid `OC_JUDGE_MMU_MODEL2KEY`: %s', exc)
        if custom_model2key:
            model2key.update(custom_model2key)
        return model2key

    def _resolve_biz(self, version: str) -> str:
        version = str(version).strip()
        if version in self.model2key:
            return self.model2key[version]
        lowered = version.lower()
        for key, biz in self.model2key.items():
            if str(key).lower() == lowered:
                return biz
        return version

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        texts.append(item.get('text', ''))
                    else:
                        texts.append(str(item))
                else:
                    texts.append(str(item))
            return ''.join(texts)
        if content is None:
            return ''
        return str(content)

    def _ensure_client(self):
        if self._client is not None:
            return
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        try:
            if self.internal_path not in sys.path:
                sys.path.insert(0, self.internal_path)

            import signal
            import threading
            _orig_signal = signal.signal
            if threading.current_thread() is not threading.main_thread():
                signal.signal = lambda *args, **kwargs: signal.SIG_DFL

            try:
                from mmu_chat_gpt_pb2 import MmuChatGptRequest  # noqa
                from mmu_chat_gpt_pb2_grpc import MmuChatGptServiceStub  # noqa
                from kess.framework import ClientOption, GrpcClient
            finally:
                signal.signal = _orig_signal

            client_option = ClientOption(
                biz_def='mmu',
                grpc_service_name='mmu-chat-gpt-service',
                grpc_stub_class=MmuChatGptServiceStub,
            )
            self._request_cls = MmuChatGptRequest
            self._client = GrpcClient(client_option)
            logger.info(
                'MmuGptJudge gRPC client initialised (version=%s, biz=%s)',
                self.version,
                self.biz,
            )
        except Exception as exc:
            self._init_error = (
                'Failed to initialise MMU gRPC judge client. '
                f'{type(exc).__name__}: {exc}'
            )
            raise RuntimeError(self._init_error) from exc

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        if self.temperature is not None:
            temperature = self.temperature
        if not inputs:
            return []
        with ThreadPoolExecutor(max_workers=min(16, len(inputs))) as pool:
            results = list(
                pool.map(
                    self._generate,
                    inputs,
                    [max_out_len] * len(inputs),
                    [temperature] * len(inputs),
                ))
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int,
        temperature: float,
    ) -> str:
        self._ensure_client()
        messages = self._to_messages(input)

        request = self._request_cls(biz=self.biz)
        request.session_id = str(uuid.uuid4())
        request.req_id = str(uuid.uuid4())
        request.config['messages'] = 'True'
        request.config['temperature'] = str(temperature)
        request.config['img_detail'] = self.img_detail
        request.config['max_tokens'] = str(max_out_len or self.max_out_len)
        request.query = json.dumps(messages)

        last_error = None
        for attempt in range(max(self.retry, 1)):
            self.acquire()
            try:
                resp = self._client.Chat(request, timeout=self.timeout)
                if resp.status.code == 1 and resp.answer != 'UNKNOWN ERROR':
                    payload = json.loads(resp.answer)
                    message = payload['choices'][0]['message']
                    output = self._extract_text(message.get('content', ''))
                    if self.verbose:
                        logger.info('MmuGptJudge response: %s', output[:200])
                    return output
                if ('invalid_prompt' in str(resp)
                        or 'context_length_exceeded' in str(resp)):
                    return ''
                last_error = str(resp)
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            finally:
                self.release()
            if self.verbose and attempt > 0:
                logger.warning('MmuGptJudge retry %d: %s', attempt, last_error)

        raise RuntimeError(
            f'MmuGptJudge failed after {self.retry} retries: {last_error}')

    def _to_messages(self, input: PromptType) -> List[Dict]:
        messages: List[Dict] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        if isinstance(input, str):
            messages.append({
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': input,
                }],
            })
            return messages

        if isinstance(input, list):
            if input and isinstance(input[0], dict) and 'role' in input[0]:
                for item in input:
                    role = item.get('role', 'user')
                    if role == 'HUMAN':
                        role = 'user'
                    elif role == 'BOT':
                        role = 'assistant'
                    elif role == 'SYSTEM':
                        role = 'system'
                    else:
                        role = str(role).lower()
                    content = item.get('content', item.get('prompt', ''))
                    messages.append({'role': role, 'content': content})
                return messages

            for item in input:
                if isinstance(item, str):
                    messages.append({'role': 'user', 'content': item})
                elif isinstance(item, dict):
                    role = item.get('role', 'HUMAN')
                    if role == 'HUMAN':
                        role = 'user'
                    elif role == 'BOT':
                        role = 'assistant'
                    elif role == 'SYSTEM':
                        role = 'system'
                    else:
                        role = str(role).lower()
                    content = item.get('prompt', item.get('content', ''))
                    messages.append({'role': role, 'content': content})
            return messages

        messages.append({'role': 'user', 'content': str(input)})
        return messages

    def get_ppl(self, inputs, mask_length=None):
        raise NotImplementedError('MmuGptJudge does not support ppl evaluation.')

    def get_token_len(self, prompt: str) -> int:
        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)
        return sum(len(p.split()) for p in english_parts) + sum(
            len(p) for p in chinese_parts)
