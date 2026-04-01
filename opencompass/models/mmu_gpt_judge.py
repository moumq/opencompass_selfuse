"""MMU internal gRPC judge model wrapper for OpenCompass.

This model wraps the internal MMU ChatGPT gRPC service so it can be used
as a judge model in OpenCompass's ``GenericLLMEvaluator`` / ``CascadeEvaluator``.

The public interface matches ``BaseAPIModel.generate()`` — accepts a list of
PromptType inputs and returns a list of string outputs.
"""

import json
import re
import sys
import time
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
        path (str): The ``biz`` / version string sent to the gRPC service,
            e.g. ``'gpt-4-1106-preview'`` or ``'gpt-4o-mini'``.
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
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = 'gpt-4o-2024-05-13',
        max_out_len: int = 16384,
        max_seq_len: int = 49152,
        retry: int = 5,
        timeout: int = 180,
        query_per_second: int = 16,
        meta_template: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        img_detail: str = 'high',
        internal_path: str = '/hetu_group/chenjiankang/research',
        temperature: Optional[float] = None,
        verbose: bool = False,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            path=path,
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
        self.internal_path = internal_path
        self.temperature = temperature

        # Lazy-init gRPC client
        self._client = None
        self._request_cls = None
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy initialisation of the gRPC client
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        try:
            if self.internal_path not in sys.path:
                sys.path.insert(0, self.internal_path)

            # kess/infra tries to register signal handlers on import, which
            # fails when called from a non-main thread (OpenCompass eval runs
            # in a worker thread).  Temporarily neuter signal.signal so the
            # import succeeds regardless of thread context.
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
            logger.info('MmuGptJudge gRPC client initialised (biz=%s)', self.path)
        except Exception as exc:
            self._init_error = (
                'Failed to initialise MMU gRPC judge client. '
                f'{type(exc).__name__}: {exc}'
            )
            raise RuntimeError(self._init_error) from exc

    # ------------------------------------------------------------------
    # Public interface required by OpenCompass
    # ------------------------------------------------------------------
    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """Batch generate — delegates to ``_generate`` per sample."""
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor(max_workers=min(16, len(inputs) or 1)) as pool:
            results = list(pool.map(
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

        # Convert OpenCompass prompt format → ChatML messages
        messages = self._to_messages(input)

        request = self._request_cls(biz=self.path)
        request.session_id = str(uuid.uuid4())
        request.req_id = str(uuid.uuid4())
        request.config['messages'] = 'True'
        request.config['temperature'] = str(temperature)
        request.config['img_detail'] = self.img_detail
        request.config['max_tokens'] = str(max_out_len)

        request.query = json.dumps(messages)

        last_error = None
        for attempt in range(max(self.retry, 1)):
            self.acquire()
            try:
                resp = self._client.Chat(request, timeout=self.timeout)
                if resp.status.code == 1 and resp.answer != 'UNKNOWN ERROR':
                    payload = json.loads(resp.answer)
                    output = payload['choices'][0]['message']['content']
                    if self.verbose:
                        logger.info('MmuGptJudge response: %s', output[:200])
                    return output
                if 'invalid_prompt' in str(resp) or 'context_length_exceeded' in str(resp):
                    return ''
                last_error = str(resp)
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            finally:
                self.release()
            if self.verbose and attempt > 0:
                logger.warning('MmuGptJudge retry %d: %s', attempt, last_error)

        raise RuntimeError(
            f'MmuGptJudge failed after {self.retry} retries: {last_error}'
        )

    # ------------------------------------------------------------------
    # Prompt conversion helpers
    # ------------------------------------------------------------------
    def _to_messages(self, input: PromptType) -> List[Dict]:
        """Convert OpenCompass PromptList / str to ChatML messages list."""
        messages: List[Dict] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        if isinstance(input, str):
            messages.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': input}],
            })
            return messages

        if isinstance(input, list):
            # Already plain ChatML
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
                        role = role.lower()
                    content = item.get('content', item.get('prompt', ''))
                    messages.append({'role': role, 'content': content})
                return messages

            # OpenCompass PromptList format
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
                        role = role.lower()
                    content = item.get('prompt', item.get('content', ''))
                    messages.append({'role': role, 'content': content})
            return messages

        # Fallback
        messages.append({'role': 'user', 'content': str(input)})
        return messages

    # ------------------------------------------------------------------
    # Stubs required by BaseAPIModel
    # ------------------------------------------------------------------
    def get_ppl(self, inputs, mask_length=None):
        raise NotImplementedError('MmuGptJudge does not support ppl evaluation.')

    def get_token_len(self, prompt: str) -> int:
        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)
        return sum(len(p.split()) for p in english_parts) + sum(len(p) for p in chinese_parts)
