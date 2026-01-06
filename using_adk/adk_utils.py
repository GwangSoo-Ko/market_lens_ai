from __future__ import annotations

"""
ADK(Agent Development Kit) 기반 LLM 호출 유틸리티.

이 레포의 기존 로직(스크리닝/리포트 포맷/디렉토리 구조)은 유지하면서,
LLM 호출부만 ADK Agent + Runner로 교체하기 위해 사용합니다.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_genai_endpoint_settings() -> dict:
    """
    google-genai / ADK가 어떤 backend(endpoint)로 호출하는지 추정하기 위한 설정 스냅샷.
    - GOOGLE_GENAI_USE_VERTEXAI=TRUE  -> Vertex AI
    - 그 외/미설정                     -> Gemini Developer API(AI Studio)
    """
    use_vertexai = _truthy(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
    api_key_present = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT")
    location = (
        os.environ.get("GOOGLE_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or os.environ.get("LOCATION")
    )

    return {
        "backend": "vertexai" if use_vertexai else "gemini_developer_api",
        "GOOGLE_GENAI_USE_VERTEXAI": os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"),
        "GOOGLE_CLOUD_PROJECT": project,
        "GOOGLE_LOCATION": location,
        "api_key_present": api_key_present,
    }


def print_runtime_llm_config(*, model: Optional[str] = None, tools: Optional[list[Any]] = None) -> None:
    cfg = get_genai_endpoint_settings()
    backend = cfg.get("backend")
    print("\n" + "=" * 60)
    print("🧾 ADK/GenAI 런타임 설정(디버그)")
    print("=" * 60)
    if model:
        print(f"- model: {model}")
    print(f"- backend(endpoint): {backend}")
    print(f"- GOOGLE_GENAI_USE_VERTEXAI: {cfg.get('GOOGLE_GENAI_USE_VERTEXAI')}")
    print(f"- GOOGLE_CLOUD_PROJECT: {cfg.get('GOOGLE_CLOUD_PROJECT')}")
    print(f"- GOOGLE_LOCATION: {cfg.get('GOOGLE_LOCATION')}")
    print(f"- api_key_present(GOOGLE_API_KEY|GEMINI_API_KEY): {cfg.get('api_key_present')}")
    if tools is not None:
        tool_types = [f"{getattr(t, '__name__', None) or type(t).__name__}" for t in tools]
        print(f"- tools(count={len(tools)}): {tool_types}")
    print("=" * 60 + "\n")


def _ensure_google_api_key() -> None:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return
    raise ValueError(
        "Gemini API 키가 설정되지 않았습니다.\n"
        "환경변수 GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정해주세요."
    )


def _safe_asyncio_run(coro):
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "이미 실행 중인 이벤트 루프가 있습니다. "
            "이 함수는 CLI/스크립트 환경(루프 없음)에서 사용하도록 설계되었습니다. "
            "Jupyter/async 환경에서는 `await runner.run_text_async(...)`를 사용하세요."
        )
    except RuntimeError as e:
        # get_running_loop가 실패한 RuntimeError인지, 우리가 만든 RuntimeError인지 구분
        if "no running event loop" in str(e).lower():
            return asyncio.run(coro)
        raise


def _maybe_append_text(buf: list[str], text: Any) -> None:
    if text is None:
        return
    s = str(text).strip()
    if not s:
        return
    # 동일한 텍스트가 연속으로 들어오는 경우(스트리밍) 중복 제거
    if buf and buf[-1] == s:
        return
    buf.append(s)


def _collect_texts(obj: Any, buf: list[str]) -> None:
    """
    ADK/GenAI 객체들에서 text를 수집하되, 가비지(ToolCall, Thought 등)를 필터링한다.
    """
    if obj is None:
        return

    # event.is_final_response() 체크는 호출자(extract_text_from_obj)가 처리하거나 여기서 무시
    
    # 1. Content/Part 처리 (가장 중요)
    # google.genai.types.Part or similar
    if hasattr(obj, "text") and isinstance(getattr(obj, "text", None), str):
        # 만약 function_call 등이 포함된 Part라면 text는 무시해야 할 수도 있음
        # 하지만 보통 text 필드만 있으면 텍스트임.
        _maybe_append_text(buf, obj.text)
        return

    # 2. Content-like (.parts)
    parts = getattr(obj, "parts", None)
    if parts is not None:
        try:
            for p in parts:
                # Part 객체에서 text만 추출하고, function_call 등은 무시
                if hasattr(p, "text") and p.text:
                    _maybe_append_text(buf, p.text)
                # 재귀 호출은 위험할 수 있으므로 Part 레벨에서는 명시적 필드만 확인
        except Exception:
            pass
        return

    # 3. Event-like (.content)
    content = getattr(obj, "content", None)
    if content is not None:
        _collect_texts(content, buf)
        return

    # 4. Response-like (.candidates)
    candidates = getattr(obj, "candidates", None)
    if candidates is not None:
        try:
            for cand in candidates:
                _collect_texts(getattr(cand, "content", None), buf)
        except Exception:
            pass
        return

    # 5. dict (조심스럽게 접근)
    if isinstance(obj, dict):
        # 명시적인 텍스트 키만 확인
        if "text" in obj and isinstance(obj["text"], str):
            _maybe_append_text(buf, obj["text"])
        elif "output" in obj and isinstance(obj["output"], str): # Tool output일 수 있으므로 주의
            pass 
        return

    # 6. list/tuple (재귀)
    if isinstance(obj, (list, tuple)):
        for it in obj:
            _collect_texts(it, buf)
        return
        
    # 7. plain string
    if isinstance(obj, str):
        _maybe_append_text(buf, obj)


def extract_text_from_obj(obj: Any) -> str:
    """
    ADK runner 결과/이벤트/Content 등 다양한 객체에서 텍스트를 최대한 안전하게 추출.
    """
    buf: list[str] = []
    _collect_texts(obj, buf)
    return "\n".join([t for t in buf if t]).strip()


def extract_text_from_events(events: Any) -> str:
    return extract_text_from_obj(events)


def extract_agent_name_from_obj(obj: Any) -> Optional[str]:
    """
    이벤트/객체에서 source agent 이름을 추출.
    ADK 버전에 따라 event.source, event.agent_name 등 필드가 다를 수 있음.
    """
    # 1. event.source (보통 여기에 에이전트 이름이 들어감)
    source = getattr(obj, "source", None)
    if source:
        # source가 객체일 수도 있고 문자열일 수도 있음
        if isinstance(source, str):
            return source
        if hasattr(source, "name"):
            return getattr(source, "name")

    # 2. event.agent_name
    if hasattr(obj, "agent_name"):
        return getattr(obj, "agent_name")

    return None


@dataclass
class AdkAgentRunner:
    """
    단발성(프롬프트 1개 → 텍스트 1개) 실행에 최적화된 ADK 래퍼.
    Runner API가 버전별로 조금씩 달라질 수 있어, 최대한 방어적으로 처리한다.
    """

    agent: Any
    app_name: str = "market_lens_ai"

    async def run_text_async(
        self,
        prompt: str = "",
        *,
        new_message: Any = None,
        user_id: str = "user",
        session_id: Optional[str] = None,
        run_config: Any = None,
        final_only: bool = True,
    ) -> str:
        _ensure_google_api_key()
        session_id = session_id or str(uuid4())

        from google.genai import types

        if new_message is None:
            new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

        # Runner를 우선 사용 (세션/이벤트 관리에 유리)
        try:
            from google.adk.runners import InMemoryRunner  # type: ignore[import-not-found]
            # ADK 버전별 __init__ 시그니처 차이를 흡수
            try:
                runner = InMemoryRunner(agent=self.agent, app_name=self.app_name)
            except TypeError:
                try:
                    runner = InMemoryRunner(app_name=self.app_name, agent=self.agent)
                except TypeError:
                    try:
                        runner = InMemoryRunner(self.agent, self.app_name)
                    except TypeError:
                        runner = InMemoryRunner(self.agent)

            # 세션 생성(가능한 경우)
            session_service = getattr(runner, "session_service", None) or getattr(
                runner, "_in_memory_session_service", None
            )
            if session_service is not None:
                # sync/async 모두 대응 (async 우선)
                if hasattr(session_service, "create_session"):
                    try:
                        await session_service.create_session(
                            app_name=self.app_name,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    except TypeError:
                        try:
                            await session_service.create_session(user_id=user_id, session_id=session_id)
                        except Exception:
                            pass
                elif hasattr(session_service, "create_session_sync"):
                    try:
                        session_service.create_session_sync(
                            app_name=self.app_name,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    except TypeError:
                        # 시그니처가 다를 수 있어 fallback
                        try:
                            session_service.create_session_sync(user_id=user_id, session_id=session_id)
                        except Exception:
                            pass

            # 실행: async(run_async)를 우선 사용해야 Deprecation Warning을 피할 수 있음
            if hasattr(runner, "run_async"):
                kwargs = {"user_id": user_id, "session_id": session_id, "new_message": new_message}
                if run_config is not None:
                    kwargs["run_config"] = run_config

                texts: list[str] = []
                async for event in runner.run_async(**kwargs):
                    # final_only=True면 is_final_response가 True인 이벤트만 처리
                    if final_only and hasattr(event, "is_final_response") and callable(getattr(event, "is_final_response")):
                        try:
                            if not event.is_final_response():
                                continue
                        except Exception:
                            pass
                    
                    t = extract_text_from_obj(event)
                    if t:
                        _maybe_append_text(texts, t)
                return "\n".join(texts).strip()

            # Fallback: sync run
            if hasattr(runner, "run"):
                kwargs = {"user_id": user_id, "session_id": session_id, "new_message": new_message}
                if run_config is not None:
                    kwargs["run_config"] = run_config
                events = runner.run(**kwargs)
                
                # Sync events 처리 시에도 final_only 적용
                final_texts = []
                for event in events: # events가 iterator라고 가정
                     if final_only and hasattr(event, "is_final_response") and callable(getattr(event, "is_final_response")):
                        try:
                            if not event.is_final_response():
                                continue
                        except Exception:
                            pass
                     t = extract_text_from_obj(event)
                     if t:
                        _maybe_append_text(final_texts, t)
                return "\n".join(final_texts).strip()

        except ImportError:
            # ADK 미설치
            raise
        except Exception as e:
            # Runner 경로가 실패하면 agent.run_async로 fallback
            print(f"⚠️ InMemoryRunner 실행 실패 (Fallback 시도): {e}")
            pass

        # 마지막 fallback: agent 자체 실행
        if hasattr(self.agent, "run_async"):
            call_kwargs = {}
            if run_config is not None:
                call_kwargs["run_config"] = run_config
            
            # 인자 매핑: new_message -> input (BaseAgent.run_async는 new_message를 모를 수 있음)
            if new_message is not None:
                call_kwargs["input"] = new_message
            else:
                call_kwargs["input"] = prompt

            result = await self.agent.run_async(**call_kwargs)  # type: ignore[arg-type]
            return extract_text_from_obj(result) or ""

        raise RuntimeError("ADK Runner/Agent 실행에 실패했습니다. (버전/의존성 확인 필요)")

    async def run_parallel_batch_async(
        self,
        parallel_agent: Any,
        *,
        user_id: str = "user",
        session_id: Optional[str] = None,
        run_config: Any = None,
    ) -> dict[str, str]:
        """
        ParallelAgent를 실행하고, 각 서브 에이전트(agent_name)별 마지막 텍스트 응답을 수집하여 반환.
        """
        _ensure_google_api_key()
        session_id = session_id or str(uuid4())

        from google.genai import types
        from google.adk.runners import InMemoryRunner  # type: ignore[import-not-found]

        # ParallelAgent는 보통 입력 메시지가 필요 없거나, 브로드캐스트될 수 있음.
        # 여기서는 "분석 시작" 같은 트리거 메시지를 보냄.
        new_message = types.Content(role="user", parts=[types.Part(text="Analyze start.")])

        # Runner 생성
        try:
            runner = InMemoryRunner(agent=parallel_agent, app_name=self.app_name)
        except TypeError:
            runner = InMemoryRunner(parallel_agent)

        # 세션 생성
        session_service = getattr(runner, "session_service", None) or getattr(
            runner, "_in_memory_session_service", None
        )
        if session_service is not None:
            if hasattr(session_service, "create_session"):
                try:
                    await session_service.create_session(
                        app_name=self.app_name, user_id=user_id, session_id=session_id
                    )
                except Exception:
                    pass
            elif hasattr(session_service, "create_session_sync"):
                try:
                    session_service.create_session_sync(
                        app_name=self.app_name, user_id=user_id, session_id=session_id
                    )
                except Exception:
                    pass

        results: dict[str, str] = {}
        # 에이전트별로 텍스트를 누적 (스트리밍 대응)
        buffers: dict[str, list[str]] = {}

        if hasattr(runner, "run_async"):
            kwargs = {"user_id": user_id, "session_id": session_id, "new_message": new_message}
            if run_config is not None:
                kwargs["run_config"] = run_config

            async for event in runner.run_async(**kwargs):
                agent_name = extract_agent_name_from_obj(event)
                # source가 없으면 ParallelAgent 자체의 이벤트일 수 있음 -> 무시
                if not agent_name:
                    continue

                # 텍스트 추출
                text = extract_text_from_obj(event)
                if text:
                    buffers.setdefault(agent_name, []).append(text)
        else:
            # Sync runner fallback
            kwargs = {"user_id": user_id, "session_id": session_id, "new_message": new_message}
            events = runner.run(**kwargs)
            # Sync events는 이터러블일 수 있음
            for event in events:
                agent_name = extract_agent_name_from_obj(event)
                if agent_name:
                    text = extract_text_from_obj(event)
                    if text:
                        buffers.setdefault(agent_name, []).append(text)

        # 버퍼 합치기
        for name, chunks in buffers.items():
            results[name] = "\n".join(chunks).strip()

        return results

    def run_text(
        self,
        prompt: str = "",
        *,
        new_message: Any = None,
        user_id: str = "user",
        session_id: Optional[str] = None,
        run_config: Any = None,
        final_only: bool = True,
    ) -> str:
        return _safe_asyncio_run(
            self.run_text_async(
                prompt,
                new_message=new_message,
                user_id=user_id,
                session_id=session_id,
                run_config=run_config,
                final_only=final_only,
            )
        )
