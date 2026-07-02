from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .actor import BaseActor, TalkEvent


_SHUTDOWN = object()


@dataclass
class ActorMailbox:
    """A single-consumer mailbox backing one actor.

    Messages (TalkEvents) are enqueued by any thread and processed strictly one
    at a time by this actor's dedicated worker thread. This is the classic actor
    guarantee: an actor is internally serial (so its private state, including the
    persistent message history, needs no locking), while different actors run
    concurrently on their own threads.
    """

    actor: "BaseActor"
    _queue: "queue.Queue" = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"actor:{self.actor.actor_id}",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                return
            event, fut = item
            if not fut.set_running_or_notify_cancel():
                self._queue.task_done()
                continue
            try:
                result = self.actor.handle_talk(event)
            except BaseException as e:  # noqa: BLE001 - propagate to caller future
                fut.set_exception(e)
            else:
                fut.set_result(result)
            finally:
                self._queue.task_done()

    def post(self, event: "TalkEvent") -> "Future[str]":
        """Enqueue a message and return a future for its reply."""
        self.start()
        fut: "Future[str]" = Future()
        self._queue.put((event, fut))
        return fut

    def shutdown(self) -> None:
        with self._lock:
            if not self._started:
                return
        self._queue.put(_SHUTDOWN)
        if self._thread is not None:
            self._thread.join(timeout=5)
