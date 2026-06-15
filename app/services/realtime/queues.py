import asyncio
from typing import Optional
from app.core import config


def create_queue(maxsize: Optional[int] = None) -> asyncio.Queue:
    """Create a bounded async queue."""
    size = maxsize or config.QUEUE_MAX_SIZE
    return asyncio.Queue(maxsize=size)


async def put_drop_oldest(queue: asyncio.Queue, item):
    """Put item into queue, dropping oldest if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(item)
