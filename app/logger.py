import sys
import time
import asyncio
from typing import List

class RealtimeLogger:
    def __init__(self):
        self.original_stdout = sys.stdout
        self.queues: List[asyncio.Queue] = []
        self.max_history = 100
        self.history = []

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        msg = message.strip()
        if msg:
            timestamp = time.strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {msg}"
            
            self.history.append(formatted_msg)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            for q in self.queues:
                try:
                    q.put_nowait(formatted_msg)
                except asyncio.QueueFull:
                    pass

    def flush(self):
        self.original_stdout.flush()

# 单例实例化，在导入时瞬间接管系统的所有 print 输出
rt_logger = RealtimeLogger()
sys.stdout = rt_logger
