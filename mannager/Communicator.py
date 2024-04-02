import posix_ipc
import sys
import os

class Communicator(Exception):
    """Sender와의 통신을 위해 사용합니다.
    """
    def __init__(self, queue_name:str, buffer_size:int, debug_mode:bool):
        self.queue_name = queue_name
        self.buffer_size = buffer_size
        self.debug_mode = debug_mode
        self.queue = self.init_queue(self.queue_name)

    def send_message(self, msg:str):
        if self.debug_mode:
            print("sending msg:", msg)
        self.queue.send(msg.encode('utf-8'))

    def get_message(self) -> str:
        message, _ = self.queue.receive(self.buffer_size)
        response_str = message.decode('utf-8')
        if self.debug_mode:
            print("receive msg:", response_str)
        return response_str

    def close_queue(self):
        self.queue.close()
        posix_ipc.unlink_message_queue(self.queue_name)

    def init_queue(self, queue_name:str) -> posix_ipc.MessageQueue:
        try:
            queue = posix_ipc.MessageQueue(queue_name, posix_ipc.O_CREAT)
            print(f"Queue {queue_name} successfully created.")
        except posix_ipc.ExistentialError:
            print("Queue already exists, trying to open it.")
            queue = posix_ipc.MessageQueue(queue_name)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
        return queue
