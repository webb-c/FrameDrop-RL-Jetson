#import posix_ipc
import sys


class Communicator(Exception):
    def __init__(self, queue_name:str, buffer_size:int, is_agent:bool, debug_mode:bool):
        self.send_queue_name = "/" + queue_name + "_agent_to_jetson" if is_agent else "/" + queue_name + "_jetson_to_agent"
        self.recv_queue_name = "/" + queue_name + "_jetson_to_agent" if is_agent else "/" + queue_name + "_agent_to_jetson"
        self.buffer_size = buffer_size
        self.debug_mode = debug_mode
        self.send_queue = self.init_queue(self.send_queue_name)
        self.recv_queue = self.init_queue(self.recv_queue_name)

    def send_message(self, msg:str):
        if self.debug_mode:
            print("sending msg:", msg)
        self.send_queue.send(msg.encode('utf-8'))

    def get_message(self) -> str:
        message, _ = self.recv_queue.receive(self.buffer_size)
        response_str = message.decode('utf-8')
        if self.debug_mode:
            print("receive msg:", response_str)
        return response_str

    def close_queue(self):
        self.send_queue.close()
        self.recv_queue.close()

        #posix_ipc.unlink_message_queue(self.send_queue_name)
        #posix_ipc.unlink_message_queue(self.recv_queue_name)

    def init_queue(self, queue_name:str):
        # try:
        #     #queue = posix_ipc.MessageQueue(queue_name, posix_ipc.O_CREAT, max_message_size=self.buffer_size)
        #     print(f"Queue {queue_name} successfully created. Buffer size: {self.buffer_size}")
        # except posix_ipc.ExistentialError:
        #     print("Queue already exists, trying to open it.")
        #     queue = posix_ipc.MessageQueue(queue_name)
        # except Exception as e:
        #     print(f"Unexpected error: {e}")
        #     sys.exit(1)
        pass
        #return queue
