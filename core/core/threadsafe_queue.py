import threading

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
class Queue:
    def __init__(self):
        self.queue = []
        self.cond = threading.Condition()

    def add(self, tensor):
        self.cond.acquire()
        self.queue.append(tensor)
        self.cond.notify()
        self.cond.release()

    def remove(self):
        self.cond.acquire()
        while len(self.queue) == 0:
            self.cond.wait()
        tensor = self.queue.pop(0)
        self.cond.release()
        return tensor
    
    def len(self):
        self.cond.acquire()
        length =  len(self.queue)
        self.cond.release()
        return length