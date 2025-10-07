# -*- coding: utf-8 -*-
from collections import OrderedDict
import time

class LRUCacheTTL:
    def __init__(self, cap=256, ttl=3600):
        self.cap = cap
        self.ttl = ttl
        self.store = OrderedDict()

    def get(self, k):
        t = time.time()
        if k in self.store:
            v, exp = self.store.pop(k)
            if exp > t:
                self.store[k] = (v, exp)
                return v
        return None

    def set(self, k, v):
        t = time.time()
        self.store[k] = (v, t + self.ttl)
        if len(self.store) > self.cap:
            self.store.popitem(last=False)

_cache = LRUCacheTTL(cap=256, ttl=2*3600)
def get_cached(k): return _cache.get(k)
def set_cached(k, v): _cache.set(k, v)
