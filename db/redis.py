import os, json
from redis import Redis

_redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)

class cache:
    @staticmethod
    def get(key: str):
        val = _redis.get(key)
        return json.loads(val) if val else None

    @staticmethod
    def set(key: str, value: dict, ex: int | None = None):
        _redis.set(key, json.dumps(value), ex=ex)
