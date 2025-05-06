import json
import hashlib
import redis

class CacheSystem:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db, #default redis database index
            decode_responses=True #decode the byte stream 
        )
        self.ttl = 60*60

        self.format_keys = {
            "paragraph": "llm:paragraph_responses",
            "single_line": "llm:single_line_responses"
        }

    def cache_response(self, query:str, response:str, response_format:str, reranked:bool):
        if response_format not in self.format_keys:
            raise ValueError(f"Invalid Response Format, must be paragraph or single_line")
        if not isinstance(reranked, bool):
            raise ValueError(f"reranked must be a boolen value")
        
        format_key = self.format_keys[response_format]

        query_hash = hashlib.md5(query.encode()).hexdigest()

        self.redis.hset(format_key, query_hash, json.dumps({
            "query": query,
            "response": response,
            "timestamp": self.redis.time()[0],
            "reranked": str(reranked)
        }))

        #If not expiration time set in the format_key, set expiration time to the entire hash
        if not self.redis.ttl(format_key) > 0:
            self.redis.expire(format_key, self.ttl)

    def get_cached_response(self, query:str, response_format:str):
        if response_format not in self.format_keys:
            raise ValueError(f"Response format error, must be either paragraph or single_line")
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        format_key = self.format_keys[response_format]

        result = self.redis.hget(format_key, query_hash)

        if result:
            return json.loads(result)["response"], json.loads(result)["reranked"]
        
        return None
    
