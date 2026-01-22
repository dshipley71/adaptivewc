"""
Check Redis and Redis Search availability.

Run: python check_redis.py
"""

import sys

try:
    import redis
except ImportError:
    print("❌ redis-py not installed. Run: pip install redis")
    sys.exit(1)


def check_python_imports():
    """Check if Redis Search Python imports work."""
    try:
        from redis.commands.search.field import TagField, TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        from redis.commands.search.query import Query
        return True, None
    except ImportError as e:
        return False, str(e)


def check_redis_search(client):
    """Check if Redis Search module is loaded."""
    try:
        modules = client.execute_command("MODULE", "LIST")
        for module in modules:
            if isinstance(module, (list, tuple)):
                for i, item in enumerate(module):
                    if item in (b'name', 'name') and i + 1 < len(module):
                        name = module[i + 1]
                        if isinstance(name, bytes):
                            name = name.decode()
                        if name.lower() in ('search', 'ft'):
                            return True, name
        return False, None
    except redis.ResponseError:
        # Try a direct FT command
        try:
            client.execute_command("FT._LIST")
            return True, "search"
        except redis.ResponseError:
            return False, None


def main():
    url = "redis://localhost:6379/0"
    if len(sys.argv) > 1:
        url = sys.argv[1]

    print(f"Checking Redis at: {url}")
    print("-" * 50)

    # Check Python imports first
    imports_ok, import_error = check_python_imports()
    if imports_ok:
        print("✅ Python imports: OK (redis.commands.search)")
    else:
        print("❌ Python imports: FAILED")
        print(f"   Error: {import_error}")
        print("   Fix: pip install --upgrade redis")

    print()

    try:
        client = redis.Redis.from_url(url)

        # Test connection
        if client.ping():
            print("✅ Redis connection: OK")
        else:
            print("❌ Redis connection: FAILED")
            return 1

        # Get Redis version
        info = client.info("server")
        version = info.get("redis_version", "unknown")
        print(f"   Redis version: {version}")

        # Check for Redis Search module
        has_search, module_name = check_redis_search(client)
        if has_search:
            print(f"✅ Redis Search module: LOADED ({module_name})")
        else:
            print("⚠️  Redis Search module: NOT LOADED")
            print("   Install Redis Stack: docker run -d -p 6379:6379 redis/redis-stack")

        # Overall status
        print()
        if imports_ok and has_search:
            print("✅ Vector search: HNSW (fast ANN) - Full support")
        elif has_search and not imports_ok:
            print("⚠️  Vector search: LINEAR SCAN (slow)")
            print("   Server has Search but Python can't use it")
            print("   Run: pip install --upgrade redis")
        else:
            print("⚠️  Vector search: LINEAR SCAN (slow)")
            print("   Install Redis Stack for HNSW indexing")

        # Test basic operations
        client.set("_radiant_test", "ok", ex=5)
        val = client.get("_radiant_test")
        if val == b"ok":
            print("✅ Redis read/write: OK")
        else:
            print("❌ Redis read/write: FAILED")

    except redis.ConnectionError as e:
        print(f"❌ Cannot connect to Redis: {e}")
        print()
        print("Make sure Redis is running:")
        print("  docker run -d -p 6379:6379 redis/redis-stack-server")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    print("-" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
