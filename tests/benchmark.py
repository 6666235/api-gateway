"""
性能基准测试
运行: python tests/benchmark.py
"""
import asyncio
import time
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkResult:
    """基准测试结果"""

    def __init__(self, name: str, times: list):
        self.name = name
        self.times = times
        self.count = len(times)
        self.total = sum(times)
        self.avg = statistics.mean(times) if times else 0
        self.min = min(times) if times else 0
        self.max = max(times) if times else 0
        self.std = statistics.stdev(times) if len(times) > 1 else 0
        self.p50 = statistics.median(times) if times else 0
        self.p95 = (
            sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else self.max
        )
        self.p99 = (
            sorted(times)[int(len(times) * 0.99)] if len(times) > 100 else self.max
        )
        self.ops_per_sec = self.count / self.total if self.total > 0 else 0

    def __str__(self):
        return f"""
{self.name}:
  Count: {self.count}
  Total: {self.total*1000:.2f}ms
  Avg: {self.avg*1000:.4f}ms
  Min: {self.min*1000:.4f}ms
  Max: {self.max*1000:.4f}ms
  Std: {self.std*1000:.4f}ms
  P50: {self.p50*1000:.4f}ms
  P95: {self.p95*1000:.4f}ms
  P99: {self.p99*1000:.4f}ms
  Ops/sec: {self.ops_per_sec:.2f}
"""


async def benchmark_db_pool(iterations: int = 1000):
    """数据库连接池基准测试"""
    from modules.db_pool import EnhancedAsyncDatabasePool
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    pool = EnhancedAsyncDatabasePool(db_path=db_path, pool_size=5)
    await pool.initialize()

    # 创建测试表
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE bench (id INTEGER PRIMARY KEY, value TEXT)"
        )
        await conn.commit()

    # 插入测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await pool.execute(
            "INSERT INTO bench (value) VALUES (?)", (f"test_{i}",)
        )
        times.append(time.perf_counter() - start)

    insert_result = BenchmarkResult("DB Insert", times)

    # 查询测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await pool.fetch_one("SELECT * FROM bench WHERE id = ?", (i + 1,))
        times.append(time.perf_counter() - start)

    select_result = BenchmarkResult("DB Select", times)

    # 缓存命中测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await pool.fetch_one(
            "SELECT * FROM bench WHERE id = ?", (1,), use_cache=True
        )
        times.append(time.perf_counter() - start)

    cache_result = BenchmarkResult("DB Select (Cached)", times)

    await pool.close()
    os.remove(db_path)

    return [insert_result, select_result, cache_result]


async def benchmark_cache(iterations: int = 10000):
    """缓存基准测试"""
    from modules.cache import MemoryCache

    cache = MemoryCache(max_size=10000)

    # 写入测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await cache.set(f"key_{i}", {"data": f"value_{i}"})
        times.append(time.perf_counter() - start)

    write_result = BenchmarkResult("Cache Write", times)

    # 读取测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await cache.get(f"key_{i}")
        times.append(time.perf_counter() - start)

    read_result = BenchmarkResult("Cache Read", times)

    return [write_result, read_result]


def benchmark_waf(iterations: int = 10000):
    """WAF 基准测试"""
    from modules.security import waf

    test_inputs = [
        "Hello, this is a normal message",
        "SELECT * FROM users WHERE id = 1",
        "<script>alert('xss')</script>",
        "Normal text with some numbers 12345",
    ]

    times = []
    for i in range(iterations):
        content = test_inputs[i % len(test_inputs)]
        start = time.perf_counter()
        waf.check(content)
        times.append(time.perf_counter() - start)

    return BenchmarkResult("WAF Check", times)


def benchmark_csrf(iterations: int = 10000):
    """CSRF Token 基准测试"""
    from modules.security import csrf_protection

    # 生成测试
    times = []
    tokens = []
    for i in range(iterations):
        start = time.perf_counter()
        token = csrf_protection.generate_token(f"session_{i}")
        times.append(time.perf_counter() - start)
        tokens.append(token)

    generate_result = BenchmarkResult("CSRF Generate", times)

    # 验证测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        csrf_protection.validate_token(tokens[i], f"session_{i}")
        times.append(time.perf_counter() - start)

    validate_result = BenchmarkResult("CSRF Validate", times)

    return [generate_result, validate_result]


async def benchmark_task_queue(iterations: int = 1000):
    """任务队列基准测试"""
    from modules.queue import TaskQueue

    queue = TaskQueue(max_workers=4, persist=False)

    results = []

    async def dummy_task(x):
        return x * 2

    queue.register("dummy", dummy_task)
    await queue.start()

    # 入队测试
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        await queue.enqueue("dummy", i)
        times.append(time.perf_counter() - start)

    enqueue_result = BenchmarkResult("Task Enqueue", times)

    # 等待任务完成
    await asyncio.sleep(2)

    await queue.stop()

    return [enqueue_result]


def benchmark_validators(iterations: int = 10000):
    """验证器基准测试"""
    from modules.validators import (
        validate_username,
        validate_password,
        validate_email,
        prevent_xss,
    )

    # 用户名验证
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        validate_username(f"user_{i}")
        times.append(time.perf_counter() - start)

    username_result = BenchmarkResult("Validate Username", times)

    # 密码验证
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        validate_password(f"Password123!")
        times.append(time.perf_counter() - start)

    password_result = BenchmarkResult("Validate Password", times)

    # XSS 防护
    times = []
    test_content = "<script>alert('xss')</script><div onclick='evil()'>test</div>"
    for i in range(iterations):
        start = time.perf_counter()
        prevent_xss(test_content)
        times.append(time.perf_counter() - start)

    xss_result = BenchmarkResult("Prevent XSS", times)

    return [username_result, password_result, xss_result]


async def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("AI Hub 性能基准测试")
    print("=" * 60)

    all_results = []

    print("\n[1/6] 数据库连接池...")
    results = await benchmark_db_pool(500)
    all_results.extend(results)

    print("[2/6] 缓存...")
    results = await benchmark_cache(5000)
    all_results.extend(results)

    print("[3/6] WAF...")
    result = benchmark_waf(5000)
    all_results.append(result)

    print("[4/6] CSRF...")
    results = benchmark_csrf(5000)
    all_results.extend(results)

    print("[5/6] 任务队列...")
    results = await benchmark_task_queue(500)
    all_results.extend(results)

    print("[6/6] 验证器...")
    results = benchmark_validators(5000)
    all_results.extend(results)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    for result in all_results:
        print(result)

    # 汇总
    print("=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'测试项':<25} {'Ops/sec':>12} {'Avg (ms)':>12} {'P99 (ms)':>12}")
    print("-" * 60)
    for result in all_results:
        print(
            f"{result.name:<25} {result.ops_per_sec:>12.2f} {result.avg*1000:>12.4f} {result.p99*1000:>12.4f}"
        )


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
