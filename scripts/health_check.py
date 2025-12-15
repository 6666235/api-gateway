#!/usr/bin/env python
"""
健康检查脚本
用于监控系统和 CI/CD 集成

用法:
    python scripts/health_check.py [--url URL] [--timeout TIMEOUT]
"""
import argparse
import sys
import json
import time

try:
    import httpx
except ImportError:
    print("请安装 httpx: pip install httpx")
    sys.exit(1)


def check_health(url: str, timeout: int = 10) -> dict:
    """检查服务健康状态"""
    try:
        start = time.time()
        response = httpx.get(f"{url}/health", timeout=timeout)
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status": data.get("status", "unknown"),
                "latency_ms": round(latency, 2),
                "details": data,
            }
        else:
            return {
                "success": False,
                "status": "error",
                "latency_ms": round(latency, 2),
                "error": f"HTTP {response.status_code}",
            }
    except httpx.TimeoutException:
        return {
            "success": False,
            "status": "timeout",
            "error": f"Request timed out after {timeout}s",
        }
    except httpx.ConnectError:
        return {
            "success": False,
            "status": "unreachable",
            "error": "Could not connect to server",
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e),
        }


def check_detailed_health(url: str, timeout: int = 10) -> dict:
    """检查详细健康状态"""
    try:
        response = httpx.get(f"{url}/api/health/detailed", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def check_metrics(url: str, timeout: int = 10) -> dict:
    """检查 Prometheus 指标"""
    try:
        response = httpx.get(f"{url}/api/system/metrics", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="AI Hub 健康检查")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="服务 URL"
    )
    parser.add_argument("--timeout", type=int, default=10, help="超时时间（秒）")
    parser.add_argument("--detailed", action="store_true", help="显示详细信息")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--exit-code", action="store_true", help="根据状态返回退出码")

    args = parser.parse_args()

    # 基础健康检查
    result = check_health(args.url, args.timeout)

    if args.detailed and result["success"]:
        result["detailed"] = check_detailed_health(args.url, args.timeout)
        result["metrics"] = check_metrics(args.url, args.timeout)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result["success"]:
            print(f"✅ 服务正常")
            print(f"   状态: {result['status']}")
            print(f"   延迟: {result['latency_ms']}ms")

            if args.detailed:
                details = result.get("detailed", {})
                print(f"\n📊 系统资源:")
                print(f"   CPU: {details.get('cpu_percent', 'N/A')}%")
                print(f"   内存: {details.get('memory_percent', 'N/A')}%")
                print(f"   磁盘: {details.get('disk_percent', 'N/A')}%")

                checks = result.get("details", {}).get("checks", {})
                if checks:
                    print(f"\n🔍 组件状态:")
                    for name, check in checks.items():
                        status_icon = "✅" if check.get("status") == "healthy" else "❌"
                        latency = check.get("latency_ms", "N/A")
                        print(f"   {status_icon} {name}: {latency}ms")
        else:
            print(f"❌ 服务异常")
            print(f"   状态: {result['status']}")
            print(f"   错误: {result.get('error', 'Unknown')}")

    if args.exit_code:
        sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
