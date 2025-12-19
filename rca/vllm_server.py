import argparse
import os
import signal
import sys
from pathlib import Path
from subprocess import Popen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 你在服务器上上传的模型相对路径，如有调整，这里改一下即可
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-8B" / "Qwen" / "Qwen3-8B"
PID_FILE = PROJECT_ROOT / "vllm_server.pid"


def start_server(
    model_path: Path = DEFAULT_MODEL_PATH,
    port: int = 8000,
    served_model_name: str = "Qwen/Qwen3-8B",
    tensor_parallel_size: int = 1,
):
    if PID_FILE.exists():
        print(f"⚠️ 检测到 {PID_FILE} 已存在，可能已有 vLLM 进程在运行。")
        print("如需强制重新启动，请先执行: python -m rca.vllm_server stop")
        return

    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    # 使用当前 Python 解释器以模块形式启动 vLLM 的 OpenAI 兼容服务
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]

    print("即将启动 vLLM 服务：")
    print("  命令：", " ".join(cmd))
    print(f"  模型路径：{model_path}")
    print(f"  监听端口：{port}")
    print(f"  served_model_name：{served_model_name}")

    # 在后台启动，不阻塞当前终端
    proc = Popen(cmd)
    PID_FILE.write_text(str(proc.pid))
    print(f"\n✅ vLLM 已在后台启动，PID={proc.pid}")
    print(f"PID 已写入：{PID_FILE}")


def stop_server():
    if not PID_FILE.exists():
        print("⚠️ 未找到 PID 文件，可能服务未运行：", PID_FILE)
        return

    pid_text = PID_FILE.read_text().strip()
    try:
        pid = int(pid_text)
    except ValueError:
        print(f"PID 文件内容异常：{pid_text}")
        PID_FILE.unlink(missing_ok=True)
        return

    print(f"尝试停止 vLLM 进程，PID={pid} ...")
    try:
        os.kill(pid, signal.SIGTERM)
        print("已发送 SIGTERM 信号。")
    except ProcessLookupError:
        print("进程不存在，可能已经退出。")
    finally:
        PID_FILE.unlink(missing_ok=True)
        print("已删除 PID 文件。")


def main():
    parser = argparse.ArgumentParser(description="管理 vLLM 大模型服务（启动/关闭）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_start = subparsers.add_parser("start", help="启动 vLLM 服务")
    p_start.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VLLM_PORT", "8000")),
        help="监听端口（默认 8000）",
    )
    p_start.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("VLLM_TP", "1")),
        help="tensor parallel size（默认 1）",
    )
    p_start.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型目录（默认使用脚本内置的 DEFAULT_MODEL_PATH）",
    )

    subparsers.add_parser("stop", help="停止 vLLM 服务")

    args = parser.parse_args()

    if args.command == "start":
        model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else DEFAULT_MODEL_PATH
        )
        start_server(
            model_path=model_path,
            port=args.port,
            served_model_name="Qwen/Qwen3-8B",
            tensor_parallel_size=args.tensor_parallel_size,
        )
    elif args.command == "stop":
        stop_server()


if __name__ == "__main__":
    main()


