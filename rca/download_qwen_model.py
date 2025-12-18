from pathlib import Path


def main():
    """
    使用魔搭 ModelScope 从云端下载 Qwen3-8B 到本地目录。

    默认下载目录（在项目根目录下）：
        <OpenRCA 根目录>/models/Qwen3-8B

    使用方法（在 macOS 上）：
        1. 安装依赖（只需一次）：
               pip install modelscope
        2. 在 OpenRCA 根目录运行脚本：
               python -m rca.download_qwen_model
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        raise RuntimeError(
            "未找到 modelscope，请先在当前环境安装：\n"
            "    pip install modelscope\n"
            "然后再次运行：python -m rca.download_qwen_model"
        )

    # 选用的模型：Qwen3-8B（在魔搭上的仓库名为 Qwen/Qwen3-8B）
    # 目前魔搭上没有单独的 “-Instruct” 命名，这个仓库本身就是对话/指令能力的版本
    model_id = "Qwen/Qwen3-8B"

    # 项目根目录：.../OpenRCA
    project_root = Path(__file__).resolve().parent.parent

    # 本地保存目录：<项目根>/models/Qwen3-8B
    target_dir = project_root / "models" / "Qwen3-8B"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"准备从魔搭下载模型：{model_id}")
    print(f"目标目录：{target_dir}")

    # 如果已经下过一次，直接提示并退出，避免重复下载
    if (target_dir / "config.json").exists():
        print(f"检测到 {target_dir} 已存在模型文件，跳过下载。")
        print("如需重新下载，请先手动删除该目录。")
        return str(target_dir)

    cache_dir = snapshot_download(
        model_id=model_id,
        cache_dir=str(target_dir),
        revision=None,
        ignore_file_pattern=None,
    )

    print(f"\n✅ 下载完成，本地目录：{cache_dir}")
    print("接下来你可以打包上传到服务器，例如：")
    print("  cd \"$PROJECT_ROOT\"  # 即 OpenRCA 根目录")
    print("  cd models")
    print("  tar -czf Qwen3-8B.tar.gz Qwen3-8B/")

    return str(cache_dir)


if __name__ == "__main__":
    main()


