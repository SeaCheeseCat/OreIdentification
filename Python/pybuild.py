from cx_Freeze import setup, Executable

# 要打包的Python脚本路径
script = "camera.py"

# 创建可执行文件的配置
exe = Executable(
    script=script,
    base=None,  # 对于GUI应用，可以设置为"Win32GUI"来隐藏控制台窗口
    targetName="stone"  # 生成的可执行文件名称
)

# 打包的参数配置
options = {
    "build_exe": {
        "packages": [],  # 需要打包的额外Python包列表
        "excludes": [],  # 不需要打包的Python包列表
        "include_files": [],  # 需要包含的文件或文件夹列表
        "include_msvcr": True  # 是否包含Microsoft Visual C++运行时库
    }
}

# 打包配置
setup(
    name="MyProgram",
    version="1.0",
    description="My Program Description",
    options=options,
    executables=[exe]
)
