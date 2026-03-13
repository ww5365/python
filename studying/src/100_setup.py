'''
1. 了解 ： python 代码打包，安装
2. python调用高性能C++的接口，怎么混合编译，打包，安装？

'''


import shutil
import stat
import os
import subprocess
import sys
import logging
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

logging.basicConfig(level=logging.INFO)


def modify_version():
    default_version = "7.3+t50"

    init_file = "dynamic_emb/__init__.py"
    with open(init_file, "r") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "__version__ = " not in line:
                continue
            lines[idx] = f"__version__ = '{default_version}'\n"
            break

    flag = os.O_WRONLY | os.O_TRUNC
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(init_file, flag, mode), "w") as out:
        out.writelines(lines)
    return default_version


def ensure_pybind11():
    """确保pybind11可用"""
    try:
        import pybind11
        return pybind11.get_cmake_dir()
    except ImportError:
        logging.info("Installing pybind11...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11
        return pybind11.get_cmake_dir()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ('pybind11-dir=', None, 'Path to pybind11 installation'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.pybind11_dir = None

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        # 确保pybind11可用
        if not self.pybind11_dir:
            self.pybind11_dir = ensure_pybind11()

        # 首先确保CMake可用
        try:
            cmake_exe = shutil.which("cmake")
            if cmake_exe is None:
                raise RuntimeError("CMake must be installed to build the extensions")
            subprocess.check_output([cmake_exe, '--version'])
        except OSError as e:
            raise RuntimeError("CMake must be installed to build the extensions") from e

        super().run()
        self.copy_cust_ops_libraries()

    def copy_cust_ops_libraries(self):
        """
        复制自定义算子库到包目录
        CMakeList阶段生成的三个so：
        1. dynamic_emb_extensions.cpython-311-x86_64-linux-gnu.so  主so，放在build/lib.linux-x86_64-cpython-311/ 目录下
        2. libasc_kernel_lib.so： 在temp.linux-x86_64-cpython-311/lib下 会被拷贝到build/lib.linux-x86_64-cpython-311/ 目录下
        3. libdynamic_emb_op_npu.so ： 在temp.linux-x86_64-cpython-311/lib下 会被拷贝到build/lib.linux-x86_64-cpython-311/ 目录下

        """


        ext_path = self.get_ext_fullpath('dynamic_emb_extensions')
        package_dir = os.path.dirname(ext_path)

        # 查找依赖库
        lib_patterns = [
            'libdynamic_emb_op_*.so',
            'libasc_kernel_lib.so'
        ]

        # 搜索路径
        search_paths = [
            os.path.join(self.build_temp, 'lib'),
        ]

        for pattern in lib_patterns:
            for search_path in search_paths:
                if os.path.exists(search_path):
                    for lib_file in Path(search_path).glob(pattern):
                        if lib_file.is_file():
                            lib_dest_path = os.path.join(package_dir, lib_file.name)
                            logging.info(f"Copying {lib_file} to {lib_dest_path}")
                            shutil.copy2(lib_file, lib_dest_path)

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)

    def build_cmake_extension(self, ext):
        '''
        配置和调用 CMake 来编译 C++ 扩展，并确保生成的动态库文件被放置在正确的位置，以便最终被打包进 Python 分发包中
        '''

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))   #  build/lib.linux-x86_64-cpython-311  setuptools工具自动创建的，这个扩展dynamic_emb_extensions的目录

        # CMake配置参数
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
            f'-Dpybind11_DIR={self.pybind11_dir}',
            f'-DCMAKE_PREFIX_PATH={self.pybind11_dir}'
        ]

        # 从环境变量获取配置或使用默认值
        run_mode = os.getenv('RUN_MODE', 'npu')
        soc_version = os.getenv('SOC_VERSION', 'Ascend950PR_9579')
        ascend_cann_path = os.getenv('ASCEND_CANN_PACKAGE_PATH', '/usr/local/Ascend/ascend-toolkit/latest')

        cmake_args.extend([
            f'-DRUN_MODE={run_mode}',
            f'-DSOC_VERSION={soc_version}',
            f'-DASCEND_CANN_PACKAGE_PATH={ascend_cann_path}'
        ])

        # 构建目录
        build_temp = self.build_temp    #  build/temp.linux-x86_64-cpython-311
        # # self.build_temp 是从 build_ext 基类继承而来的属性，由 setuptools 在命令执行过程中自动设置。它通常指向一个临时目录（如 build/temp.linux-x86_64-cpython-310），用于存放编译过程中的中间文件。在自定义的 CMakeBuild 类中，可以直接使用该属性，因为它已经在基类中初始化

        os.makedirs(build_temp, exist_ok=True)

        logging.info("Configuring CMake project...")
        logging.info(f"CMake args: {cmake_args}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)

        logging.info("Building project...")
        subprocess.check_call(['cmake', '--build', '.', '--verbose', '--config', 'Debug' if self.debug else 'Release'],
                              cwd=build_temp)

        # 这里使用cmake进行这个扩展：dynamic_emb_extensions 包的编译构建工作   其中build_temp 存放构建过程中生成的临时文件
        # 具体的构建过程，可以参考本级目录下：CMakeList.txt文件


# 设置依赖
setup_requires = ['pybind11']


class CustomInstall(install):
    def run(self):
        super().run()
        # 设置库文件权限
        self.fix_library_permissions()

    def fix_library_permissions(self):
        """修复库文件权限"""
        package_dir = self.install_lib
        if package_dir and os.path.exists(package_dir):
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    if file.endswith('.so'):
                        file_path = os.path.join(root, file)
                        os.chmod(file_path, 0o755)  # 设置可执行权限


# 编译 so文件
script_path = os.getcwd()
common_script = os.path.join(script_path, "./scripts/build.sh")
os.chmod(common_script, 0o755)
res = subprocess.run([common_script], shell=False)
if res.returncode:
    raise RuntimeError("compile so files failed!")

# 安装common
common_dir = os.path.join(script_path, "../../common")
subprocess.run(
    [
        "python3",
        "setup.py",
        "bdist_wheel",
    ],
    cwd=common_dir,
    shell=False,
)
if os.path.exists("rec_sdk_common"):
    shutil.rmtree("rec_sdk_common")
current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, "..", "..", "common", "rec_sdk_common")
dest_path = os.path.join(current_dir, "rec_sdk_common")

if os.path.exists(source_path):
    shutil.copytree(source_path, dest_path)

# 将本地包名替换为目标包名
TARGET_PACKAGE_NAME = "dynamic_emb"
version = modify_version()
local_packages = find_packages(exclude=("tests*", "*test"))     # 搜索当前目录及其子目录中所有包含 __init__.py 文件的文件夹

target_packages = []
package_dir_mapping = {}
for local_pkg in local_packages:
    target_pkg = local_pkg.replace("dynamic_emb", TARGET_PACKAGE_NAME, 1)
    target_packages.append(target_pkg)
    package_dir_mapping[target_pkg] = local_pkg.replace(".", os.sep)

setup(
    name="dynamic_emb",   # 指定分发的名称，即最终的包名。用户安装时使用 pip install dynamic_emb，生成的 wheel 文件名也以此开头
    version=version,
    packages=target_packages,   # 列出需要包含在分发包中的所有 Python 包（即含有 __init__.py 的目录）。target_packages 是通过 find_packages() 获取原始包列表后，经过包名映射得到的列表。它确保 dynamic_emb 及其子包都被打包
    package_dir=package_dir_mapping, # 建立包名到实际目录的映射。package_dir_mapping 将目标包名（如 dynamic_emb.sub）映射到源码路径（如 dynamic_emb/sub）。这使得即使包名与目录名不一致，setuptools 也能正确找到源码。本例中包名未变，但该映射保证了结构清晰
    ext_modules=[CMakeExtension("dynamic_emb_extensions", "./")],
    # 声明需要编译的 C/C++ 扩展模块。这里使用自定义的 CMakeExtension 类，它继承自 setuptools.Extension，但将实际构建过程委托给 CMake。
    # 扩展名为 dynamic_emb_extensions，源码目录为当前目录 ./。构建后生成的动态库（如 .so 文件）将成为包的一部分，供 Python 导入。
    # 自定义类的主要作用是为 CMake 构建提供一个“占位符”，让 setuptools 知道存在一个需要构建的扩展模块，并传递构建所需的源目录信息。
    cmdclass={
        'build_ext': CMakeBuild,  # cmdclass 允许覆盖 setuptools 的默认命令类。这里将 build_ext 命令替换为自定义的 CMakeBuild 类
        'install': CustomInstall
    },

    #  python setup.py build_ext（或任何隐式调用 build_ext 的命令，如 build、install、bdist_wheel）时
    #  setuptools 会实例化 CMakeBuild（因为被 cmdclass 替换了默认的 build_ext）。
    #
    # 在 CMakeBuild.run() 方法中，它会先进行一些准备工作（确保 pybind11 可用、检查 CMake 等），然后调用父类的 run()，最终遍历所有扩展模块。
    #
    # 对于每个扩展模块，CMakeBuild.build_extension(ext) 会被调用。该方法检查 ext 是否为 CMakeExtension 的实例：
    #
    # 如果是，则调用自定义的 build_cmake_extension(ext)，利用 CMake 配置和编译该扩展。
    #
    # 如果不是（例如普通的 Extension），则回退到父类的构建逻辑（即传统的 distutils 编译）。
    #
    # 在 build_cmake_extension 中，会使用之前保存的 sourcedir 和当前环境变量，通过 CMake 生成并编译扩展库，最终将生成的 .so 文件放到正确位置。


    # 总结： 编译扩展模块： dynamic_emb_extensions  使用的CMakebuild实例中的build_extension实现。


    package_data={
        TARGET_PACKAGE_NAME: ['*.so'],  # 主包下的.so
        "rec_sdk_common": ['lib/*.so'],
    },
    # 执行 pip install . 或 python setup.py install 后，setuptools 会根据配置将文件复制到 Python 的 site-packages 目录中（例如 /usr/local/lib/python3.11/dist-packages/ 或虚拟环境下的 site-packages）
    #  dynamic_emb_extensions.cpython-311-x86_64-linux-gnu.so
    # → 安装为 site-packages/dynamic_emb_extensions.so （setuptools 会自动重命名为不带平台标签的模块名）
    # libasc_kernel_lib.so 和 libdynamic_emb_op_npu.so
    # → 根据 package_data 配置，它们会被安装到 site-packages/dynamic_emb/ 目录下

    include_package_data=True,
    setup_requires=setup_requires,
    install_requires=['pybind11'],
    zip_safe=False,
)

if 'bdist_wheel' in sys.argv:
    move_whl_script = os.path.join(
        script_path, "./scripts/move_whl_file_2_pkg_dir.sh"
    )
    os.chmod(move_whl_script, 0o755)
    res = subprocess.run([move_whl_script], shell=False)
    if res.returncode:
        raise RuntimeError("move whl file to pkg dir failed!")

    gen_tar_script = os.path.join(
        script_path, "./scripts/gen_tar_pkg.sh"
    )
    os.chmod(gen_tar_script, 0o755)
    res = subprocess.run([gen_tar_script], shell=False)
    if res.returncode:
        raise RuntimeError("gen dynamicemb's tar pkg failed!")
