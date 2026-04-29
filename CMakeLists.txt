# ==============================================================================
# Project Setup
# ==============================================================================
cmake_minimum_required(VERSION 3.18)

project(FSI_Simulator LANGUAGES CXX CUDA C)

# ---------------------------------------------------------------------------
# MSVC 下，spdlog 依赖的 fmt 会在 base.h 里做如下静态断言：
#   static_assert(!1 || use_utf8(), "Unicode support requires compiling with /utf-8")
# 它要求源代码与执行字符集都是 UTF-8。中文 Windows 上 MSVC 默认用 GBK(936)，
# 必须显式打开 /utf-8（等价于 /source-charset:utf-8 /execution-charset:utf-8）。
#
# 注意：仅通过 target_compile_options 加 /utf-8 在 Windows + CUDA + MSBuild 下
# 有时会被模板里的其它 -Xcompiler 组抵消。最稳妥的做法是写进全局 flags，
# 让它进入 vcxproj 的 AdditionalOptions，所有 .cpp / .cu 一视同仁。
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# ---------------------------------------------------------------------------
# fmt 的 Unicode 静态断言在 nvcc 前端 (cudafe1) 里无法识别 /utf-8，
# 因为 nvcc 前端不定义 _MSVC_EXECUTION_CHARACTER_SET。
# 定义 FMT_UNICODE=0 让 fmt 的 use_utf8() 返回 true（注意其内部实现是
# static_assert(!FMT_UNICODE || use_utf8())，FMT_UNICODE=0 后整条断言短路）。
# 这不影响 fmt 的实际功能，因为我们的日志都是 ASCII / UTF-8 安全字符。
# ---------------------------------------------------------------------------
add_compile_definitions(FMT_UNICODE=0)

# 显式指定 CUDA 架构，避免使用被 CUDA 12.6+ 弃用的 sm_52 默认值。
# 使用 "native" 让 nvcc 自动检测本机 GPU 架构，只为当前机器的显卡编译，
# 编译速度最快、产物最小，但换机器/换卡后需要重新编译。
# 需要分发给他人时可改回显式列表，例如: 75 86 89 120
#   75 = Turing (RTX 20xx), 86 = Ampere (RTX 30xx),
#   89 = Ada (RTX 40xx),    120 = Blackwell (RTX 50xx，需 CUDA >= 12.8)。
# 用 FORCE 强制覆盖 CMake 缓存，防止遗留的 52 值继续生效。
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 防止release模式下的LTO问题
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  message(STATUS "No build type specified, defaulting to ${CMAKE_BUILD_TYPE}.")
endif()

# 为所有目标设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# 为所有目标设置库文件输出目录
# 这同时影响静态库和共享库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)

# 静态库的输出目录（可选，如果想和共享库分开）
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)

message(STATUS "Configuring FSI_Simulator version 0.1.0")
message(STATUS "Compiler CXX: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler CUDA: ${CMAKE_CUDA_COMPILER}")

# 启用 compile_commands.json 生成，用于编辑器IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


# ==============================================================================
# Find Dependencies (via vcpkg)
# ==============================================================================
message(STATUS "Finding required packages...")

# Bypass vcpkg's Python wrapper: we want to use the active (conda/venv) Python,
# not the python3 package that vcpkg pulls in as a transitive dependency of VTK.
# NOTE: vcpkg's wrapper is triggered via the toolchain's find_package override.
# The correct way to suppress it is CMAKE_DISABLE_FIND_PACKAGE_<name> or by
# explicitly clearing the wrapper path before calling find_package(Python).
set(CMAKE_DISABLE_FIND_PACKAGE_Python3 TRUE)

# Prefer the active Python (conda / venv / the interpreter that invoked us).
set(Python_FIND_VIRTUALENV ONLY)
set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY NEVER)
set(Python_FIND_FRAMEWORK NEVER)

# If the build is driven by scikit-build-core / pip, Python_EXECUTABLE is
# already injected. Derive Python_ROOT_DIR from it so FindPython can locate
# the matching headers and import library in a conda environment.
if(DEFINED Python_EXECUTABLE AND NOT DEFINED Python_ROOT_DIR)
    get_filename_component(_py_exec_dir "${Python_EXECUTABLE}" DIRECTORY)
    set(Python_ROOT_DIR "${_py_exec_dir}" CACHE PATH "Python root dir derived from Python_EXECUTABLE")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Version: ${Python_VERSION}")
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")

find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(pybind11 2.6 REQUIRED CONFIG)
find_package(spdlog REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)
find_package(glm REQUIRED)
find_package(VTK REQUIRED)
find_package(OpenMP REQUIRED)

# 必须显式 find_package(CUDAToolkit)，才能获得 CUDA::cusparse / CUDA::cusolver
# 等 imported target。仅靠 project(... LANGUAGES CUDA) 不会创建这些 target。
# CMake 3.17+ 起 CUDAToolkit 模块是获取 CUDA 运行时库的官方推荐方式。
find_package(CUDAToolkit REQUIRED)
message(STATUS "  - CUDAToolkit: ${CUDAToolkit_VERSION}")

option(USE_CUDSS "Enable the cuDSS sparse direct solver" OFF)
option(USE_SUITESPARSE "Enable the CholmodLDLT solver" ON)

if(USE_SUITESPARSE)
    message(STATUS "Attempting to find SuiteSparse/CHOLMOD...")
    find_package(CHOLMOD CONFIG REQUIRED)
    # Prefer the static target when available (requires x64-windows-static triplet
    # or a build that provides it); otherwise fall back to the shared target,
    # which is what vcpkg's default x64-windows triplet ships.
    if(TARGET SuiteSparse::CHOLMOD_static)
        set(USE_SUITESPARSE_LIBS SuiteSparse::CHOLMOD_static)
    elseif(TARGET SuiteSparse::CHOLMOD)
        set(USE_SUITESPARSE_LIBS SuiteSparse::CHOLMOD)
    else()
        message(FATAL_ERROR "CHOLMOD target not found (neither SuiteSparse::CHOLMOD_static nor SuiteSparse::CHOLMOD).")
    endif()
    message(STATUS "Using SuiteSparse target: ${USE_SUITESPARSE_LIBS}")
else()
    message(STATUS "Skipping SuiteSparse/CHOLMOD as per configuration.")
    set(USE_SUITESPARSE_LIBS "")
endif()

if(USE_CUDSS)
    # CUDAToolkit 已在上方无条件 find_package 过，这里直接用 imported target。
    # CUDA::cudss 是 CMake 3.25+ 引入的官方目标名；旧 CMake 走手动查找分支。
    if(TARGET CUDA::cudss)
        message(STATUS "Found cuDSS via CMake's CUDA support (TARGET CUDA::cudss)")
        set(USE_CUDSS_LIBS CUDA::cudss)
    else()
        # 手动查找，适用于较旧的CMake版本
        find_library(CUDSS_LIBRARY NAMES cudss
                     HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib/x64)
        if(CUDSS_LIBRARY)
            message(STATUS "Found cuDSS library manually: ${CUDSS_LIBRARY}")
            set(USE_CUDSS_LIBS ${CUDSS_LIBRARY})
        else()
            message(FATAL_ERROR "cuDSS library not found, but USE_CUDSS is ON. Please check your CUDA installation.")
        endif()
    endif()
else()
    message(STATUS "Skipping cuDSS as per configuration.")
    set(USE_CUDSS_LIBS "")
endif()

# find_package(tinyobjloader REQUIRED) # 如果需要
# find_package(GAIA REQUIRED) # 假设你的VBD库也通过vcpkg或find_package管理

message(STATUS "  - Eigen3: ${EIGEN3_VERSION}")
message(STATUS "  - pybind11: ${pybind11_VERSION}")
message(STATUS "  - spdlog: ${spdlog_VERSION}")
message(STATUS "  - nlohmann_json: ${nlohmann_json_VERSION}")
message(STATUS "  - glm: ${GLM_VERSION_STRING}")
message(STATUS "  - OpenMP: ${OpenMP_CXX_FLAGS}")

# ==============================================================================
# Project Structure and Source Files
# ==============================================================================

# 定义头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 自动查找所有源文件 (.cpp, .cu)
# 注意：当添加或删除文件时，需要重新运行CMake
file(GLOB_RECURSE CORE_SOURCES
    "src/*.cpp"
    "src/*.cu"
)

# 排除Python绑定文件，它将单独处理
file(GLOB PYTHON_BINDINGS "src/python/*.cpp")
list(REMOVE_ITEM CORE_SOURCES ${PYTHON_BINDINGS})

# 排除main.cpp，它将用于可执行文件
file(GLOB MAIN_EXECUTABLE "src/app/*.cpp")
list(REMOVE_ITEM CORE_SOURCES ${MAIN_EXECUTABLE})

# ==============================================================================
# Build Core Library (Static Library)
# ==============================================================================
# 我们将核心功能编译成一个静态库，这样可以被Python绑定和C++可执行文件共享

# 把 fsi_core 改为 STATIC 库（原来是 OBJECT）。
# 原因：CUDA_SEPARABLE_COMPILATION=ON 时，所有 .cu 都以 -rdc=true 编译，
# 会产生 __cudaRegisterLinkedBinary_* 需要在最终 target 做 device link 解析。
# 但 OBJECT 库 + $<TARGET_OBJECTS:fsi_core> 的用法只把 .obj 文件塞进下游命令行，
# CMake 不会识别下游含 CUDA 源，也就不会触发 nvlink 这一步，最终表现为：
#   error LNK2019: unresolved external symbol __cudaRegisterLinkedBinary_xxx
# 改成 STATIC 后，CMake 可以在 fsi_core 自身做 device link 并正确传播到下游。
add_library(fsi_core STATIC ${CORE_SOURCES})

set_target_properties(fsi_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    # 让 fsi_core 静态库在产出前就把 device 符号解析掉，
    # 下游 pybind11 模块 / 可执行文件就无需再做 device link。
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 为fsi_core设置包含目录
target_include_directories(fsi_core PUBLIC
    # 当安装时，头文件在'include'目录下
    $<INSTALL_INTERFACE:include>
    # 当在构建树中时，头文件在'${PROJECT_SOURCE_DIR}/include'目录下
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

if(USE_SUITESPARSE)
    target_compile_definitions(fsi_core PUBLIC USE_SUITESPARSE)
endif()

if(USE_CUDSS)
    target_compile_definitions(fsi_core PUBLIC USE_CUDSS)
endif()

target_compile_definitions(fsi_core PUBLIC GLM_ENABLE_EXPERIMENTAL)

# 为fsi_core链接它需要的所有库
# 使用PUBLIC关键字，这样任何链接到fsi_core的目标也会自动链接到这些库
target_link_libraries(fsi_core PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    glm::glm
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    CUDA::cudart
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML # 用于写入.vtu, .vts等XML格式文件
    ${USE_SUITESPARSE_LIBS}
    ${USE_CUDSS_LIBS}
    CUDA::cusparse
    CUDA::cusolver
)

# （CUDA_SEPARABLE_COMPILATION / CUDA_RESOLVE_DEVICE_SYMBOLS 已在
# add_library(fsi_core STATIC ...) 的 set_target_properties 里设置。）

# ==============================================================================
# Build Python Bindings
# ==============================================================================
if(PYTHON_BINDINGS)
    message(STATUS "Building Python bindings...")

    # 使用 pybind11 提供的函数来创建 Python 模块
    # 第一个参数是模块名，也是最终生成的文件名 (e.g., fsi_simulator.so)
    # 直接把 fsi_core 作为链接库，而不是用 $<TARGET_OBJECTS:fsi_core>。
    # 原因：$<TARGET_OBJECTS:...> 只把 .obj 文件塞进命令行，CMake 无法识别
    # 下游含 CUDA 代码，从而跳过 device link，导致 __cudaRegisterLinkedBinary_*
    # 未定义。用 target_link_libraries + STATIC 库的组合，CMake 会自动处理。
    pybind11_add_module(fsi_simulator ${PYTHON_BINDINGS})

    # 链接核心静态库和 pybind11 库
    target_link_libraries(fsi_simulator PRIVATE
        fsi_core
        pybind11::module
    )

    install(TARGETS fsi_simulator
            LIBRARY  # 表示这是一个库文件
            DESTINATION . # 安装到包的根目录
            COMPONENT python_module # 匹配 pyproject.toml 中的组件名
    )
    
    # 2. 创建并安装 __init__.py 文件
    # 这样做可以让用户直接 `import fsi_simulator`
    # 并从.so文件中导入所有内容
    set(INIT_PY_CONTENT "from .fsi_simulator import *")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${INIT_PY_CONTENT}")
    
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
            DESTINATION . # 和.so文件安装到同一个目录下
            COMPONENT python_module
    )

endif()


# ==============================================================================
# Build C++ Executable (for testing or C++-only runs)
# ==============================================================================
if(MAIN_EXECUTABLE)
    message(STATUS "Building C++ executable...")

    # 不再用 $<TARGET_OBJECTS:fsi_core>，直接把 fsi_core 作为链接库。
    # fsi_core 已在内部完成 device link（CUDA_RESOLVE_DEVICE_SYMBOLS ON），
    # 下游只需普通 host 链接即可。
    add_executable(fsi_main ${MAIN_EXECUTABLE})

    target_link_libraries(fsi_main PRIVATE
        fsi_core
    )

endif()


# ==============================================================================
# Final Touches
# ==============================================================================
# 打印目标信息
message(STATUS "Configuration finished. Targets to be built:")
if(TARGET fsi_core)
    message(STATUS "  - Library: fsi_core")
endif()
if(TARGET fsi_simulator)
    message(STATUS "  - Python Module: fsi_simulator")
endif()
if(TARGET fsi_main)
    message(STATUS "  - Executable: fsi_main")
endif()