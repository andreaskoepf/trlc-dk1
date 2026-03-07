cmake_minimum_required(VERSION 3.17)
project(trlc_dk1_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Python and nanobind ---
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find nanobind via CMake (pip-installed)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Find MuJoCo ---
# The pip-installed mujoco package provides headers and libraries.
# We find them via the mujoco Python package path.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mujoco; import os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MUJOCO_PKG_DIR
    RESULT_VARIABLE MUJOCO_RESULT)

if(MUJOCO_RESULT EQUAL 0 AND EXISTS "${MUJOCO_PKG_DIR}")
    # Headers are typically in the package directory or a subdirectory
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
        HINTS "${MUJOCO_PKG_DIR}/include" "${MUJOCO_PKG_DIR}"
              "${MUJOCO_PKG_DIR}/../mujoco.libs/../include"
        NO_DEFAULT_PATH)

    # Also check system paths
    if(NOT MUJOCO_INCLUDE_DIR)
        find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h)
    endif()

    # Find the shared library
    find_library(MUJOCO_LIBRARY
        NAMES mujoco
        HINTS "${MUJOCO_PKG_DIR}" "${MUJOCO_PKG_DIR}/../mujoco.libs"
              "${MUJOCO_PKG_DIR}/lib"
        NO_DEFAULT_PATH)

    if(NOT MUJOCO_LIBRARY)
        # Look for the shared library in the package directory
        # Linux: libmujoco.so.X.Y.Z in mujoco/ or mujoco.libs/
        # macOS: libmujoco.X.Y.Z.dylib in mujoco/
        file(GLOB MUJOCO_LIBRARY_GLOB "${MUJOCO_PKG_DIR}/libmujoco*.so*"
                                       "${MUJOCO_PKG_DIR}/libmujoco*.dylib"
                                       "${MUJOCO_PKG_DIR}/../mujoco.libs/libmujoco*.so*")
        if(MUJOCO_LIBRARY_GLOB)
            list(GET MUJOCO_LIBRARY_GLOB 0 MUJOCO_LIBRARY)
        endif()
    endif()

    if(NOT MUJOCO_LIBRARY)
        find_library(MUJOCO_LIBRARY NAMES mujoco)
    endif()

    if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
        message(STATUS "MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
        message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")
    else()
        message(WARNING "MuJoCo headers or library not found. Include: ${MUJOCO_INCLUDE_DIR}, Lib: ${MUJOCO_LIBRARY}")
    endif()
else()
    message(WARNING "mujoco Python package not found, trying system paths")
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h)
    find_library(MUJOCO_LIBRARY NAMES mujoco)
endif()

# --- Build extension module ---
nanobind_add_module(_trlc_dk1_rt
    bindings.cpp
    serial_port.cpp
    rt_utils.cpp
    gravity_comp.cpp
    control_loop.cpp
)

target_include_directories(_trlc_dk1_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# MuJoCo
if(MUJOCO_INCLUDE_DIR)
    target_include_directories(_trlc_dk1_rt PRIVATE ${MUJOCO_INCLUDE_DIR})
endif()
if(MUJOCO_LIBRARY)
    target_link_libraries(_trlc_dk1_rt PRIVATE ${MUJOCO_LIBRARY})
endif()

# pthreads
find_package(Threads REQUIRED)
target_link_libraries(_trlc_dk1_rt PRIVATE Threads::Threads)

# Set RPATH so the extension can find libmujoco in the mujoco sibling package
if(APPLE)
    set_target_properties(_trlc_dk1_rt PROPERTIES
        INSTALL_RPATH "@loader_path/../mujoco"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    set_target_properties(_trlc_dk1_rt PROPERTIES
        INSTALL_RPATH "$ORIGIN/../mujoco"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Install the module into the Python package
install(TARGETS _trlc_dk1_rt LIBRARY DESTINATION trlc_dk1_control)

# --- Unit tests (optional) ---
option(BUILD_TESTS "Build C++ unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_dm_protocol tests/test_dm_protocol.cpp)
    target_include_directories(test_dm_protocol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_dm_protocol COMMAND test_dm_protocol)

    add_executable(test_perf_counters tests/test_perf_counters.cpp)
    target_include_directories(test_perf_counters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_perf_counters PRIVATE Threads::Threads)
    add_test(NAME test_perf_counters COMMAND test_perf_counters)
endif()
