cmake_minimum_required(VERSION 3.15)
project(DNLP_Diff_Engine C)
set(CMAKE_C_STANDARD 99)

set(DIFF_ENGINE_VERSION_MAJOR 0)
set(DIFF_ENGINE_VERSION_MINOR 1)
set(DIFF_ENGINE_VERSION_PATCH 1)
set(DIFF_ENGINE_VERSION "${DIFF_ENGINE_VERSION_MAJOR}.${DIFF_ENGINE_VERSION_MINOR}.${DIFF_ENGINE_VERSION_PATCH}")
add_compile_definitions(DIFF_ENGINE_VERSION="${DIFF_ENGINE_VERSION}")

message(STATUS "Configuring DNLP Differentiation Engine (version ${DIFF_ENGINE_VERSION})")


#Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# Warning flags (compiler-specific)
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(
        -Wall                  # Enable most warnings
        -Wextra                # Extra warnings
        -Wpedantic             # Strict ISO C compliance
        -Wshadow               # Warn about variable shadowing
        -Wformat=2             # Extra format string checks
        -Wcast-qual            # Warn about cast that removes qualifiers
        -Wcast-align           # Warn about pointer cast alignment issues
        -Wunused               # Warn about unused variables/functions
        -Wdouble-promotion     # Warn about float to double promotion
        -Wnull-dereference     # Warn about null pointer dereference
    )
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files - automatically gather all .c files from src/
file(GLOB_RECURSE SOURCES "src/*.c")


# Create core library
add_library(dnlp_diff ${SOURCES})

# Link math library (Unix/Linux only)
if(NOT MSVC)
    target_link_libraries(dnlp_diff m)
endif()

# Config-specific compile options (compiler-specific)
if(MSVC)
    target_compile_options(dnlp_diff PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /DNDEBUG>
        $<$<CONFIG:MinSizeRel>:/Os /DNDEBUG>
    )
else()
    target_compile_options(dnlp_diff PRIVATE
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
        $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    )
endif()

# This is needed for clock_gettime on Linux without compiler extensions
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# Enable position-independent code for shared library compatibility
set_property(TARGET dnlp_diff PROPERTY POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
if(NOT SKBUILD)
    include_directories(${PROJECT_SOURCE_DIR}/tests)
    enable_testing()

    add_executable(all_tests
        tests/all_tests.c
        tests/test_helpers.c
    )
    target_link_libraries(all_tests dnlp_diff)
    add_test(NAME AllTests COMMAND all_tests)
endif()
