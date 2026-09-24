# app/CMakeLists.txt — native library for the panda-cube-sorter app
cmake_minimum_required(VERSION 3.22)
project(panda_sorter CXX C)
message(STATUS "PCS-SRC-DIR: ${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# prebuilt MuJoCo (arm64-v8a), cross-compiled with -O3 -ffp-contract=fast
add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libmujoco.so)

# CMAKE_SOURCE_DIR = app/src/main/cpp; repo root is 4 levels up
include_directories(
    ${CMAKE_SOURCE_DIR}/../../../..
    ${CMAKE_SOURCE_DIR}/../../../../native/include
    ${CMAKE_SOURCE_DIR}/third_party/mujoco/include
)

add_library(panda_sorter SHARED
    native_main.cpp
    gles_renderer.cpp
    ui_font.cpp
    ../../../../native/src/rl_policy.cpp
    ../../../../native/src/event_camera.cpp
    ../../../../native/src/alif_lsnn.cpp
    ../../../../native/src/pred_coder.cpp
    ../../../../native/src/soft_moe.cpp
    ../../../../native/src/kan_mlp.cpp
    ../../../../native/src/lora_lyap.cpp
    ../../../../native/src/proto_bank.cpp
    ../../../../native/src/task_layer.cpp
    ../../../../native/src/controller.cpp
    ../../../../native/src/weights.cpp
    ../../../../glue/sim_glue.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(egl-lib EGL)
find_library(gles-lib GLESv3)

target_link_libraries(panda_sorter
    mujoco
    ${log-lib} ${android-lib} ${egl-lib} ${gles-lib}
)

# 16 KB kernel page support (mandatory on new Samsung/Qualcomm flagships):
# without this the dynamic linker refuses to load the library -> black screen
target_link_options(panda_sorter PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)

#骁龙 8 Elite (S26 Ultra): performance cores only for this library
set_property(TARGET panda_sorter APPEND_STRING PROPERTY COMPILE_FLAGS " -O3 -ffp-contract=fast")
