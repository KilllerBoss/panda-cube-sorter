# desktop/CMakeLists.txt — host build of the validation harness
cmake_minimum_required(VERSION 3.16)
project(pcs_harness CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MUJOCO_DIR must contain include/mujoco/*.h and libmujoco.so
set(MUJOCO_DIR "" CACHE PATH "MuJoCo install dir (e.g. the pip wheel package dir)")

add_executable(pcs_harness
  src/harness_main.cpp
  ../native/src/rl_policy.cpp
  ../native/src/event_camera.cpp
  ../native/src/alif_lsnn.cpp
  ../native/src/pred_coder.cpp
  ../native/src/soft_moe.cpp
  ../native/src/kan_mlp.cpp
  ../native/src/lora_lyap.cpp
  ../native/src/proto_bank.cpp
  ../native/src/task_layer.cpp
  ../native/src/controller.cpp
  ../native/src/weights.cpp
  ../glue/sim_glue.cpp
)

target_include_directories(pcs_harness PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../native/include
  ${MUJOCO_DIR}/include
)

target_compile_options(pcs_harness PRIVATE -O3 -ffp-contract=fast)

find_library(MUJOCO_LIB NAMES mujoco libmujoco.so PATHS ${MUJOCO_DIR} PATH_SUFFIXES lib . NO_DEFAULT_PATH)
if(NOT MUJOCO_LIB)
  file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so*")
  list(GET MUJOCO_LIB 0 MUJOCO_LIB)
endif()
target_link_libraries(pcs_harness PRIVATE ${MUJOCO_LIB})
set_target_properties(pcs_harness PROPERTIES BUILD_RPATH "${MUJOCO_DIR}")
