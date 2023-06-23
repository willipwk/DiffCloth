# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    # GIT_TAG c1ba93d475ceb8e906e4cf44d8cf992b67235788
    GIT_TAG 67c3d8cfc3fc7f50599c221a338fc1b131c9a325
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
add_definitions( -DIPC_TOOLKIT_WITH_CORRECT_CCD)


