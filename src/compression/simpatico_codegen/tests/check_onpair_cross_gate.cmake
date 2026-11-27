# SPDX-License-Identifier: Apache-2.0
# Each reader has its own cached compression gate and reads the other writer's file.
foreach(writer IN ITEMS 4 6)
  math(EXPR reader "10 - ${writer}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_ONPAIR_K=${writer}"
            "${ONPAIR_EXE}" --write-layout "${ONPAIR_FILE}"
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_ONPAIR_K=${reader}"
            "${ONPAIR_EXE}" --read-other-layout "${ONPAIR_FILE}"
    COMMAND_ERROR_IS_FATAL ANY)
endforeach()
file(REMOVE "${ONPAIR_FILE}")
