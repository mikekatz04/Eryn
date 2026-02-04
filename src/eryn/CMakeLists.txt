# Add subdirectories which contain compiled sources

# Handle Git Metadata
get_target_property(_ERYN_WITH_CPU eryn WITH_CPU)
get_target_property(_ERYN_WITH_GPU eryn WITH_GPU)

message(CHECK_START "Building eryn.git_version metadata")
message(CHECK_START "Find Git executable")
find_package(Git)
if(Git_FOUND)
  message(CHECK_PASS "found")
  message(CHECK_START "Checking if in Git worktree")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_QUIET ERROR_QUIET
    RESULT_VARIABLE ERYN_GIT_IN_WORKTREE_CODE)
  if(ERYN_GIT_IN_WORKTREE_CODE EQUAL "0")
    message(CHECK_PASS "yes")

    execute_process(
      COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ERYN_GIT_COMMIT_ID
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ERYN_GIT_COMMIT_SHORT_ID
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    configure_file(git_version.py.in git_version.py @ONLY)

    if(_ERYN_WITH_CPU)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/git_version.py DESTINATION eryn)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/git_version.py
              DESTINATION eryn_backend_cpu)
    endif()
    if(_ERYN_WITH_GPU)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/git_version.py
              DESTINATION eryn_backend_cuda${CUDAToolkit_VERSION_MAJOR}x)
    endif()

    message(CHECK_PASS "done")
  else()
    message(CHECK_FAIL "no")
  endif()
else()
  message(CHECK_FAIL "not found")
  message(CHECK_FAIL "eryn.git_version will not be available")
endif()
