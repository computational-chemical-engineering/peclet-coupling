# Python integration tests for the CFD-DEM coupling. They import peclet.flow + peclet.dem +
# peclet.coupling, so ctest runs them with all three build trees on PYTHONPATH (host-openmp: the
# arrays are host NumPy). Registered when -DPECLET_COUPLING_TESTS=ON and the sibling flow/dem build
# dirs are provided via -DPECLET_FLOW_BUILD=.. / -DPECLET_DEM_BUILD=..
if(PECLET_COUPLING_TESTS)
  set(_pp "${CMAKE_BINARY_DIR}:${PECLET_FLOW_BUILD}:${PECLET_DEM_BUILD}")
  foreach(t terminal_velocity fixed_bed_ergun smoothing_isotropy)
    add_test(NAME ${t} COMMAND ${Python_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/test_${t}.py)
    set_tests_properties(${t} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${_pp};PYTHONUNBUFFERED=1;OMP_PROC_BIND=false")
  endforeach()
  # The drivers' input validation (python/peclet_coupling/_checks.py): a pytest file.
  add_test(NAME input_validation COMMAND ${Python_EXECUTABLE} -m pytest -q -p no:cacheprovider
           ${CMAKE_CURRENT_SOURCE_DIR}/test_input_validation.py)
  set_tests_properties(input_validation PROPERTIES
    ENVIRONMENT "PYTHONPATH=${_pp};PYTHONUNBUFFERED=1;OMP_PROC_BIND=false")
endif()
