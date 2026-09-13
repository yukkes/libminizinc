/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */

/*
 *  Main authors:
 *     Jip J. Dekker <jip.dekker@monash.edu>
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <minizinc/file_utils.hh>
#include <minizinc/plugin.hh>
#include <minizinc/solvers/MIP/MIP_highs_wrap.hh>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace MiniZinc;

namespace {
using AddLinearObjectiveFn = HighsInt (*)(const void*, double, double, const double*, double, double,
                                          HighsInt);
using ClearLinearObjectivesFn = HighsInt (*)(const void*);

class HiGHSMultiObjectiveApi {
public:
  explicit HiGHSMultiObjectiveApi(const std::string& dll) {
#ifdef HIGHS_PLUGIN
    if (dll.empty()) {
      _plugin = std::unique_ptr<Plugin>(new Plugin(std::vector<std::string>{
#ifdef _WIN32
          FileUtils::progpath() + "\\bin\\highs.dll", "highs"
#elif __APPLE__
          FileUtils::progpath() + "/lib/libhighs.dylib", "libhighs"
#else
          FileUtils::progpath() + "/../lib/libhighs.so", "libhighs"
#endif
      }));
    } else {
      _plugin = std::unique_ptr<Plugin>(new Plugin(dll));
    }
    *(void**)(&_addLinearObjective) = _plugin->symbol("Highs_addLinearObjective");
    *(void**)(&_clearLinearObjectives) = _plugin->symbol("Highs_clearLinearObjectives");
#else
    _addLinearObjective = ::Highs_addLinearObjective;
    _clearLinearObjectives = ::Highs_clearLinearObjectives;
#endif
  }

  HighsInt addLinearObjective(const void* highs, double weight, double offset,
                              const double* coefficients, double absTolerance, double relTolerance,
                              HighsInt priority) const {
    return _addLinearObjective(highs, weight, offset, coefficients, absTolerance, relTolerance,
                               priority);
  }

  HighsInt clearLinearObjectives(const void* highs) const { return _clearLinearObjectives(highs); }

private:
#ifdef HIGHS_PLUGIN
  std::unique_ptr<Plugin> _plugin;
#endif
  AddLinearObjectiveFn _addLinearObjective = nullptr;
  ClearLinearObjectivesFn _clearLinearObjectives = nullptr;
};
}  // namespace

bool MIPHiGHSWrapper::defineMultipleObjectives(const MultipleObjectives& mo) {
  if (mo.size() == 0) {
    return true;
  }

  // Match the existing Gurobi/Xpress convention: use one global maximize sense and encode each
  // goal's MIN/MAX direction in its +/-1 weight. Distinct descending priorities make HiGHS run
  // the goals lexicographically rather than blending them.
  setObjSense(1);
  checkHiGHSReturn(_plugin->Highs_setBoolOptionValue(_highs, "blend_multi_objectives", 0),
                   "unable to enable lexicographic multi-objective optimization");

  HiGHSMultiObjectiveApi api(_factoryOptions.highsDll);
  checkHiGHSReturn(api.clearLinearObjectives(_highs), "unable to clear linear objectives");

  const int nCols = getNCols();
  std::vector<double> coefficients(static_cast<size_t>(nCols), 0.0);
  for (size_t iobj = 0; iobj < mo.size(); ++iobj) {
    std::fill(coefficients.begin(), coefficients.end(), 0.0);
    const auto& obj = mo.getObjectives()[iobj];
    const int objVar = obj.getVariable();
    if (objVar < 0 || objVar >= nCols) {
      throw InternalError("HiGHS multi-objective variable index out of range");
    }
    coefficients[static_cast<size_t>(objVar)] = 1.0;
    const auto priority = static_cast<HighsInt>(mo.size() - iobj);
    checkHiGHSReturn(api.addLinearObjective(_highs, obj.getWeight(), 0.0, coefficients.data(), 0.0,
                                            0.0, priority),
                     "unable to define multi-objective goal " + std::to_string(iobj));
  }
  return true;
}
