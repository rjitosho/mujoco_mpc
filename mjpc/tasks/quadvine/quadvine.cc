// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/quadvine/quadvine.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Quadvine::XmlPath() const {
  return GetModelPath("quadvine/task.xml");
}
std::string Quadvine::Name() const { return "Quadvine"; }

// --------------- Residuals for quadvine task ---------------
//   Number of residuals: 5
//     Residual (0): position - goal position
//     Residual (1): orientation - goal orientation
//     Residual (2): linear velocity - goal linear velocity
//     Residual (3): angular velocity - goal angular velocity
//     Residual (4): control
//   Number of parameters: 6
// ------------------------------------------------------------
void Quadvine::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residuals) const {
  // ---------- Residual (0) ----------
  double* position = SensorByName(model, data, "position");
  mju_sub(residuals, position, data->mocap_pos, 3);

  // ---------- Residual (1) ----------
  double quadvine_mat[9];
  double* orientation = SensorByName(model, data, "orientation");
  mju_quat2Mat(quadvine_mat, orientation);

  double goal_mat[9];
  mju_quat2Mat(goal_mat, data->mocap_quat);

  mju_sub(residuals + 3, quadvine_mat, goal_mat, 9);

  // ---------- Residual (2) ----------
  double* linear_velocity = SensorByName(model, data, "linear_velocity");
  mju_sub(residuals + 12, linear_velocity, parameters_.data(), 3);

  // ---------- Residual (3) ----------
  double* angular_velocity = SensorByName(model, data, "angular_velocity");
  mju_sub(residuals + 15, angular_velocity, parameters_.data() + 3, 3);

  // ---------- Residual (4) ----------
  mju_copy(residuals + 18, data->ctrl, model->nu);
}

// ----- Transition for quadvine task -----
void Quadvine::TransitionLocked(mjModel* model, mjData* data) {
  // set mode to GUI selection
  if (mode > 0) {
    current_mode_ = mode - 1;
  } else {
    // goal position
    const double* goal_position = data->mocap_pos;

    // goal orientation
    const double* goal_orientation = data->mocap_quat;

    // system's position
    double* position = SensorByName(model, data, "position");

    // system's orientation
    double* orientation = SensorByName(model, data, "orientation");

    // position error
    double position_error[3];
    mju_sub3(position_error, position, goal_position);
    double position_error_norm = mju_norm3(position_error);

    // orientation error
    double geodesic_distance =
        1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

    double tolerance = 5.0e-2;
    if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
      std::cout << "goal reached with errors: " << position_error_norm << ", " << geodesic_distance << std::endl;
      // update task state
      current_mode_ += 1;
      if (current_mode_ == model->nkey) {
        current_mode_ = 0;
      }
    }
  }

  // set goal
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * current_mode_);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * current_mode_);
}

}  // namespace mjpc
