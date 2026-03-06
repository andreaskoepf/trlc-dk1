#pragma once

#include <string>
#include <mujoco/mujoco.h>

namespace trlc {

class GravityCompensator {
public:
    GravityCompensator() = default;
    ~GravityCompensator();

    GravityCompensator(const GravityCompensator&) = delete;
    GravityCompensator& operator=(const GravityCompensator&) = delete;

    // Load model from URDF (with mesh stripping) or MJCF XML.
    // ndof: number of arm joints to compute torques for.
    // Returns true on success.
    bool load(const std::string& path, int ndof = 6);

    // Compute gravity compensation torques.
    // q: input joint positions (at least ndof elements)
    // tau_out: output torques (at least ndof elements)
    void compute(const double* q, double* tau_out);

    int ndof() const { return ndof_; }
    bool is_loaded() const { return model_ != nullptr; }

private:
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int ndof_ = 6;
};

} // namespace trlc
