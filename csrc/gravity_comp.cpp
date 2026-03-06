#include "gravity_comp.h"

#include <cstdio>
#include <cstring>

namespace trlc {

GravityCompensator::~GravityCompensator() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

bool GravityCompensator::load(const std::string& path, int ndof) {
    ndof_ = ndof;
    char error[1000] = {};

    // Load directly — works for XML files and URDFs with
    // <mujoco><compiler strippath="false"/></mujoco>
    model_ = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));

    if (!model_) {
        std::fprintf(stderr, "gravity_comp: mj_loadXML failed: %s\n", error);
        return false;
    }

    if (model_->nq < ndof) {
        std::fprintf(stderr, "gravity_comp: model has %ld DoFs but ndof=%d\n", (long)model_->nq, ndof);
        mj_deleteModel(model_);
        model_ = nullptr;
        return false;
    }

    data_ = mj_makeData(model_);
    std::fprintf(stderr, "gravity_comp: loaded %s (%ld DoF model, using first %d)\n",
                 path.c_str(), (long)model_->nq, ndof);
    return true;
}

void GravityCompensator::compute(const double* q, double* tau_out) {
    if (!model_ || !data_) {
        std::memset(tau_out, 0, static_cast<size_t>(ndof_) * sizeof(double));
        return;
    }

    for (int i = 0; i < ndof_; ++i) {
        data_->qpos[i] = q[i];
    }
    std::memset(data_->qvel, 0, static_cast<size_t>(model_->nv) * sizeof(double));
    std::memset(data_->qacc, 0, static_cast<size_t>(model_->nv) * sizeof(double));

    // Use mj_forward + qfrc_bias instead of mj_inverse + qfrc_inverse.
    // mj_inverse includes constraint forces from joint limits which can
    // produce wildly incorrect torques near limit boundaries.
    // qfrc_bias with qvel=0 gives pure gravity torques.
    mj_forward(model_, data_);

    for (int i = 0; i < ndof_; ++i) {
        tau_out[i] = data_->qfrc_bias[i];
    }
}

} // namespace trlc
