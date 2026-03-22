#include "gravity_comp.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace trlc {

// Read file contents into a string.
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Strip all <visual>...</visual> and <collision>...</collision> elements
// from a URDF.  Gravity compensation only needs the kinematic chain
// (link masses, inertias, joint axes) — no meshes of any kind.
static std::string strip_geometry_elements(const std::string& xml) {
    std::string result = xml;
    for (const char* tag : {"<visual>", "<collision>"}) {
        std::string open_tag(tag);
        std::string close_tag = "</" + open_tag.substr(1);  // </visual> or </collision>
        while (true) {
            auto start = result.find(open_tag);
            if (start == std::string::npos) break;
            auto end = result.find(close_tag, start);
            if (end == std::string::npos) break;
            result.erase(start, end + close_tag.size() - start);
        }
    }
    return result;
}

GravityCompensator::~GravityCompensator() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

bool GravityCompensator::load(const std::string& path, int ndof) {
    ndof_ = ndof;
    char error[1000] = {};

    // Read the URDF/XML and strip <visual> and <collision> elements so
    // MuJoCo never tries to load any mesh files (STL, OBJ, GLB).
    std::string xml = read_file(path);
    if (xml.empty()) {
        std::fprintf(stderr, "gravity_comp: cannot read file: %s\n", path.c_str());
        return false;
    }
    xml = strip_geometry_elements(xml);

    // Load from the modified XML string.  We use a VFS with the model
    // registered as a file so mj_loadXML can resolve relative paths for
    // any non-mesh assets.  The meshdir is set to the URDF's directory.
    mjVFS vfs;
    mj_defaultVFS(&vfs);

    const char* vfs_name = "model.urdf";
    if (mj_addBufferVFS(&vfs, vfs_name, xml.data(),
                         static_cast<int>(xml.size())) != 0) {
        std::fprintf(stderr, "gravity_comp: mj_addBufferVFS failed\n");
        mj_deleteVFS(&vfs);
        return false;
    }

    model_ = mj_loadXML(vfs_name, &vfs, error, sizeof(error));
    mj_deleteVFS(&vfs);

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
