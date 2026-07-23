#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <map>
#include <algorithm>
#include <limits>

using namespace std;

struct InsKinematicsResult
{
    std::vector<Eigen::Vector3d> r_A;
    std::vector<Eigen::Vector3d> r_B;
    std::vector<Eigen::Vector3d> r_C;
    std::vector<Eigen::Vector3d> r_bar;
    std::vector<Eigen::Vector3d> r_rod;
    Eigen::Vector2d THETA;
};

struct ForwardMappingResult
{
    int count;
    Eigen::Vector2d ankle_joint_ori;
    std::vector<Eigen::MatrixXd> Jac;
};

class Decouple
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Decouple() {}
    void print_vector3d(const Eigen::Vector3d &vec);

    void print_kinematics_result(const InsKinematicsResult &result);

    std::vector<Eigen::Vector3d> sphere_circle_intersections(Eigen::Vector3d &sphere_center,
        double &sphere_radius, Eigen::Vector3d &circle_center, double &circle_radius);
    InsKinematicsResult inverse_kinematics(double q_roll, double q_pitch, bool leftLegFlag);

    std::vector<Eigen::MatrixXd> jacobian(const std::vector<Eigen::Vector3d> &r_C, const std::vector<Eigen::Vector3d> &r_bar,
                                          const std::vector<Eigen::Vector3d> &r_rod, double q_pitch);

    std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>> get_decouple(double roll, double pitch, bool leftLegFlag);

    ForwardMappingResult forward_kinematics(const Eigen::Vector2d &thetaRef, bool leftLegFlag);

    bool get_decoupleQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag);
    void get_forwardQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag);

    bool isInsidePolygon(const Eigen::Vector2d &p, bool leftLegFlag);
    Eigen::Vector2d closestPointOnSegment(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b);
    Eigen::Vector2d closestPointOnPolygon(const Eigen::Vector2d& p, bool leftLegFlag);
    void setAnkleAngleRange(const std::vector<Eigen::Vector2d> &leftLegRange, const std::vector<Eigen::Vector2d> &rightLegRange);

    std::vector<Eigen::Vector2d> ankleAngleRange[2]; // 0: left, 1: right

    std::map<bool, Eigen::Vector2d> last_solution_;
    double l_bar = 0.040;

    double l_rod[2] = {0.165, 0.095};     // upper rod, lower rod
    double motor_x[2] = {-0.015, -0.015};
    double motor_y[2] = {0.0443, 0.0443};
    double motor_z[2] = {0.1543, 0.0893};
    double conn_x[2] = {-0.0456, 0.0456};
    double conn_y[2] = {0.0467, 0.0467};
    double conn_z[2] = {-0.008, -0.008};
};