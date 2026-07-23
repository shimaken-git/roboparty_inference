#include "close_chain_mapping.hpp"

//////********************print******************************//////

void Decouple::print_vector3d(const Eigen::Vector3d &vec)
{
    std::cout << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]";
}

void Decouple::print_kinematics_result(const InsKinematicsResult &result)
{
    std::cout << "r_A: ";
    for (const auto &vec : result.r_A)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_B: ";
    for (const auto &vec : result.r_B)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_C: ";
    for (const auto &vec : result.r_C)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_bar: ";
    for (const auto &vec : result.r_bar)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_rod: ";
    for (const auto &vec : result.r_rod)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "THETA: ";
    std::cout << result.THETA;
    std::cout << std::endl;
}
//////********************print******************************//////

//////********************inverse kinematics*****************//////
std::vector<Eigen::Vector3d>
Decouple::sphere_circle_intersections(
    Eigen::Vector3d &sphere_center, double &sphere_radius, Eigen::Vector3d &circle_center, double &circle_radius)
{
    std::vector<Eigen::Vector3d> result;
    double dy = circle_center[1] - sphere_center[1]; //dy = cy - sy
    if (fabs(dy) > sphere_radius) return result;  //空の結果＝エラー

    double r_cross = sqrt(sphere_radius * sphere_radius - dy * dy);
    Eigen::Vector2d p0 = {sphere_center[0], sphere_center[2]};
    Eigen::Vector2d p1 = {circle_center[0], circle_center[2]};
    double d = (p1 - p0).norm();

    if(d > r_cross + circle_radius) return result;  // 交点なし
    if(d < fabs(r_cross - circle_radius)) return result; // 一方が他方を内包
    if(d == 0 && r_cross == circle_radius) return result;  // 同心円

    double a = (r_cross * r_cross - circle_radius * circle_radius + d * d) / (2 * d);
    double h_sq = r_cross * r_cross - a * a;
    if(h_sq < 0) h_sq = 0;  // 数値誤差対策
    double h = sqrt(h_sq);

    Eigen::Vector2d p2 = p0 + a * (p1 - p0) / d;
    double rx = -(p1[1] - p0[1]) * (h / d);
    double rz =  (p1[0] - p0[0]) * (h / d);

    Eigen::Vector3d intersection1 = {p2[0] + rx, circle_center[1], p2[1] + rz};
    Eigen::Vector3d intersection2 = {p2[0] - rx, circle_center[1], p2[1] - rz};
    result.push_back(intersection1);
    result.push_back(intersection2);
    return result;
}

InsKinematicsResult
Decouple::inverse_kinematics(
    double q_roll,
    double q_pitch, bool leftLegFlag)
{
    InsKinematicsResult result;

    result.THETA = Eigen::Vector2d::Zero();

    double y_sign = leftLegFlag? 1.0 : -1.0;

    // Define points
    Eigen::Vector3d r_A1_0{motor_x[0], motor_y[0] * y_sign, motor_z[0]};  // upper motor axis
    Eigen::Vector3d r_B1_0{-l_bar, 0, 0};
    Eigen::Vector3d r_C1_0{conn_x[0], conn_y[0] * y_sign, conn_z[0]};

    Eigen::Vector3d r_A2_0{motor_x[1], motor_y[1] * y_sign, motor_z[1]};  // lower motor axis
    Eigen::Vector3d r_B2_0{l_bar, 0, 0};
    Eigen::Vector3d r_C2_0{conn_x[1], conn_y[1] * y_sign, conn_z[1]};

    std::vector<Eigen::Vector3d> r_A_0;
    r_A_0.push_back(r_A1_0);
    r_A_0.push_back(r_A2_0);

    std::vector<Eigen::Vector3d> r_B_0;
    r_B_0.push_back(r_B1_0);
    r_B_0.push_back(r_B2_0);

    std::vector<Eigen::Vector3d> r_C_0;
    r_C_0.push_back(r_C1_0);
    r_C_0.push_back(r_C2_0);

    // Rotation matrices
    Eigen::Matrix3d R_y = Eigen::Matrix3d::Zero();
    R_y << cos(q_pitch), 0, sin(q_pitch),
        0, 1, 0,
        -sin(q_pitch), 0, cos(q_pitch);

    Eigen::Matrix3d R_x = Eigen::Matrix3d::Zero();
    R_x << 1, 0, 0,
        0, cos(q_roll), -sin(q_roll),
        0, sin(q_roll), cos(q_roll);

    Eigen::Matrix3d x_rot = R_y * R_x;

    // Vectors to store results
    // std::vector<Eigen::Vector3d> results;

    for (int i = 0; i < 2; i++)
    {
        Eigen::Vector3d r_A_i = r_A_0[i];
        Eigen::Vector3d r_C_i = x_rot * r_C_0[i];
        Eigen::Vector3d rBA_bar = r_B_0[i];

        auto inter_sections = sphere_circle_intersections(r_C_i, l_rod[i], r_A_i, l_bar);
        Eigen::Vector3d is;
        //解説：２つの交点のうち、inside(front)はindex1の交点、outside(rear)はindex0の交点を採用
        if(inter_sections.size() != 0) is = inter_sections[i];
        else{
            std::cout << "inverse_kinematics err" << std::endl;
            std::cout << "pitch: " << q_pitch << " roll: " << q_roll << " side: " << leftLegFlag << std::endl;
            return result;
        }
        double theta_i = asin((is[2] - r_A_i[2]) / l_bar) * (i == 0 ? 1.0 : -1.0);

        Eigen::Matrix3d R_y_theta = Eigen::Matrix3d::Zero();
        R_y_theta << std::cos(theta_i), 0, std::sin(theta_i),
            0, 1, 0,
            -std::sin(theta_i), 0, std::cos(theta_i);

        Eigen::Vector3d r_B_i = r_A_i + R_y_theta * rBA_bar;
        Eigen::Vector3d r_bar_i = r_B_i - r_A_i;
        Eigen::Vector3d r_rod_i = r_C_i - r_B_i;

        // Populate results
        result.r_A.push_back(r_A_i);
        result.r_B.push_back(r_B_i);
        result.r_C.push_back(r_C_i);
        result.r_bar.push_back(r_bar_i);
        result.r_rod.push_back(r_rod_i);
        result.THETA[i] = theta_i;
    }

    return result;
}
//////********************inverse kinematics*****************//////

//////********************jacobian***************************//////
std::vector<Eigen::MatrixXd>
Decouple::jacobian(const std::vector<Eigen::Vector3d> &r_C,
                   const std::vector<Eigen::Vector3d> &r_bar,
                   const std::vector<Eigen::Vector3d> &r_rod,
                   double q_pitch)
{
    // Assuming r_C, r_bar, r_rod are vectors of Eigen::Vector3d with at least 2 elements each
    static const Eigen::Vector3d s_unit(0, 1, 0);
    
    Eigen::Matrix<double, 2, 6> J_x;
    J_x << r_rod[0].transpose(), (r_C[0].cross(r_rod[0])).transpose(),
           r_rod[1].transpose(), (r_C[1].cross(r_rod[1])).transpose();

    Eigen::Matrix2d J_theta;
    J_theta << s_unit.dot(r_bar[0].cross(r_rod[0])), 0,
               0, s_unit.dot(r_bar[1].cross(r_rod[1]));
    
    Eigen::Matrix<double, 6, 2> J_q;
    J_q << 0, 0,
           0, 0,
           0, 0,
           0, cos(q_pitch),
           1, 0,
           0, -sin(q_pitch);

    Eigen::Matrix2d J_Temp = J_x * J_q;
    
    Eigen::PartialPivLU<Eigen::Matrix2d> lu_decomp(J_Temp);
    Eigen::PartialPivLU<Eigen::Matrix2d> lu_theta(J_theta);
    
    std::vector<Eigen::MatrixXd> J_ankle(2);
    J_ankle[0] = lu_decomp.solve(J_theta);
    J_ankle[1] = lu_theta.solve(J_Temp);
    
    return J_ankle;
}
//////********************jacobian***************************//////

// from x to theta， from S to P
std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>>
Decouple::get_decouple(double roll, double pitch, bool leftLegFlag)
{
    InsKinematicsResult kinematics = inverse_kinematics(roll, pitch, leftLegFlag);
    // print_kinematics_result(kinematics);
    std::vector<Eigen::MatrixXd> Jac = jacobian(kinematics.r_C, kinematics.r_bar, kinematics.r_rod, pitch);
    return {kinematics.THETA, Jac};
}

//////********************forward kinematics*****************//////
ForwardMappingResult
Decouple::forward_kinematics(const Eigen::Vector2d &thetaRef, bool leftLegFlag)
{

    ForwardMappingResult mapping_result;

    int count = 0;
    Eigen::Vector2d f_error{10, 10};
    Eigen::Vector2d x_c_k = last_solution_.count(leftLegFlag) ? 
                            last_solution_[leftLegFlag] : 
                            Eigen::Vector2d::Zero();

    std::vector<Eigen::MatrixXd> Jac;
    static constexpr int MAX_ITERATIONS = 100;
    static constexpr double TOLERANCE = 1e-3;
    static constexpr double ALPHA = 0.5;
    /*after*/
    while (f_error.norm() > TOLERANCE && count < MAX_ITERATIONS)
    {
        InsKinematicsResult kinematics = inverse_kinematics(x_c_k[1], x_c_k[0], leftLegFlag);
        // print_kinematics_result(kinematics);

        Jac = jacobian(kinematics.r_C, kinematics.r_bar, kinematics.r_rod, x_c_k[0]);
        // std::cout << "===== count:" << count << "\n Jac: " << Jac << "\n THEAT:" << kinematics.THETA << std::endl;
        Eigen::MatrixXd J_motor2Joint = Jac[0];
        // Eigen::MatrixXd J_Joint2motor = Jac[1];
        if (J_motor2Joint.hasNaN())
        {
            std::cerr << "Decouple::forward_kinematics() Jac is nan!!" << std::endl;
            std::cerr << "  roll x_c_k[1],pitch  x_c_k[0] n!!" << x_c_k[1] << "   ---   " << x_c_k[0] << std::endl;
            mapping_result.count = -1;
            mapping_result.ankle_joint_ori = Eigen::Vector2d::Zero();
            mapping_result.Jac = Jac;
            return mapping_result; // -1 是失败的标记
        }

        f_error = thetaRef - kinematics.THETA;

        x_c_k = x_c_k + ALPHA * J_motor2Joint * f_error;
        // std::cout <<  " thetaCal: " << thetaCal << "\n f_error: " << f_error << "\n pitch_roll:" << x_c_k << std::endl;

        count++;
    }
    /*after*/

    if (f_error.norm() < TOLERANCE)
    {
        last_solution_[leftLegFlag] = x_c_k;
        // std::cout << leftLegFlag << " Converged in " << count << " iterations." << std::endl;
    }

    mapping_result.count = count;
    mapping_result.ankle_joint_ori = x_c_k;
    mapping_result.Jac = Jac;

    return mapping_result; // -1 是失败的标记
}
//////********************forward kinematics*****************//////

// from x to theta， from Serial to Parallel
// force control ,should input current pitch roll
bool Decouple::get_decoupleQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag)
{
    //　pitch, rollを受け取って、2つのモーター角度に分解する。
    double Pitch, Roll;
    Pitch = q[0]; // rotation axis [0 1 0]
    Roll = q[1];

    std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>> motor;

    motor = get_decouple(Roll, Pitch, leftLegFlag);
    q.segment<2>(0) = motor.first;
    vel.segment<2>(0) = motor.second[1] * (vel.segment<2>(0));
    tau.segment<2>(0) = motor.second[0].transpose() * (tau.segment<2>(0));
    if(motor.first[0] == 100) return false;
    else return true;
}

void Decouple::get_forwardQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag)
{
    Eigen::Vector2d motor = Eigen::Vector2d::Zero(2, 1);
    motor = q.segment<2>(0);

    ForwardMappingResult joint = forward_kinematics(motor, leftLegFlag);
    q.segment<2>(0) = joint.ankle_joint_ori;
    vel.segment<2>(0) = joint.Jac[0] * (vel.segment<2>(0));             // vel transfer from motor to ankle joint
    tau.segment<2>(0) = joint.Jac[1].transpose() * (tau.segment<2>(0)); // tau transfer from motor to ankle joint
}

inline double cross2d(const Eigen::Vector2d& a,
                      const Eigen::Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

bool Decouple::isInsidePolygon(const Eigen::Vector2d &p, bool leftLegFlag)
{
    constexpr double EPS = 1e-10;
    std::vector<Eigen::Vector2d> polygon = leftLegFlag ? ankleAngleRange[0] : ankleAngleRange[1];
    int n = polygon.size();
    bool inside = false;

    for (int i = 0; i < n; ++i)
    {
        const auto& a = polygon[i];
        const auto& b = polygon[(i + 1) % n];

        // -------------------------
        // 境界上の判定
        // -------------------------
        // double cross =
        //     (p.x() - a.x()) * (b.y() - a.y()) -
        //     (p.y() - a.y()) * (b.x() - a.x());
        double cross = cross2d(p - a, b - a);

        if (std::abs(cross) < EPS)
        {
            if (std::min(a.x(), b.x()) - EPS <= p.x() &&
                p.x() <= std::max(a.x(), b.x()) + EPS &&
                std::min(a.y(), b.y()) - EPS <= p.y() &&
                p.y() <= std::max(a.y(), b.y()) + EPS)
            {
                return true;
            }
        }

        // -------------------------
        // Ray Casting
        // -------------------------
        if ((a.y() > p.y()) != (b.y() > p.y()))
        {
            double intersectX =
                a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y());

            if (p.x() < intersectX)
                inside = !inside;
        }
    }

    return inside;
}

void Decouple::setAnkleAngleRange(const std::vector<Eigen::Vector2d> &leftLegRange, const std::vector<Eigen::Vector2d> &rightLegRange)
{
    ankleAngleRange[0] = leftLegRange;
    ankleAngleRange[1] = rightLegRange;
}

//辺への最近接点を求める関数
Eigen::Vector2d Decouple::closestPointOnSegment(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    Eigen::Vector2d ab = b - a;

    double t = (p - a).dot(ab) / ab.squaredNorm();
    t = std::clamp(t, 0.0, 1.0);

    return a + t * ab;
}

//多角形上の最近接点を求める関数
Eigen::Vector2d Decouple::closestPointOnPolygon(const Eigen::Vector2d& p, bool leftLegFlag)
{
    std::vector<Eigen::Vector2d> polygon = leftLegFlag ? ankleAngleRange[0] : ankleAngleRange[1];
    double min_dist2 = std::numeric_limits<double>::max();
    Eigen::Vector2d closest;

    int n = polygon.size();

    for (int i = 0; i < n; ++i)
    {
        const auto& a = polygon[i];
        const auto& b = polygon[(i + 1) % n];

        Eigen::Vector2d q = closestPointOnSegment(p, a, b);

        double dist2 = (p - q).squaredNorm();

        if (dist2 < min_dist2)
        {
            min_dist2 = dist2;
            closest = q;
        }
    }

    return closest;
}