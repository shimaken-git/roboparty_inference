#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "utils/close_chain_mapping.hpp"
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "imu_driver.hpp"
#include <fstream>

class RobotInterface {
   public:
    RobotInterface(const std::string& config_file);
    ~RobotInterface() {
        if(robot_cfg_->quit_mode_){
            relax_joints();  //モーターのトルクが入っていなければmit_cmdを送っても無視される
        }else{
            deinit_motors();
        }
        motors_.clear();
        imu_.reset();
        if (motorpos_file_.is_open()) {
            motorpos_file_.close();
        }
        if (motorcur_file_.is_open()) {
            motorcur_file_.close();
        }
    }
    struct IMUCfg{
        int imu_id_, baudrate_;
        std::string imu_type_, imu_interface_type_, imu_interface_;
    };
    struct MotorsCfg{
        int master_id_offset_;
        std::vector<std::string> motor_type_;
        std::string motor_interface_type_;
        std::vector<std::string> motor_interface_;
        std::vector<long int> motor_id_, motor_model_, motor_num_;
        std::vector<double> motor_zero_offset_;
    };
    struct RobotCfg{
        std::vector<long int> close_chain_motor_id_, motor_sign_, urdf2motor_;
        std::vector<double> ankle_limit_;
        std::vector<double> kp_, kd_, extrinsic_R_;
        bool quit_mode_ = false;  // true: deinit_motorsを呼ばない　false: deinit_motorsを呼ぶ
    };

    void apply_action(std::vector<float> action);
    void test_action();
    void init_motors();
    void deinit_motors();
    void reset_joints(std::vector<double> joint_default_angle);
    void set_zeros();
    void clear_errors();
    void refresh_joints();
    void relax_joints();
    std::vector<float> get_joint_q() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_q_;
    }
    std::vector<float> get_joint_vel() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_vel_;
    }
    std::vector<float> get_joint_tau() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_tau_;
    }
    std::vector<uint64_t> get_joint_time() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_time_;
    }
    // BEGIN ankle frequency response test
    const std::vector<float>& get_quat() {
        const auto q_body = read_body_quat();
        imu_timestamp_ = imu_->get_timestamp();
        quat_buf_[0] = q_body.w();
        quat_buf_[1] = q_body.x();
        quat_buf_[2] = q_body.y();
        quat_buf_[3] = q_body.z();
        return quat_buf_;
    }
    const std::vector<float>& get_ang_vel() {
        Eigen::Map<Eigen::Vector3f>(ang_vel_buf_.data()) = read_body_ang_vel();
        return ang_vel_buf_;
    }

    struct ImuSample {
        Eigen::Quaternionf orientation;
        Eigen::Vector3f angular_velocity;
        uint64_t timestamp_ns;
    };

    // The control thread reads into local storage while policy inference uses its own buffers.
    ImuSample read_imu_sample() const {
        const auto orientation = read_body_quat();
        const auto timestamp = imu_->get_timestamp();
        return {orientation, read_body_ang_vel(), timestamp};
    }

    // END ankle frequency response test
    std::atomic<bool> is_init_{false};
    void load_joint_limits(std::vector<double> limits);
    std::atomic<uint64_t> imu_timestamp_{0};


   private:
    std::shared_ptr<IMUCfg> imu_cfg_;
    std::shared_ptr<MotorsCfg> motors_cfg_;
    std::shared_ptr<RobotCfg> robot_cfg_;
    int offline_threshold_ = 25;
    std::shared_ptr<IMUDriver> imu_;
    std::shared_ptr<Decouple> ankle_decouple_;
    Eigen::Matrix3f extrinsic_R_mat_ = Eigen::Matrix3f::Identity();
    Eigen::Quaternionf extrinsic_q_inv_ = Eigen::Quaternionf::Identity();
    std::vector<float> quat_buf_{0.f, 0.f, 0.f, 0.f};
    std::vector<float> ang_vel_buf_{0.f, 0.f, 0.f};
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    std::unique_ptr<ThreadPool> thread_pool_;

    std::mutex motors_mutex_, joint_mutex_;
    std::vector<float> joint_q_, joint_vel_, joint_tau_, motor_target_;
    std::vector<uint64_t> joint_time_;
    std::vector<int> close_chain_motor_idx_, close_chain_joint_idx_, motor2urdf_;

    std::vector<double> joint_limits_;

    std::ofstream motorpos_file_;
    std::ofstream motorcur_file_;

    // BEGIN ankle frequency response test
    Eigen::Quaternionf read_body_quat() const {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        const auto raw = imu_->get_quat();
        return (Eigen::Quaternionf(raw[0], raw[1], raw[2], raw[3]) * extrinsic_q_inv_).normalized();
    }

    Eigen::Vector3f read_body_ang_vel() const {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        const auto raw = imu_->get_ang_vel();
        return extrinsic_R_mat_ * Eigen::Map<const Eigen::Vector3f>(raw.data());
    }

    // END ankle frequency response test
    void setup_motors();
    void setup_imu();
    void check_motors_angle();
    void check_ankle_angle(float &pitch, float &roll, bool leftLegFlag);
    void exec_motors_parallel(const std::function<void(std::shared_ptr<MotorDriver>&, int)>& cmd_func);
};