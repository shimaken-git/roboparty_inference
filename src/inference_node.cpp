#include "inference_node.hpp"
// BEGIN ankle frequency response test
#include "utils/close_chain_mapping.hpp"
#include <iomanip>
#include <limits>

namespace {
constexpr uint64_t ankle_settle_ns = 3'000'000'000ULL;
constexpr uint64_t ankle_freshness_ns = 50'000'000ULL;

bool fresh_ankle_sample(uint64_t now, uint64_t stamp) {
    return stamp != 0 && now >= stamp && now - stamp <= ankle_freshness_ns;
}
}
// END ankle frequency response test

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    if (stack_order == ObsStackOrder::FrameMajor) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin(), obs.end(), input_buffer.begin() + frame * obs_num);
            }
        } else {
            std::move(input_buffer.begin() + obs_num,
                      input_buffer.begin() + frame_stack * obs_num,
                      input_buffer.begin());
            std::copy(obs.begin(), obs.end(), input_buffer.begin() + (frame_stack - 1) * obs_num);
        }
        return;
    }

    int input_offset = 0;
    int obs_offset = 0;

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset += field_size;
    }
}

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size){
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " + std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    // BEGIN ankle frequency response test
    stop_ankle_test("runtime reset");
    ankle_toggle_.store(false);
    // END ankle frequency response test
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    is_test_.store(false);
    is_teleop_.store(false);
    active_policy_idx_ = 0;
    mixing_ratio = 0.0f;
    mixing_switch_period = 0.01f;
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
        std::fill(last_cmd_vel_.begin(), last_cmd_vel_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (int i = 0; i < joint_num_; i++) {
            act_[i] = static_cast<float>(joint_default_angle_[i]);
            last_act_[i] = static_cast<float>(joint_default_angle_[i]);
        }
    }
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    for (PolicyRuntime& policy : policies_) {
        reset_policy_runtime(policy);
    }
}

InferenceNode::PolicyRuntime& InferenceNode::active_policy() {
    return policies_[active_policy_idx_];
}

const InferenceNode::PolicyRuntime& InferenceNode::active_policy() const {
    return policies_[active_policy_idx_];
}

void InferenceNode::initialize_runtime_state() {
    active_policy_idx_ = 0;

    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i] = "action_" + std::to_string(i + 1);
    }

    cmd_vel_.assign(3, 0.0f);
    last_cmd_vel_.assign(3, 0.0f);
    act_.assign(joint_num_, 0.0f);
    last_act_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    teleop_action_.assign(10, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);
    } else {
        interrupt_action_.clear();
    }
}

bool InferenceNode::has_motion_policy() const {
    return !motion_policy_indices_.empty();
}

bool InferenceNode::supports_interrupt() const {
    return !interrupt_action_.empty();
}

void InferenceNode::reset_policy_runtime(PolicyRuntime& policy) {
    std::fill(policy.obs.begin(), policy.obs.end(), 0.0f);
    for (auto& segment : policy.obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    for (auto& segment : policy.extra_obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    if (policy.ctx) {
        std::fill(policy.ctx->input_buffer.begin(), policy.ctx->input_buffer.end(), 0.0f);
        std::fill(policy.ctx->output_buffer.begin(), policy.ctx->output_buffer.end(), 0.0f);
    }
    policy.motion_frame = 0;
    policy.is_first_frame = true;
}

void InferenceNode::apply_action() {
    if((!is_running_.load() && !is_test_.load()) || !robot_->is_init_.load()){
        return;
    }
    if(is_running_.load()){
        {
            std::unique_lock<std::mutex> lock(act_mutex_);
            for (size_t i = 0; i < act_.size(); i++) {
                last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
            }
            // BEGIN ankle frequency response test
            if (ankle_test_active_.load()) {
                const auto imu = robot_->read_imu_sample();
                const auto now = monotonic_ns();
                const auto begin = ankle_phase_start_ + ankle_settle_ns;
                const Eigen::Quaternionf q = imu_correct_quat_ * imu.orientation;
                const auto up = q * Eigen::Vector3f::UnitZ();
                const double gyro_y = imu.angular_velocity.y();
                if (!fresh_ankle_sample(now, ankle_policy_stamp_) ||
                    !fresh_ankle_sample(now, imu.timestamp_ns) || !(up.z() > std::cos(0.30)) ||
                    !(std::abs(gyro_y) < 1.5) || is_test_.load() ||
                    is_motion_policy_.load() || is_interrupt_.load() || is_teleop_.load()) {
                    stop_ankle_test_locked("posture, gyro, stale data, or mode changed");
                } else if (now >= begin && (now - begin) * 1e-9 < ankle_test_duration_s_) {
                    auto command = last_act_;
                    AnkleSample sample{};
                    sample.stamp = now;
                    sample.imu_stamp = imu.timestamp_ns;
                    sample.run = ankle_run_;
                    sample.phase_start = begin;
                    sample.frequency = ankle_frequencies_[ankle_frequency_index_];
                    sample.input = ankle_test_amplitude_ * std::sin(
                        6.283185307179586 * sample.frequency * ((now - begin) * 1e-9));
                    const auto forward = q * Eigen::Vector3f::UnitX();
                    sample.pitch = std::atan2(-forward.z(), std::hypot(forward.x(), forward.y()));
                    sample.gyro_y = gyro_y;
                    const auto positions = robot_->get_joint_q();
                    const auto velocities = robot_->get_joint_vel();
                    const auto torques = robot_->get_joint_tau();
                    const auto timestamps = robot_->get_joint_time();
                    bool feedback_ok = true;
                    for (size_t i = 0; i < ankle_motors_.size(); ++i) {
                        const size_t motor = ankle_motors_[i];
                        sample.motor_position[i] = positions[motor];
                        sample.motor_velocity[i] = velocities[motor];
                        sample.motor_torque[i] = torques[motor];
                        sample.motor_stamps[i] = timestamps[motor];
                        feedback_ok = feedback_ok && fresh_ankle_sample(now, timestamps[motor]) &&
                            std::isfinite(positions[motor]) && std::abs(velocities[motor]) < 3.0f &&
                            std::isfinite(torques[motor]);
                    }
                    command[ankle_test_joints_[0]] += sample.input;
                    command[ankle_test_joints_[1]] += sample.input;
                    for (size_t i = 0; i < ankle_test_joints_.size(); ++i) {
                        const size_t joint = ankle_test_joints_[i];
                        sample.policy[i] = act_[joint];
                        sample.command[i] = command[joint];
                        sample.position[i] = positions[joint];
                        feedback_ok = feedback_ok && std::isfinite(command[joint]);
                    }
                    if (feedback_ok && ankle_samples_.size() < ankle_samples_.capacity() &&
                        ankle_test_active_.load() && is_running_.load()) {
                        if (!ankle_measurement_started_) {
                            ankle_measurement_started_ = true;
                            RCLCPP_INFO(get_logger(), "Ankle sweep [%zu/%zu] %.3g Hz started (%.3g s)",
                                ankle_frequency_index_ + 1, ankle_frequencies_.size(),
                                sample.frequency, ankle_test_duration_s_);
                        }
                        ankle_samples_.push_back(sample);
                        lock.unlock();
                        robot_->apply_action(std::move(command));
                        return;
                    }
                    stop_ankle_test_locked("feedback invalid, motor speed limit, or log buffer full");
                }
            }
            // END ankle frequency response test
        }
        robot_->apply_action(last_act_);
    } else if(is_test_.load()){
        robot_->test_action();
        publish_joint_states();
        publish_imu();
    }
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    auto next_deadline = std::chrono::steady_clock::now();
    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        next_deadline += period;
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        if (loop_end < next_deadline) {
            std::this_thread::sleep_until(next_deadline);
        }
    }
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));
    auto next_deadline = std::chrono::steady_clock::now();
    // PLL parameters
    constexpr double target_age_ms = 2.0;
    constexpr double kp = 0.1;
    constexpr double max_correction_ms = 0.2;
    double correction_ms = 0.0;
    uint64_t previous_start_time = 0;

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        // BEGIN ankle frequency response test
        if(!is_running_.load() || !robot_->is_init_.load()){
            stop_ankle_test("inference paused or motors deinitialized");
            ankle_toggle_.store(false);
            next_deadline = loop_start + period;
            std::this_thread::sleep_until(next_deadline);
            continue;
        }
        // END ankle frequency response test

        try {
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            auto& policy = active_policy();
            // BEGIN ankle frequency response test
            const bool toggle_ankles = ankle_toggle_.exchange(false);
            bool starting_ankle_test = false;
            if (toggle_ankles || ankle_test_active_.load()) {
                bool neutral;
                {
                    std::scoped_lock lock(cmd_mutex_);
                    neutral = std::all_of(cmd_vel_.begin(), cmd_vel_.end(),
                        [](float v) { return std::abs(v) <= 0.03f; });
                }
                const bool ready = neutral && is_running_.load() && !is_test_.load() &&
                    !is_teleop_.load() && !is_motion_policy_.load() && !is_interrupt_.load() &&
                    mixing_ratio == 0.0f && joint_num_ == 23 && decimation_ == 5 &&
                    std::abs(dt_ - 0.003f) < 1e-6f && !ankle_log_failed_.load();
                std::scoped_lock lock(act_mutex_);
                const auto now = monotonic_ns();
                if (ankle_test_active_.load()) {
                    if (toggle_ankles || !ready) {
                        stop_ankle_test_locked(toggle_ankles ? "R1 + D-pad left" : "standing conditions changed");
                    } else if (now - ankle_phase_start_ >= ankle_settle_ns &&
                        (now - ankle_phase_start_ - ankle_settle_ns) * 1e-9 >= ankle_test_duration_s_) {
                        finish_ankle_frequency_locked("completed");
                        if (++ankle_frequency_index_ == ankle_frequencies_.size()) {
                            stop_ankle_test_locked(nullptr);
                        } else {
                            ankle_phase_start_ = now;
                            RCLCPP_INFO(get_logger(), "Ankle sweep: settling before %.3g Hz",
                                ankle_frequencies_[ankle_frequency_index_]);
                        }
                    }
                } else if (toggle_ankles && ready) {
                    ankle_frequency_index_ = 0;
                    ankle_phase_start_ = now;
                    ankle_run_ = now;
                    starting_ankle_test = true;
                    // Activate after this tick has captured fresh posture and policy output.
                } else if (toggle_ankles) {
                    RCLCPP_WARN(get_logger(), "Ankle sweep start rejected: check standing mode, neutral stick, timing, and log");
                }
            }
            // END ankle frequency response test
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            auto joint_times_ = robot_->get_joint_time();
            publish_imu();
            publish_joint_states();
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            if(obs_file.is_open()){
                obs_file << robot_->imu_timestamp_.load() << ",";
                for(auto tm : joint_times_){
                    obs_file << tm << ",";
                }
                for(int i = 0; i < policy.obs.size(); i++){
                    // if(i < policy.obs.size() - 1){
                        obs_file << policy.obs[i] << ",";
                    // }else{
                        // obs_file << policy.obs[i] << std::endl;
                    // }
                }
            }

            update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num, policy.frame_stack,
                               policy.stack_order, policy.obs_layout_sizes, policy.is_first_frame);
            if(policy.extra_obs_num > 0){
                update_obs_segments(policy.extra_obs_segments, policy.extra_obs_layout);
                flatten_obs_segments(policy.extra_obs_segments, policy.ctx->input_buffer.begin() + policy.frame_stack * policy.obs_num);
            }
            if (policy.motion_loader) {
                step_motion_frame();
            }
            policy.is_first_frame = false;

            auto start_time = monotonic_ns();

            policy.ctx->session->Run(Ort::RunOptions{nullptr},
                policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(), policy.ctx->num_inputs,
                policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(), policy.ctx->num_outputs);

            auto end_time = monotonic_ns();

            // auto age = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::nanoseconds(start_time - joint_times_[0]));
            // double age_ms = static_cast<double>(age.count()) / 1000.0;
            auto age_ns = static_cast<int64_t>(start_time) - static_cast<int64_t>(joint_times_[5]);
            double age_ms = age_ns * 1e-6;
            double error_ms = age_ms - target_age_ms;
            correction_ms = std::clamp(kp * error_ms, -max_correction_ms, max_correction_ms);
            uint64_t inference_interval_ns = 0;
            if(previous_start_time != 0){
                inference_interval_ns = start_time - previous_start_time;
            }
            previous_start_time = start_time;

            if(obs_file.is_open()){
                obs_file << age_ms << "," << error_ms << "," << correction_ms << "," << inference_interval_ns << ",";
                obs_file << start_time << "," << end_time << std::endl;
            } 
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (int i = 0; i < policy.ctx->output_buffer.size(); i++) {
                    policy.ctx->output_buffer[i] = std::clamp(policy.ctx->output_buffer[i], -clip_actions_, clip_actions_);
                    act_[usd2urdf_[i]] = policy.ctx->output_buffer[i];
                    act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_ + joint_default_angle_[usd2urdf_[i]];
                }
                if(act_file.is_open()){
                    for(int i = 0; i < act_.size(); i++){
                        if(i < act_.size() - 1){
                            act_file << act_[i] << ",";
                        }else{
                            act_file << act_[i] << std::endl;
                        }
                    }
                }
                // if(supports_interrupt() && is_interrupt_.load()){
                //     std::unique_lock<std::mutex> lock(interrupt_mutex_);
                //     for (size_t i = 0; i < interrupt_action_.size(); i++) {
                //         act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                //     }
                // }
                if(supports_interrupt()){
                    std::unique_lock<std::mutex> lock(interrupt_mutex_);
                    if(is_interrupt_.load()){
                        if(mixing_ratio < 1.0){
                            mixing_ratio += mixing_switch_period;
                            std::cout << "mixing_ratio: " << mixing_ratio << std::endl;
                            if(mixing_ratio > 1.0){
                                mixing_ratio = 1.0;
                            }
                        }
                    }else{
                        if(mixing_ratio > 0.0){
                            mixing_ratio -= mixing_switch_period;
                            std::cout << "mixing_ratio: " << mixing_ratio << std::endl;
                            if(mixing_ratio < 0.0){
                                mixing_ratio = 0.0;
                            }
                        }
                    }
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = act_[act_.size() - interrupt_action_.size() + i] * (1.0 - mixing_ratio) + interrupt_action_[i] * mixing_ratio;
                    }
                }
                //ノーマルポリシーで割り込みさせるケース
                // {
                //     std::unique_lock<std::mutex> lock(teleop_mutex_);
                //     if(is_teleop_.load()){
                //         if(mixing_ratio < 1.0){
                //             mixing_ratio += mixing_switch_period;
                //             if(mixing_ratio > 1.0){
                //                 mixing_ratio = 1.0;
                //             }
                //         }
                //     }else{
                //         if(mixing_ratio > 0.0){
                //             mixing_ratio -= mixing_switch_period;
                //             if(mixing_ratio < 0.0){
                //                 mixing_ratio = 0.0;
                //             }
                //         }
                //     }
                //     for (size_t i = 0; i < teleop_action_.size(); i++) {
                //         act_[act_.size() - teleop_action_.size() + i] = act_[act_.size() - teleop_action_.size() + i] * (1.0 - mixing_ratio) + teleop_action_[i] * mixing_ratio;
                //     }
                // }
                // BEGIN ankle frequency response test
                if (ankle_test_active_.load() || starting_ankle_test) {
                    const Eigen::Quaternionf q = imu_correct_quat_ *
                        Eigen::Quaternionf(quat_buffer_[0], quat_buffer_[1], quat_buffer_[2], quat_buffer_[3]);
                    const auto up = q * Eigen::Vector3f::UnitZ();
                    ankle_policy_stamp_ = monotonic_ns();
                    const bool standing = up.z() > std::cos(0.30) && std::abs(ang_vel_buffer_[1]) < 1.5 &&
                        fresh_ankle_sample(ankle_policy_stamp_, robot_->imu_timestamp_.load()) &&
                        std::all_of(act_.begin(), act_.end(), [](float v) { return std::isfinite(v); });
                    if (!standing) {
                        if (starting_ankle_test) {
                            RCLCPP_WARN(get_logger(), "Ankle sweep start rejected: posture, gyro, IMU age, or policy output");
                        } else {
                            stop_ankle_test_locked("posture, gyro, IMU age, or policy output");
                        }
                    } else if (starting_ankle_test && is_running_.load()) {
                        ankle_test_active_.store(true);
                        RCLCPP_INFO(get_logger(), "Ankle sweep started: %zu frequencies, %.3g rad, %.3g s per frequency",
                            ankle_frequencies_.size(), ankle_test_amplitude_, ankle_test_duration_s_);
                        RCLCPP_INFO(get_logger(), "Ankle sweep: settling before %.3g Hz", ankle_frequencies_.front());
                    }
                }
                // END ankle frequency response test
                publish_action();
            }
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        next_deadline += period - std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::milli>(correction_ms));

        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);

        if (loop_end < next_deadline) {
            std::this_thread::sleep_until(next_deadline);
        } else {
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %lld us, but period is %lld us.", static_cast<long long>(elapsed_time.count()), static_cast<long long>(period.count()));
        }
    }
}

// BEGIN ankle frequency response test
void InferenceNode::finish_ankle_frequency_locked(const char* status) {
    if (ankle_measurement_started_) {
        RCLCPP_INFO(get_logger(), "Ankle sweep [%zu/%zu] %.3g Hz ended (%s)",
            ankle_frequency_index_ + 1, ankle_frequencies_.size(),
            ankle_frequencies_[ankle_frequency_index_], status);
        ankle_measurement_started_ = false;
    }
}

void InferenceNode::stop_ankle_test_locked(const char* reason) {
    if (!ankle_test_active_.exchange(false)) {
        return;
    }
    if (reason) {
        finish_ankle_frequency_locked("interrupted");
        RCLCPP_WARN(get_logger(), "Ankle sweep ended (interrupted): %s", reason);
    } else {
        RCLCPP_INFO(get_logger(), "Ankle sweep ended (completed)");
    }
}

void InferenceNode::stop_ankle_test(const char* reason) {
    std::scoped_lock lock(act_mutex_);
    stop_ankle_test_locked(reason);
}

void InferenceNode::write_ankle_samples() {
    {
        std::scoped_lock lock(act_mutex_);
        ankle_samples_.swap(ankle_samples_to_write_);
    }
    if (ankle_samples_to_write_.empty() || ankle_log_failed_.load()) {
        ankle_samples_to_write_.clear();
        return;
    }
    if (!ankle_file_.is_open()) {
        const std::string path = "ankle_frequency_response_" +
            std::to_string(ankle_samples_to_write_.front().run) + ".csv";
        ankle_file_.open(path);
        ankle_file_ << std::setprecision(12)
            << "run,command_time_ns,time_s,frequency_hz,test_rad,imu_time_ns,imu_gyro_y_rad_s,body_pitch_rad";
        constexpr std::array<const char*, ankle_test_joints_.size()> joints{
            "left_ankle_pitch", "right_ankle_pitch", "left_ankle_roll", "right_ankle_roll",
            "left_thigh_pitch", "right_thigh_pitch", "left_knee", "right_knee"
        };
        for (const auto* joint : joints) {
            ankle_file_ << ',' << joint << "_policy_target_rad," << joint << "_command_rad," << joint << "_angle_rad";
        }
        ankle_file_ << ",left_ankle_pitch_velocity_rad_s,right_ankle_pitch_velocity_rad_s"
            ",left_ankle_roll_velocity_rad_s,right_ankle_roll_velocity_rad_s"
            ",left_ankle_pitch_estimated_torque_nm,right_ankle_pitch_estimated_torque_nm"
            ",left_ankle_roll_estimated_torque_nm,right_ankle_roll_estimated_torque_nm"
            ",left_motor_a_time_ns,left_motor_b_time_ns,right_motor_a_time_ns,right_motor_b_time_ns\n";
        RCLCPP_INFO(get_logger(), "Ankle sweep CSV: %s", path.c_str());
    }
    Decouple mapping;
    for (const auto& s : ankle_samples_to_write_) {
        ankle_file_ << s.run << ',' << s.stamp << ',' << (s.stamp - s.phase_start) * 1e-9 << ','
            << s.frequency << ',' << s.input << ',' << s.imu_stamp << ',' << s.gyro_y << ',' << s.pitch;
        auto position = s.position;
        std::array<double, 4> velocity{}, torque{};
        for (size_t side = 0; side < 2; ++side) {
            const size_t a = 2 * side, b = a + 1;
            const Eigen::Vector2d motor(s.motor_position[a], s.motor_position[b]);
            const auto result = mapping.forward_kinematics(motor, side == 0);
            Eigen::Vector2d angle, speed, effort;
            if (result.count < 0 || result.Jac.size() < 2 || !result.ankle_joint_ori.allFinite() ||
                !result.Jac[0].allFinite() || !result.Jac[1].allFinite()) {
                angle.setConstant(std::numeric_limits<double>::quiet_NaN());
                speed = effort = angle;
            } else {
                angle = result.ankle_joint_ori;
                speed = result.Jac[0] * Eigen::Vector2d(s.motor_velocity[a], s.motor_velocity[b]);
                effort = result.Jac[1].transpose() * Eigen::Vector2d(s.motor_torque[a], s.motor_torque[b]);
            }
            for (size_t axis = 0; axis < 2; ++axis) {
                const size_t joint = axis * 2 + side; // Pitch L/R, then roll L/R.
                position[joint] = angle[axis];
                velocity[joint] = speed[axis];
                torque[joint] = effort[axis];
            }
        }
        for (size_t i = 0; i < ankle_test_joints_.size(); ++i) {
            ankle_file_ << ',' << s.policy[i] << ',' << s.command[i] << ',' << position[i];
        }
        for (double value : velocity) {
            ankle_file_ << ',' << value;
        }
        for (double value : torque) {
            ankle_file_ << ',' << value;
        }
        for (auto t : s.motor_stamps) {
            ankle_file_ << ',' << t;
        }
        ankle_file_ << '\n';
    }
    ankle_file_.flush();
    if (!ankle_file_) {
        ankle_log_failed_.store(true);
        stop_ankle_test("CSV write failed");
        RCLCPP_ERROR(get_logger(), "Ankle sweep CSV write failed");
    }
    ankle_samples_to_write_.clear();
}

// END ankle frequency response test
void InferenceNode::imu_correction_initialize() {
    imu_correct_quat_ = Eigen::Quaternionf::Identity();
    Eigen::AngleAxisf roll_correction(imu_correction_[0], Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_correction(imu_correction_[1], Eigen::Vector3f::UnitY());
    imu_correct_quat_ = roll_correction * pitch_correction;
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    struct sched_param sp{}; sp.sched_priority = 50;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Failed to set realtime priority for main thread");
        rclcpp::shutdown();
        return 1;
    }
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'OPTIONS' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'PS' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press '○' to start/pause inference");
        RCLCPP_INFO(node->get_logger(), "Press 'SHARE' to switch between Gamepad Control / cmd_vel Control");
        if (node->supports_interrupt() || node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press '□' to switch policy mode (available in beyondmimic / interrupt modes)");
        }
        if (node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'X' to switch motion sequence (available in beyondmimic mode)");
        }
        RCLCPP_INFO(node->get_logger(), "Left Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control turning (left / right rotation)");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}

uint64_t InferenceNode::monotonic_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}