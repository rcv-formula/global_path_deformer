#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/int32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

struct Waypoint
{
  double x{0.0};
  double y{0.0};
  double v{0.0};  // input path pose.position.z is used as velocity
};

struct DeformationWindow
{
  int start{0};
  int collision_start{0};
  int collision_end{-1};
  int end{-1};

  bool active() const
  {
    return start >= 0 && end >= 0 && collision_start >= 0 && collision_end >= 0;
  }
};

struct PathDeformation
{
  DeformationWindow window;
  double shift{0.0};

  bool active() const
  {
    return window.active() && std::abs(shift) > 1e-6;
  }
};

struct ScanObstacle
{
  double x{0.0};
  double y{0.0};
};

struct DynamicPoint
{
  double x{0.0};
  double y{0.0};
  double range{0.0};
  uint32_t track_id{0};
  double relative_speed{0.0};
  double relative_yaw{0.0};
};

struct ProcessProfile
{
  double range_ms{0.0};
  double hold_check_ms{0.0};
  double release_check_ms{0.0};
  double detect_collision_ms{0.0};
  double cluster_ms{0.0};
  double shift_sign_ms{0.0};
  double find_safe_shift_ms{0.0};
  double partial_update_ms{0.0};
  double merge_ms{0.0};
  double smooth_ms{0.0};
  double make_path_ms{0.0};
  double publish_ms{0.0};
};

struct ProfileCounters
{
  std::atomic<uint64_t> estimate_shift_sign_calls{0};
  std::atomic<uint64_t> estimate_side_clearance_calls{0};
  std::atomic<uint64_t> safe_shift_calls{0};
  std::atomic<uint64_t> safe_shift_successes{0};
  std::atomic<uint64_t> safe_shift_failures{0};
  std::atomic<uint64_t> shift_candidates_tested{0};
  std::atomic<uint64_t> corrected_window_collision_calls{0};
  std::atomic<uint64_t> deformations_collision_calls{0};
  std::atomic<uint64_t> path_range_collision_calls{0};
  std::atomic<uint64_t> segment_collision_calls{0};
  std::atomic<uint64_t> deformation_trigger_collision_calls{0};
  std::atomic<uint64_t> footprint_collision_calls{0};
  std::atomic<uint64_t> scan_trigger_collision_calls{0};
  std::atomic<uint64_t> map_trigger_collision_calls{0};
  std::atomic<uint64_t> shifted_waypoint_calls{0};

  void reset()
  {
    estimate_shift_sign_calls.store(0, std::memory_order_relaxed);
    estimate_side_clearance_calls.store(0, std::memory_order_relaxed);
    safe_shift_calls.store(0, std::memory_order_relaxed);
    safe_shift_successes.store(0, std::memory_order_relaxed);
    safe_shift_failures.store(0, std::memory_order_relaxed);
    shift_candidates_tested.store(0, std::memory_order_relaxed);
    corrected_window_collision_calls.store(0, std::memory_order_relaxed);
    deformations_collision_calls.store(0, std::memory_order_relaxed);
    path_range_collision_calls.store(0, std::memory_order_relaxed);
    segment_collision_calls.store(0, std::memory_order_relaxed);
    deformation_trigger_collision_calls.store(0, std::memory_order_relaxed);
    footprint_collision_calls.store(0, std::memory_order_relaxed);
    scan_trigger_collision_calls.store(0, std::memory_order_relaxed);
    map_trigger_collision_calls.store(0, std::memory_order_relaxed);
    shifted_waypoint_calls.store(0, std::memory_order_relaxed);
  }
};

class GlobalPathDeformer : public rclcpp::Node
{
public:
  GlobalPathDeformer()
  : Node("global_path_deformer")
  {
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    const auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    const auto non_latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    const auto scan_overlay_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    realtime_map_topic_ = declare_parameter<std::string>("realtime_map_topic", "/map");
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan_matched_points2");
    global_path_topic_ = declare_parameter<std::string>("global_path_topic", "/global_path");
    output_path_topic_ = declare_parameter<std::string>("output_path_topic", "/Path");
    marker_topic_ = declare_parameter<std::string>("marker_topic", "/local_path");
    acc_marker_topic_ = declare_parameter<std::string>("acc_marker_topic", "/dynamic_acc_markers");

    static_obstacle_topic_ = declare_parameter<std::string>("static_obstacle_topic", "/static_obstacle");
    dynamic_obstacle_topic_ = declare_parameter<std::string>("dynamic_obstacle_topic", "/dynamic_obstacle");
    obj_flag_topic_ = declare_parameter<std::string>("obj_flag_topic", "/obj_flag");
    obstacle_mode_topic_ = declare_parameter<std::string>("obstacle_mode_topic", "/obstacle_mode");
    pcd_static_obstacle_topic_ = declare_parameter<std::string>("pcd_static_obstacle_topic", "");
    latched_input_qos_ = declare_parameter<bool>("latched_input_qos", true);
    obstacle_mode_control_enabled_ = declare_parameter<bool>("obstacle_mode_control_enabled", true);
    obstacle_mode_ = declare_parameter<int>("obstacle_mode_default", 1);
    mode1_dynamic_acc_enabled_ = declare_parameter<bool>("mode1_dynamic_acc_enabled", false);
    mode2_dynamic_acc_enabled_ = declare_parameter<bool>("mode2_dynamic_acc_enabled", true);
    mode3_dynamic_acc_enabled_ = declare_parameter<bool>("mode3_dynamic_acc_enabled", true);
    dynamic_acc_enabled_ = declare_parameter<bool>("dynamic_acc_enabled", true);
    dynamic_acc_marker_enabled_ = declare_parameter<bool>("dynamic_acc_marker_enabled", true);
    dynamic_acc_requires_path_window_ =
      declare_parameter<bool>("dynamic_acc_requires_path_window", true);
    dynamic_acc_path_lateral_width_ =
      declare_parameter<double>("dynamic_acc_path_lateral_width", 0.60);
    dynamic_acc_path_forward_distance_ =
      declare_parameter<double>("dynamic_acc_path_forward_distance", 1.00);
    dynamic_acc_path_extra_margin_ =
      declare_parameter<double>("dynamic_acc_path_extra_margin", 0.0);
    dynamic_acc_ttl_ms_ = declare_parameter<int>("dynamic_acc_ttl_ms", 500);
    dynamic_acc_front_yaw_limit_ = declare_parameter<double>("dynamic_acc_front_yaw_limit", 0.60);
    dynamic_acc_lateral_width_ = declare_parameter<double>("dynamic_acc_lateral_width", 0.60);
    dynamic_acc_min_distance_ = declare_parameter<double>("dynamic_acc_min_distance", 0.80);
    dynamic_acc_max_distance_ = declare_parameter<double>("dynamic_acc_max_distance", 4.00);
    dynamic_acc_time_headway_ = declare_parameter<double>("dynamic_acc_time_headway", 1.00);
    dynamic_acc_distance_gain_ = declare_parameter<double>("dynamic_acc_distance_gain", 0.80);
    dynamic_acc_speed_margin_ = declare_parameter<double>("dynamic_acc_speed_margin", 0.0);
    dynamic_acc_min_speed_ = declare_parameter<double>("dynamic_acc_min_speed", 0.0);
    curvature_speed_limit_enabled_ = declare_parameter<bool>("curvature_speed_limit_enabled", true);
    curvature_yaw_slowdown_start_ =
      declare_parameter<double>("curvature_yaw_slowdown_start", 0.25);
    curvature_yaw_slowdown_full_ =
      declare_parameter<double>("curvature_yaw_slowdown_full", 0.80);
    curvature_min_speed_scale_ = declare_parameter<double>("curvature_min_speed_scale", 0.45);
    curvature_min_speed_ = declare_parameter<double>("curvature_min_speed", 0.05);

    robot_radius_ = declare_parameter<double>("robot_radius", 0.14);
    vehicle_width_ = declare_parameter<double>("vehicle_width", 0.28);
    vehicle_length_ = declare_parameter<double>("vehicle_length", 0.512);
    footprint_margin_ = declare_parameter<double>("footprint_margin", 0.10);
    map_collision_enabled_ = declare_parameter<bool>("map_collision_enabled", true);
    map_occupied_requires_scan_confirmation_ =
      declare_parameter<bool>("map_occupied_requires_scan_confirmation", false);
    map_occupied_scan_confirmation_radius_ =
      declare_parameter<double>("map_occupied_scan_confirmation_radius", 0.20);
    map_trigger_enabled_ = declare_parameter<bool>("map_trigger_enabled", true);
    map_trigger_extra_margin_ = declare_parameter<double>("map_trigger_extra_margin", 0.15);
    map_trigger_lateral_width_ = declare_parameter<double>("map_trigger_lateral_width", 0.50);
    map_trigger_forward_distance_ = declare_parameter<double>("map_trigger_forward_distance", 1.20);
    occupied_threshold_ = declare_parameter<int>("occupied_threshold", 65);
    unknown_occupied_ = declare_parameter<bool>("unknown_occupied", true);
    scan_overlay_enabled_ = declare_parameter<bool>("scan_overlay_enabled", true);
    scan_overlay_ttl_ms_ = declare_parameter<int>("scan_overlay_ttl_ms", 250);
    scan_overlay_min_range_ = declare_parameter<double>("scan_overlay_min_range", 0.10);
    scan_overlay_max_range_ = declare_parameter<double>("scan_overlay_max_range", 0.0);
    scan_overlay_stride_ = declare_parameter<int>("scan_overlay_stride", 1);
    scan_overlay_index_cell_size_ = declare_parameter<double>("scan_overlay_index_cell_size", 0.25);
    scan_overlay_extra_margin_ = declare_parameter<double>("scan_overlay_extra_margin", 0.05);
    scan_overlay_boundary_enabled_ = declare_parameter<bool>("scan_overlay_boundary_enabled", true);
    scan_overlay_trigger_enabled_ = declare_parameter<bool>("scan_overlay_trigger_enabled", true);
    scan_overlay_boundary_extra_margin_ =
      declare_parameter<double>("scan_overlay_boundary_extra_margin", scan_overlay_extra_margin_);
    scan_overlay_trigger_extra_margin_ =
      declare_parameter<double>("scan_overlay_trigger_extra_margin", scan_overlay_extra_margin_);
    scan_overlay_trigger_lateral_width_ =
      declare_parameter<double>("scan_overlay_trigger_lateral_width", 0.0);
    scan_overlay_trigger_forward_distance_ =
      declare_parameter<double>("scan_overlay_trigger_forward_distance", 0.0);
    scan_overlay_near_trigger_radius_ =
      declare_parameter<double>("scan_overlay_near_trigger_radius", 0.0);
    scan_overlay_sensor_x_ = declare_parameter<double>("scan_overlay_sensor_x", 0.0);
    scan_overlay_sensor_y_ = declare_parameter<double>("scan_overlay_sensor_y", 0.0);
    scan_overlay_sensor_yaw_ = declare_parameter<double>("scan_overlay_sensor_yaw", 0.0);

    backward_waypoints_ = declare_parameter<int>("backward_waypoints", 8);
    forward_waypoints_ = declare_parameter<int>("forward_waypoints", 20);
    deformation_tail_waypoints_ = declare_parameter<int>("deformation_tail_waypoints", 0);
    gap_tolerance_ = declare_parameter<int>("gap_tolerance", 2);
    max_collision_cluster_span_ = declare_parameter<int>("max_collision_cluster_span", 0);
    closed_loop_path_ = declare_parameter<bool>("closed_loop_path", true);
    closest_index_backward_search_waypoints_ =
      declare_parameter<int>("closest_index_backward_search_waypoints", 5);
    closest_index_forward_search_waypoints_ =
      declare_parameter<int>("closest_index_forward_search_waypoints", 45);
    closest_index_reset_distance_ =
      declare_parameter<double>("closest_index_reset_distance", 1.20);
    use_full_path_search_ = declare_parameter<bool>("use_full_path_search", true);
    collision_search_start_ahead_waypoints_ =
      declare_parameter<int>("collision_search_start_ahead_waypoints", 8);
    collision_search_lookahead_waypoints_ =
      declare_parameter<int>("collision_search_lookahead_waypoints", 160);
    dynamic_acc_search_start_ahead_waypoints_ =
      declare_parameter<int>("dynamic_acc_search_start_ahead_waypoints", 3);
    dynamic_acc_search_lookahead_waypoints_ =
      declare_parameter<int>("dynamic_acc_search_lookahead_waypoints", 80);
    deformation_hold_cycles_ = declare_parameter<int>("deformation_hold_cycles", -1);
    deformation_release_cycles_ = declare_parameter<int>("deformation_release_cycles", 60);
    deformation_partial_update_grace_cycles_ =
      declare_parameter<int>("deformation_partial_update_grace_cycles", 12);
    preserve_active_deformations_ = declare_parameter<bool>("preserve_active_deformations", true);
    active_deformation_hold_uses_trigger_ =
      declare_parameter<bool>("active_deformation_hold_uses_trigger", false);
    commit_deformations_to_global_path_ =
      declare_parameter<bool>("commit_deformations_to_global_path", false);

    shift_step_ = declare_parameter<double>("shift_step", 0.05);
    max_shift_ = declare_parameter<double>("max_shift", 0.50);
    minimum_deformation_shift_ = declare_parameter<double>("minimum_deformation_shift", 0.0);
    deformation_extra_shift_ = declare_parameter<double>("deformation_extra_shift", 0.0);
    deformation_smoothing_alpha_ = declare_parameter<double>("deformation_smoothing_alpha", 0.35);
    shift_direction_clearance_margin_ =
      declare_parameter<double>("shift_direction_clearance_margin", 0.05);
    shift_refine_enabled_ = declare_parameter<bool>("shift_refine_enabled", true);
    shift_refine_iterations_ = declare_parameter<int>("shift_refine_iterations", 5);
    allow_opposite_shift_ = declare_parameter<bool>("allow_opposite_shift", true);
    process_period_ms_ = declare_parameter<int>("process_period_ms", 20);
    collision_check_step_ = declare_parameter<double>("collision_check_step", 0.0);
    parallel_enabled_ = declare_parameter<bool>("parallel_enabled", true);
    parallel_threads_ = declare_parameter<int>("parallel_threads", 0);
    parallel_min_items_ = declare_parameter<int>("parallel_min_items", 16);
    metrics_enabled_ = declare_parameter<bool>("metrics_enabled", true);
    metrics_csv_path_ = declare_parameter<std::string>(
      "metrics_csv_path",
      "/home/rcv/Documents/global_path_deformer/metrics/global_path_deformer_metrics.csv");
    metrics_flush_period_ = declare_parameter<int>("metrics_flush_period", 20);
    openMetricsFile();

    sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      realtime_map_topic_, latched_input_qos_ ? latched_qos : non_latched_qos,
      std::bind(&GlobalPathDeformer::cb_map, this, std::placeholders::_1));

    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      global_path_topic_, latched_input_qos_ ? latched_qos : non_latched_qos,
      std::bind(&GlobalPathDeformer::cb_path, this, std::placeholders::_1));

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_odom, this, std::placeholders::_1));

    if (scan_overlay_enabled_) {
      sub_scan_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        scan_topic_, scan_overlay_qos,
        std::bind(&GlobalPathDeformer::cb_scan, this, std::placeholders::_1));
    }

    sub_static_ = create_subscription<geometry_msgs::msg::PointStamped>(
      static_obstacle_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_static, this, std::placeholders::_1));

    if (!pcd_static_obstacle_topic_.empty()) {
      sub_pcd_static_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pcd_static_obstacle_topic_, qos,
        std::bind(&GlobalPathDeformer::cb_pcd_static_cloud, this, std::placeholders::_1));
    }

    sub_dynamic_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      dynamic_obstacle_topic_, scan_overlay_qos,
      std::bind(&GlobalPathDeformer::cb_dynamic, this, std::placeholders::_1));

    sub_obstacle_mode_ = create_subscription<std_msgs::msg::Int32>(
      obstacle_mode_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_obstacle_mode, this, std::placeholders::_1));

    sub_flag_ = create_subscription<geometry_msgs::msg::PointStamped>(
      obj_flag_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_flag, this, std::placeholders::_1));

    pub_path_ = create_publisher<nav_msgs::msg::Path>(output_path_topic_, 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(marker_topic_, 1);
    acc_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(acc_marker_topic_, 1);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(process_period_ms_),
      std::bind(&GlobalPathDeformer::process, this));

    RCLCPP_INFO(
      get_logger(),
      "global_path_deformer started. realtime_map_topic=%s, scan_topic=%s, scan_qos=best_effort, global_path_topic=%s, output_path_topic=%s, scan_boundary_margin=%.3f, scan_trigger_margin=%.3f, scan_trigger_lateral_width=%.3f, scan_trigger_forward_distance=%.3f",
      realtime_map_topic_.c_str(),
      scan_topic_.c_str(),
      global_path_topic_.c_str(),
      output_path_topic_.c_str(),
      scan_overlay_boundary_extra_margin_,
      scan_overlay_trigger_extra_margin_,
      scan_overlay_trigger_lateral_width_,
      scan_overlay_trigger_forward_distance_);
  }

private:
  void cb_map(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    map_ = *msg;
    map_received_ = true;

    if (!map_info_logged_ ||
        last_map_width_ != map_.info.width ||
        last_map_height_ != map_.info.height ||
        std::abs(last_map_resolution_ - map_.info.resolution) > 1e-9) {
      last_map_width_ = map_.info.width;
      last_map_height_ = map_.info.height;
      last_map_resolution_ = map_.info.resolution;
      map_info_logged_ = true;

      RCLCPP_INFO(
        get_logger(),
        "map received: size=%ux%u resolution=%.3f origin=(%.2f, %.2f)",
        map_.info.width,
        map_.info.height,
        map_.info.resolution,
        map_.info.origin.position.x,
        map_.info.origin.position.y);
    }
  }

  void cb_path(const nav_msgs::msg::Path::SharedPtr msg)
  {
    if (msg->poses.empty()) return;

    const std::string incoming_frame_id = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;
    if (isSameGlobalPath(*msg, incoming_frame_id)) {
      return;
    }

    path_frame_id_ = incoming_frame_id;
    original_global_path_.clear();
    original_global_path_.reserve(msg->poses.size());

    for (const auto & ps : msg->poses) {
      Waypoint wp;
      wp.x = ps.pose.position.x;
      wp.y = ps.pose.position.y;
      wp.v = ps.pose.position.z;  // z is velocity
      original_global_path_.push_back(wp);
    }

    deformed_global_path_ = original_global_path_;
    global_path_ = deformed_global_path_;
    path_received_ = true;
    tracked_closest_index_valid_ = false;
    clearActiveDeformation();

    if (last_path_size_ != global_path_.size()) {
      last_path_size_ = global_path_.size();
      RCLCPP_INFO(
        get_logger(),
        "global path received: poses=%zu frame=%s",
        global_path_.size(),
        path_frame_id_.c_str());
    }
  }

  bool isSameGlobalPath(const nav_msgs::msg::Path & msg, const std::string & frame_id) const
  {
    if (!path_received_ || frame_id != path_frame_id_ ||
        msg.poses.size() != original_global_path_.size()) {
      return false;
    }

    constexpr double path_epsilon = 1e-6;
    for (size_t i = 0; i < msg.poses.size(); ++i) {
      const auto & pose = msg.poses[i].pose.position;
      const auto & waypoint = original_global_path_[i];
      if (std::abs(pose.x - waypoint.x) > path_epsilon ||
          std::abs(pose.y - waypoint.y) > path_epsilon ||
          std::abs(pose.z - waypoint.v) > path_epsilon)
      {
        return false;
      }
    }

    return true;
  }

  void cb_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    odom_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    odom_frame_id_ = msg->header.frame_id;
    odom_received_ = true;
  }

  bool transformPointToPathFrame(
    double x,
    double y,
    const std::string & cloud_frame,
    double & world_x,
    double & world_y) const
  {
    const std::string path_frame = path_frame_id_.empty() ? "map" : path_frame_id_;
    if (cloud_frame.empty() || cloud_frame == path_frame) {
      world_x = x;
      world_y = y;
      return true;
    }

    if (!odom_received_) {
      return false;
    }

    const double cos_odom = std::cos(odom_yaw_);
    const double sin_odom = std::sin(odom_yaw_);
    const double cos_sensor = std::cos(scan_overlay_sensor_yaw_);
    const double sin_sensor = std::sin(scan_overlay_sensor_yaw_);
    const double base_x = scan_overlay_sensor_x_ + cos_sensor * x - sin_sensor * y;
    const double base_y = scan_overlay_sensor_y_ + sin_sensor * x + cos_sensor * y;
    world_x = odom_x_ + cos_odom * base_x - sin_odom * base_y;
    world_y = odom_y_ + sin_odom * base_x + cos_odom * base_y;
    return true;
  }

  void cb_scan(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!scan_overlay_enabled_) return;

    const int stride = std::max(1, scan_overlay_stride_);
    std::vector<ScanObstacle> obstacles;
    obstacles.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height) /
      static_cast<size_t>(stride) + 1);
    const std::string cloud_frame = msg->header.frame_id.empty() ? path_frame_id_ : msg->header.frame_id;
    const std::string path_frame = path_frame_id_.empty() ? "map" : path_frame_id_;
    const bool cloud_in_path_frame = cloud_frame == path_frame;
    const bool can_transform_from_robot_frame = cloud_in_path_frame || odom_received_;

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      size_t index = 0;

      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++index) {
        if (index % static_cast<size_t>(stride) != 0) continue;

        const double x = static_cast<double>(*iter_x);
        const double y = static_cast<double>(*iter_y);
        if (!std::isfinite(x) || !std::isfinite(y)) continue;

        double world_x = x;
        double world_y = y;
        if (!transformPointToPathFrame(x, y, cloud_frame, world_x, world_y)) {
          continue;
        }

        if (odom_received_) {
          const double range = std::hypot(world_x - odom_x_, world_y - odom_y_);
          if (range < scan_overlay_min_range_) continue;
          if (scan_overlay_max_range_ > 0.0 && range > scan_overlay_max_range_) continue;
        }

        obstacles.push_back(ScanObstacle{world_x, world_y});
      }
    } catch (const std::runtime_error & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "scan overlay PointCloud2 missing x/y float fields: %s",
        e.what());
      return;
    }

    scan_obstacles_ = std::move(obstacles);
    rebuildScanObstacleIndex();
    last_scan_overlay_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "scan overlay received: topic=%s frame=%s path_frame=%s transform=%s input_points=%zu used_points=%zu",
      scan_topic_.c_str(),
      cloud_frame.c_str(),
      path_frame.c_str(),
      cloud_in_path_frame ? "none" : (can_transform_from_robot_frame ? "odom" : "dropped_no_odom"),
      static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height),
      scan_obstacles_.size());
  }

  void cb_static(const geometry_msgs::msg::PointStamped::SharedPtr) {}
  void cb_obstacle_mode(const std_msgs::msg::Int32::SharedPtr msg)
  {
    obstacle_mode_ = msg->data;
    obstacle_mode_received_ = true;

    RCLCPP_INFO_ONCE(
      get_logger(),
      "obstacle mode received: mode=%d",
      obstacle_mode_);
  }

  void cb_dynamic(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const std::string cloud_frame = msg->header.frame_id.empty() ? path_frame_id_ : msg->header.frame_id;
    const std::string path_frame = path_frame_id_.empty() ? "map" : path_frame_id_;
    const bool cloud_in_path_frame = cloud_frame == path_frame;
    const bool can_transform_from_robot_frame = cloud_in_path_frame || odom_received_;
    std::vector<DynamicPoint> dynamic_points;
    dynamic_points.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height));

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_track_id(*msg, "track_id");
      sensor_msgs::PointCloud2ConstIterator<float> iter_speed(*msg, "relative_speed");
      sensor_msgs::PointCloud2ConstIterator<float> iter_yaw(*msg, "relative_yaw");

      for (;
        iter_x != iter_x.end();
        ++iter_x, ++iter_y, ++iter_track_id, ++iter_speed, ++iter_yaw)
      {
        const double x = static_cast<double>(*iter_x);
        const double y = static_cast<double>(*iter_y);
        if (!std::isfinite(x) || !std::isfinite(y)) continue;

        double world_x = x;
        double world_y = y;
        if (!transformPointToPathFrame(x, y, cloud_frame, world_x, world_y)) {
          continue;
        }

        dynamic_points.push_back(DynamicPoint{
          world_x,
          world_y,
          std::hypot(x, y),
          *iter_track_id,
          static_cast<double>(*iter_speed),
          static_cast<double>(*iter_yaw)});
      }
    } catch (const std::runtime_error & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "dynamic PointCloud2 missing expected fields x/y/track_id/relative_speed/relative_yaw: %s",
        e.what());
      return;
    }

    dynamic_points_ = std::move(dynamic_points);
    last_dynamic_pointcloud_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "dynamic pointcloud received: topic=%s frame=%s path_frame=%s transform=%s input_points=%zu used_points=%zu",
      dynamic_obstacle_topic_.c_str(),
      cloud_frame.c_str(),
      path_frame.c_str(),
      cloud_in_path_frame ? "none" : (can_transform_from_robot_frame ? "odom" : "dropped_no_odom"),
      static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height),
      dynamic_points_.size());
  }
  void cb_flag(const geometry_msgs::msg::PointStamped::SharedPtr) {}
  void cb_pcd_static_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr) {}

  bool worldToMap(double wx, double wy, int & mx, int & my) const
  {
    const double ox = map_.info.origin.position.x;
    const double oy = map_.info.origin.position.y;
    const double res = map_.info.resolution;

    if (res <= 0.0) return false;

    mx = static_cast<int>(std::floor((wx - ox) / res));
    my = static_cast<int>(std::floor((wy - oy) / res));

    return mx >= 0 && my >= 0 &&
           mx < static_cast<int>(map_.info.width) &&
           my < static_cast<int>(map_.info.height);
  }

  int getCost(int mx, int my) const
  {
    if (mx < 0 || my < 0 ||
        mx >= static_cast<int>(map_.info.width) ||
        my >= static_cast<int>(map_.info.height)) {
      return 100;
    }

    const int idx = my * static_cast<int>(map_.info.width) + mx;
    return static_cast<int>(map_.data[idx]);
  }

  bool isOccupiedCost(int cost) const
  {
    if (cost < 0) return unknown_occupied_;
    return cost >= occupied_threshold_;
  }

  bool isMapOccupiedWorld(double x, double y) const
  {
    if (!map_collision_enabled_) return false;
    int mx = 0;
    int my = 0;
    if (!worldToMap(x, y, mx, my)) return false;
    return isOccupiedCost(getCost(mx, my));
  }

  bool scanOverlayFresh() const
  {
    if (!scan_overlay_enabled_ || scan_obstacles_.empty() || scan_overlay_ttl_ms_ <= 0) {
      return false;
    }

    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - last_scan_overlay_time_);
    return age.count() <= scan_overlay_ttl_ms_;
  }

  int scanObstacleCell(double value) const
  {
    const double cell_size = std::max(0.05, scan_overlay_index_cell_size_);
    return static_cast<int>(std::floor(value / cell_size));
  }

  int64_t scanObstacleCellKey(int cx, int cy) const
  {
    return (static_cast<int64_t>(cx) << 32) ^
           static_cast<int64_t>(static_cast<uint32_t>(cy));
  }

  void rebuildScanObstacleIndex()
  {
    scan_obstacle_index_.clear();
    scan_obstacle_index_.reserve(scan_obstacles_.size() * 2 + 1);
    for (size_t i = 0; i < scan_obstacles_.size(); ++i) {
      const auto & obstacle = scan_obstacles_[i];
      const int cx = scanObstacleCell(obstacle.x);
      const int cy = scanObstacleCell(obstacle.y);
      scan_obstacle_index_[scanObstacleCellKey(cx, cy)].push_back(i);
    }
  }

  template<typename Callback>
  bool anyScanObstacleNear(double x, double y, double radius, Callback cb) const
  {
    if (scan_obstacles_.empty()) return false;
    if (scan_obstacle_index_.empty()) {
      for (const auto & obstacle : scan_obstacles_) {
        if (cb(obstacle)) return true;
      }
      return false;
    }

    const double cell_size = std::max(0.05, scan_overlay_index_cell_size_);
    const int center_x = scanObstacleCell(x);
    const int center_y = scanObstacleCell(y);
    const int cell_radius = std::max(0, static_cast<int>(std::ceil(radius / cell_size)));
    const double radius2 = std::max(0.0, radius) * std::max(0.0, radius);

    for (int dx = -cell_radius; dx <= cell_radius; ++dx) {
      for (int dy = -cell_radius; dy <= cell_radius; ++dy) {
        const auto it = scan_obstacle_index_.find(
          scanObstacleCellKey(center_x + dx, center_y + dy));
        if (it == scan_obstacle_index_.end()) continue;
        for (const size_t index : it->second) {
          if (index >= scan_obstacles_.size()) continue;
          const auto & obstacle = scan_obstacles_[index];
          const double ox = obstacle.x - x;
          const double oy = obstacle.y - y;
          if (ox * ox + oy * oy > radius2) continue;
          if (cb(obstacle)) return true;
        }
      }
    }

    return false;
  }

  double footprintBoundingRadius() const
  {
    const double margin = std::max(0.0, footprint_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    return std::hypot(half_length, half_width);
  }

  bool scanConfirmsMapOccupied(double x, double y) const
  {
    if (!map_occupied_requires_scan_confirmation_) return true;
    if (!scanOverlayFresh()) return true;

    const double radius = std::max(0.0, map_occupied_scan_confirmation_radius_);
    return anyScanObstacleNear(x, y, radius, [](const ScanObstacle &) {
      return true;
    });
  }

  bool isFootprintCollision(double x, double y, double yaw) const
  {
    profile_counters_.footprint_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (isScanBoundaryCollision(x, y, yaw)) return true;
    if (!map_collision_enabled_) return false;

    int mx = 0;
    int my = 0;
    if (!worldToMap(x, y, mx, my)) return true;

    const double res = map_.info.resolution;
    if (res <= 0.0) return true;

    const double margin = std::max(0.0, footprint_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    const int r_cell = static_cast<int>(std::ceil(footprintBoundingRadius() / res));
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    for (int dx = -r_cell; dx <= r_cell; ++dx) {
      for (int dy = -r_cell; dy <= r_cell; ++dy) {
        const int cell_mx = mx + dx;
        const int cell_my = my + dy;
        const double cell_x = map_.info.origin.position.x +
          (static_cast<double>(cell_mx) + 0.5) * res;
        const double cell_y = map_.info.origin.position.y +
          (static_cast<double>(cell_my) + 0.5) * res;
        const double vx = cell_x - x;
        const double vy = cell_y - y;
        const double longitudinal = cos_yaw * vx + sin_yaw * vy;
        const double lateral = -sin_yaw * vx + cos_yaw * vy;
        if (std::abs(longitudinal) > half_length || std::abs(lateral) > half_width) {
          continue;
        }

        const int cost = getCost(cell_mx, cell_my);
        if (isOccupiedCost(cost)) return true;
      }
    }

    return false;
  }

  bool scanObstacleInLocalBox(
    const ScanObstacle & obstacle,
    double x,
    double y,
    double yaw,
    double extra_margin,
    double trigger_forward_distance,
    double trigger_lateral_width) const
  {
    const double margin = std::max(0.0, footprint_margin_ + extra_margin);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double footprint_half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    const double half_width = trigger_lateral_width > 0.0 ?
      std::max(footprint_half_width, trigger_lateral_width) :
      footprint_half_width;
    const double forward_limit = trigger_forward_distance > 0.0 ?
      std::max(half_length, trigger_forward_distance) :
      half_length;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    const double vx = obstacle.x - x;
    const double vy = obstacle.y - y;
    const double longitudinal = cos_yaw * vx + sin_yaw * vy;
    const double lateral = -sin_yaw * vx + cos_yaw * vy;
    return longitudinal >= -half_length &&
           longitudinal <= forward_limit &&
           std::abs(lateral) <= half_width;
  }

  bool scanObstacleNearPathPoint(const ScanObstacle & obstacle, double x, double y) const
  {
    if (scan_overlay_near_trigger_radius_ <= 0.0) return false;

    const double radius = scan_overlay_near_trigger_radius_;
    const double dx = obstacle.x - x;
    const double dy = obstacle.y - y;
    return dx * dx + dy * dy <= radius * radius;
  }

  bool isScanBoundaryCollision(double x, double y, double yaw) const
  {
    if (!scan_overlay_boundary_enabled_ || !scanOverlayFresh()) return false;

    const double query_radius = footprintBoundingRadius() +
      std::max(0.0, scan_overlay_boundary_extra_margin_);
    return anyScanObstacleNear(
      x,
      y,
      query_radius,
      [&](const ScanObstacle & obstacle) {
        return scanObstacleInLocalBox(
          obstacle,
          x,
          y,
          yaw,
          scan_overlay_boundary_extra_margin_,
          0.0,
          0.0);
      });
  }

  bool isScanTriggerCollision(double x, double y, double yaw) const
  {
    profile_counters_.scan_trigger_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (!scan_overlay_trigger_enabled_ || !scanOverlayFresh()) return false;

    const double margin = std::max(0.0, footprint_margin_ + scan_overlay_trigger_extra_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double footprint_half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    const double half_width = scan_overlay_trigger_lateral_width_ > 0.0 ?
      std::max(footprint_half_width, scan_overlay_trigger_lateral_width_) :
      footprint_half_width;
    const double forward_limit = scan_overlay_trigger_forward_distance_ > 0.0 ?
      std::max(half_length, scan_overlay_trigger_forward_distance_) :
      half_length;
    const double query_radius = std::max(
      std::max(0.0, scan_overlay_near_trigger_radius_),
      std::hypot(std::max(half_length, forward_limit), half_width));

    return anyScanObstacleNear(
      x,
      y,
      query_radius,
      [&](const ScanObstacle & obstacle) {
        if (scanObstacleNearPathPoint(obstacle, x, y)) return true;
        return scanObstacleInLocalBox(
          obstacle,
          x,
          y,
          yaw,
          scan_overlay_trigger_extra_margin_,
          scan_overlay_trigger_forward_distance_,
          scan_overlay_trigger_lateral_width_);
      });
  }

  bool isMapTriggerCollision(double x, double y, double yaw) const
  {
    profile_counters_.map_trigger_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (!map_collision_enabled_ || !map_trigger_enabled_ || !map_received_) return false;

    int mx = 0;
    int my = 0;
    if (!worldToMap(x, y, mx, my)) return false;

    const double res = map_.info.resolution;
    if (res <= 0.0) return false;

    const double margin = std::max(0.0, footprint_margin_ + map_trigger_extra_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double footprint_half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    const double half_width = map_trigger_lateral_width_ > 0.0 ?
      std::max(footprint_half_width, map_trigger_lateral_width_) :
      footprint_half_width;
    const double forward_limit = map_trigger_forward_distance_ > 0.0 ?
      std::max(half_length, map_trigger_forward_distance_) :
      half_length;
    const double search_radius = std::hypot(forward_limit, half_width);
    const int r_cell = static_cast<int>(std::ceil(search_radius / res));
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    for (int dx = -r_cell; dx <= r_cell; ++dx) {
      for (int dy = -r_cell; dy <= r_cell; ++dy) {
        const int cell_mx = mx + dx;
        const int cell_my = my + dy;
        if (cell_mx < 0 || cell_my < 0 ||
            cell_mx >= static_cast<int>(map_.info.width) ||
            cell_my >= static_cast<int>(map_.info.height)) {
          continue;
        }

        const int cost = getCost(cell_mx, cell_my);
        if (!isOccupiedCost(cost)) continue;

        const double cell_x = map_.info.origin.position.x +
          (static_cast<double>(cell_mx) + 0.5) * res;
        const double cell_y = map_.info.origin.position.y +
          (static_cast<double>(cell_my) + 0.5) * res;
        if (!scanConfirmsMapOccupied(cell_x, cell_y)) continue;

        const double vx = cell_x - x;
        const double vy = cell_y - y;
        const double longitudinal = cos_yaw * vx + sin_yaw * vy;
        const double lateral = -sin_yaw * vx + cos_yaw * vy;
        if (longitudinal >= -half_length &&
            longitudinal <= forward_limit &&
            std::abs(lateral) <= half_width) {
          return true;
        }
      }
    }

    return false;
  }

  bool isDeformationTriggerCollision(double x, double y, double yaw) const
  {
    profile_counters_.deformation_trigger_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (isScanTriggerCollision(x, y, yaw)) return true;
    if (isMapTriggerCollision(x, y, yaw)) return true;
    return isFootprintCollision(x, y, yaw);
  }

  double pathYawAtIndex(int i) const
  {
    const int last_index = static_cast<int>(global_path_.size()) - 1;
    int i0 = std::max(0, i - 1);
    int i1 = std::min(last_index, i + 1);

    if (closed_loop_path_ && global_path_.size() > 2) {
      i0 = (i == 0) ? last_index : i - 1;
      i1 = (i == last_index) ? 0 : i + 1;
    }

    return std::atan2(global_path_[i1].y - global_path_[i0].y, global_path_[i1].x - global_path_[i0].x);
  }

  int wrapIndex(int index) const
  {
    const int size = static_cast<int>(global_path_.size());
    if (size <= 0) return 0;
    const int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
  }

  int ringDistance(int from, int to) const
  {
    const int size = static_cast<int>(global_path_.size());
    if (size <= 0) return 0;
    return (wrapIndex(to) - wrapIndex(from) + size) % size;
  }

  bool indexInForwardRange(int index, int start, int end) const
  {
    if (global_path_.empty()) return false;
    if (!closed_loop_path_ && start > end) return false;
    if (!closed_loop_path_) return index >= start && index <= end;
    return ringDistance(start, index) <= ringDistance(start, end);
  }

  template<typename Callback>
  void forEachIndexInForwardRange(int start, int end, Callback cb) const
  {
    if (global_path_.empty()) return;

    if (!closed_loop_path_) {
      start = std::max(0, start);
      end = std::min(static_cast<int>(global_path_.size()) - 1, end);
      for (int i = start; i <= end; ++i) {
        cb(i);
      }
      return;
    }

    int i = wrapIndex(start);
    const int wrapped_end = wrapIndex(end);
    for (size_t steps = 0; steps < global_path_.size(); ++steps) {
      cb(i);
      if (i == wrapped_end) break;
      i = wrapIndex(i + 1);
    }
  }

  double effectiveCollisionCheckStep() const
  {
    if (collision_check_step_ > 0.0) return collision_check_step_;
    if (map_.info.resolution > 0.0) return std::max(0.01, map_.info.resolution * 0.5);
    return 0.05;
  }

  Waypoint shiftedWaypoint(int i, const DeformationWindow & window, double shift) const
  {
    Waypoint wp = global_path_[i];
    if (!window.active() || !indexInForwardRange(i, window.start, window.end)) {
      return wp;
    }

    double nx = 0.0;
    double ny = 0.0;
    computeNormal(i, nx, ny);
    const double beta = blendingWeight(i, window);
    wp.x += beta * shift * nx;
    wp.y += beta * shift * ny;
    return wp;
  }

  Waypoint shiftedWaypoint(int i, const std::vector<PathDeformation> & deformations) const
  {
    profile_counters_.shifted_waypoint_calls.fetch_add(1, std::memory_order_relaxed);
    Waypoint wp = global_path_[i];

    for (const auto & deformation : deformations) {
      if (!deformation.active() ||
          !indexInForwardRange(i, deformation.window.start, deformation.window.end)) {
        continue;
      }

      double nx = 0.0;
      double ny = 0.0;
      computeNormal(i, nx, ny);
      const double beta = blendingWeight(i, deformation.window);
      wp.x += beta * deformation.shift * nx;
      wp.y += beta * deformation.shift * ny;
    }

    return wp;
  }

  bool segmentCollision(
    double x0,
    double y0,
    double x1,
    double y1,
    bool use_deformation_trigger = false) const
  {
    profile_counters_.segment_collision_calls.fetch_add(1, std::memory_order_relaxed);
    const double length = std::hypot(x1 - x0, y1 - y0);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / effectiveCollisionCheckStep())));
    const double yaw = std::atan2(y1 - y0, x1 - x0);

    for (int k = 0; k <= steps; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(steps);
      const double x = x0 + t * (x1 - x0);
      const double y = y0 + t * (y1 - y0);
      if (use_deformation_trigger) {
        if (isDeformationTriggerCollision(x, y, yaw)) return true;
      } else if (isFootprintCollision(x, y, yaw)) {
        return true;
      }
    }

    return false;
  }

  double pathDistanceSquaredToOdom(int index) const
  {
    const double dx = global_path_[index].x - odom_x_;
    const double dy = global_path_[index].y - odom_y_;
    return dx * dx + dy * dy;
  }

  int getClosestPathIndexFull() const
  {
    if (!odom_received_ || global_path_.empty()) return 0;

    int best_index = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();

    for (int i = 0; i < static_cast<int>(global_path_.size()); ++i) {
      const double dist2 = pathDistanceSquaredToOdom(i);
      if (dist2 < best_dist2) {
        best_dist2 = dist2;
        best_index = i;
      }
    }

    return best_index;
  }

  int getClosestPathIndex()
  {
    if (!odom_received_ || global_path_.empty()) return 0;

    if (!tracked_closest_index_valid_ ||
        tracked_closest_path_size_ != global_path_.size()) {
      tracked_closest_index_ = getClosestPathIndexFull();
      tracked_closest_path_size_ = global_path_.size();
      tracked_closest_index_valid_ = true;
      return tracked_closest_index_;
    }

    int best_index = tracked_closest_index_;
    double best_dist2 = pathDistanceSquaredToOdom(best_index);
    const int backward = std::max(0, closest_index_backward_search_waypoints_);
    const int forward = std::max(1, closest_index_forward_search_waypoints_);

    if (closed_loop_path_) {
      const int start = wrapIndex(tracked_closest_index_ - backward);
      const int end = wrapIndex(tracked_closest_index_ + forward);
      forEachIndexInForwardRange(start, end, [&](int i) {
        const double dist2 = pathDistanceSquaredToOdom(i);
        if (dist2 < best_dist2) {
          best_dist2 = dist2;
          best_index = i;
        }
      });
    } else {
      const int start = std::max(0, tracked_closest_index_ - backward);
      const int end = std::min(
        static_cast<int>(global_path_.size()) - 1,
        tracked_closest_index_ + forward);
      for (int i = start; i <= end; ++i) {
        const double dist2 = pathDistanceSquaredToOdom(i);
        if (dist2 < best_dist2) {
          best_dist2 = dist2;
          best_index = i;
        }
      }
    }

    const double reset_distance = std::max(0.0, closest_index_reset_distance_);
    if (reset_distance > 0.0 && best_dist2 > reset_distance * reset_distance) {
      best_index = getClosestPathIndexFull();
    }

    tracked_closest_index_ = best_index;
    tracked_closest_path_size_ = global_path_.size();
    tracked_closest_index_valid_ = true;
    return tracked_closest_index_;
  }

  int getUpcomingPathIndex(int closest_index) const
  {
    if (global_path_.empty()) return 0;

    const int last_index = static_cast<int>(global_path_.size()) - 1;
    const int start_ahead = std::max(0, collision_search_start_ahead_waypoints_);
    if (closed_loop_path_) {
      return wrapIndex(closest_index + start_ahead);
    }

    return std::clamp(
      closest_index + start_ahead,
      0,
      last_index);
  }

  void getCollisionSearchRange(int & closest_index, int & start, int & end)
  {
    const int last_index = static_cast<int>(global_path_.size()) - 1;
    if (use_full_path_search_) {
      closest_index = odom_received_ ? std::clamp(getClosestPathIndex(), 0, last_index) : 0;
      start = 0;
      end = last_index;
      return;
    }

    if (odom_received_) {
      closest_index = std::clamp(getClosestPathIndex(), 0, last_index);
      start = getUpcomingPathIndex(closest_index);
      end = closed_loop_path_ ?
        wrapIndex(start + collision_search_lookahead_waypoints_) :
        std::min(last_index, start + collision_search_lookahead_waypoints_);
      return;
    }

    closest_index = 0;
    start = 0;
    end = last_index;
  }

  void getDynamicAccSearchRange(int closest_index, int & start, int & end) const
  {
    const int last_index = static_cast<int>(global_path_.size()) - 1;
    if (global_path_.empty()) {
      start = 0;
      end = 0;
      return;
    }

    const int start_ahead = std::max(0, dynamic_acc_search_start_ahead_waypoints_);
    const int lookahead = std::max(0, dynamic_acc_search_lookahead_waypoints_);

    if (closed_loop_path_) {
      start = wrapIndex(closest_index + start_ahead);
      end = wrapIndex(start + lookahead);
      return;
    }

    start = std::clamp(closest_index + start_ahead, 0, last_index);
    end = std::min(last_index, start + lookahead);
  }

  size_t effectiveParallelWorkers(size_t item_count) const
  {
    if (item_count == 0) {
      return 0;
    }
    if (!parallel_enabled_ || item_count < static_cast<size_t>(std::max(1, parallel_min_items_))) {
      return 1;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const size_t configured_threads = parallel_threads_ > 0 ?
      static_cast<size_t>(parallel_threads_) :
      static_cast<size_t>(std::max(1u, hardware_threads));
    return std::clamp(configured_threads, static_cast<size_t>(1), item_count);
  }

  template<typename Fn>
  size_t parallelForRanges(size_t item_count, Fn && fn) const
  {
    const size_t workers = effectiveParallelWorkers(item_count);
    if (workers == 0) {
      return 0;
    }
    if (workers <= 1) {
      fn(0, item_count, 0);
      return 1;
    }

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
      const size_t begin = item_count * worker / workers;
      const size_t end = item_count * (worker + 1) / workers;
      threads.emplace_back([begin, end, worker, &fn]() {
        fn(begin, end, worker);
      });
    }

    for (auto & thread : threads) {
      thread.join();
    }
    return workers;
  }

  std::vector<int> detectCollisionIndices(int start, int end) const
  {
    std::vector<int> range_indices;

    forEachIndexInForwardRange(start, end, [&](int i) {
      range_indices.push_back(i);
    });

    if (range_indices.empty()) {
      return {};
    }

    const size_t workers = effectiveParallelWorkers(range_indices.size());
    std::vector<std::vector<int>> local_indices(workers);
    parallelForRanges(range_indices.size(), [&](size_t begin, size_t end, size_t worker) {
      auto & local = local_indices[worker];
      local.reserve((end - begin) / 2 + 2);

      for (size_t k = begin; k < end; ++k) {
        const int i = range_indices[k];
        if (isDeformationTriggerCollision(global_path_[i].x, global_path_[i].y, pathYawAtIndex(i))) {
          local.push_back(i);
        }

        if (k == 0) {
          continue;
        }
        const int prev = range_indices[k - 1];
        const int curr = range_indices[k];
        if (segmentCollision(
            global_path_[prev].x,
            global_path_[prev].y,
            global_path_[curr].x,
            global_path_[curr].y,
            true))
        {
          local.push_back(prev);
          local.push_back(curr);
        }
      }
    });

    std::vector<int> indices;
    std::vector<bool> seen(global_path_.size(), false);
    auto add_index = [&indices, &seen](int index) {
      if (index < 0 || index >= static_cast<int>(seen.size()) || seen[index]) {
        return;
      }
      seen[index] = true;
      indices.push_back(index);
    };

    for (const auto & local : local_indices) {
      for (const int index : local) {
        add_index(index);
      }
    }

    if (closed_loop_path_ && range_indices.size() == global_path_.size()) {
      const int first = range_indices.front();
      const int last = range_indices.back();
      if (segmentCollision(
          global_path_[last].x,
          global_path_[last].y,
          global_path_[first].x,
          global_path_[first].y,
          true))
      {
        add_index(last);
        add_index(first);
      }
    }

    return indices;
  }

  bool getFirstCluster(const std::vector<int> & collision_indices, int & ia, int & ib) const
  {
    if (collision_indices.empty()) return false;

    ia = collision_indices.front();
    ib = collision_indices.front();

    for (size_t k = 1; k < collision_indices.size(); ++k) {
      if (collision_indices[k] - ib <= gap_tolerance_) {
        ib = collision_indices[k];
      } else {
        break;
      }
    }
    return true;
  }

  std::vector<std::pair<int, int>> getCollisionClusters(
    const std::vector<int> & collision_indices) const
  {
    std::vector<std::pair<int, int>> clusters;
    if (collision_indices.empty()) return clusters;

    int ia = collision_indices.front();
    int ib = collision_indices.front();

    for (size_t k = 1; k < collision_indices.size(); ++k) {
      const int gap = closed_loop_path_ ?
        ringDistance(ib, collision_indices[k]) :
        collision_indices[k] - ib;
      if (gap <= gap_tolerance_) {
        ib = collision_indices[k];
      } else {
        clusters.emplace_back(ia, ib);
        ia = collision_indices[k];
        ib = collision_indices[k];
      }
    }

    clusters.emplace_back(ia, ib);

    if (closed_loop_path_ && use_full_path_search_ && clusters.size() > 1) {
      const int last_index = static_cast<int>(global_path_.size()) - 1;
      const auto first_cluster = clusters.front();
      const auto last_cluster = clusters.back();
      const int wrap_gap = (last_index - last_cluster.second) + first_cluster.first;

      if (wrap_gap <= gap_tolerance_) {
        clusters.front() = {last_cluster.first, first_cluster.second};
        clusters.pop_back();
      }
    }

    return clusters;
  }

  std::vector<std::pair<int, int>> splitLargeCollisionClusters(
    const std::vector<std::pair<int, int>> & clusters) const
  {
    const int max_span = max_collision_cluster_span_;
    if (max_span <= 0 || clusters.empty() || global_path_.empty()) {
      return clusters;
    }

    std::vector<std::pair<int, int>> split_clusters;
    for (const auto & cluster : clusters) {
      int start = cluster.first;
      const int end = cluster.second;
      int remaining = closed_loop_path_ ?
        ringDistance(start, end) :
        std::max(0, end - start);

      if (remaining <= max_span) {
        split_clusters.push_back(cluster);
        continue;
      }

      while (remaining > max_span) {
        const int chunk_end = closed_loop_path_ ?
          wrapIndex(start + max_span) :
          std::min(end, start + max_span);
        split_clusters.emplace_back(start, chunk_end);
        start = closed_loop_path_ ? wrapIndex(chunk_end + 1) : chunk_end + 1;
        remaining = closed_loop_path_ ?
          ringDistance(start, end) :
          std::max(0, end - start);
      }

      split_clusters.emplace_back(start, end);
    }

    return split_clusters;
  }

  void getWindow(int ia, int ib, int min_start, int & is, int & ie) const
  {
    if (closed_loop_path_) {
      (void)min_start;
      const int last_index = static_cast<int>(global_path_.size()) - 1;
      if (ia == 0 && ib == last_index) {
        is = 0;
        ie = last_index;
        return;
      }
      is = wrapIndex(ia - backward_waypoints_);
      ie = wrapIndex(ib + std::max(0, deformation_tail_waypoints_) + forward_waypoints_);
      return;
    }

    is = std::max(min_start, ia - backward_waypoints_);
    ie = std::min(
      static_cast<int>(global_path_.size()) - 1,
      ib + std::max(0, deformation_tail_waypoints_) + forward_waypoints_);
  }

  DeformationWindow getWindow(int ia, int ib, int min_start) const
  {
    DeformationWindow window;
    window.collision_start = ia;
    window.collision_end = ib;
    getWindow(ia, ib, min_start, window.start, window.end);
    return window;
  }

  void computeNormal(int i, double & nx, double & ny) const
  {
    const int last_index = static_cast<int>(global_path_.size()) - 1;
    int i0 = std::max(0, i - 1);
    int i1 = std::min(last_index, i + 1);

    if (closed_loop_path_ && global_path_.size() > 2) {
      i0 = (i == 0) ? last_index : i - 1;
      i1 = (i == last_index) ? 0 : i + 1;
    }

    double tx = global_path_[i1].x - global_path_[i0].x;
    double ty = global_path_[i1].y - global_path_[i0].y;

    const double norm = std::hypot(tx, ty);
    if (norm < 1e-6) {
      nx = 0.0;
      ny = 0.0;
      return;
    }

    tx /= norm;
    ty /= norm;
    nx = -ty;
    ny = tx;
  }

  double smoothRamp(double eta) const
  {
    eta = std::clamp(eta, 0.0, 1.0);
    return 0.5 - 0.5 * std::cos(M_PI * eta);
  }

  double blendingWeight(int i, const DeformationWindow & window) const
  {
    if (!window.active() || !indexInForwardRange(i, window.start, window.end)) return 0.0;

    const int dist_i = ringDistance(window.start, i);
    const int dist_collision_start = ringDistance(window.start, window.collision_start);
    const int dist_collision_end = ringDistance(window.start, window.collision_end);
    const int dist_end = ringDistance(window.start, window.end);

    if (dist_i >= dist_collision_start && dist_i <= dist_collision_end) return 1.0;

    if (dist_i < dist_collision_start) {
      if (dist_collision_start <= 0) return 1.0;
      const double eta = static_cast<double>(dist_i) /
        static_cast<double>(dist_collision_start);
      return smoothRamp(eta);
    }

    const int dist_full_shift_end = std::min(
      dist_end,
      dist_collision_end + std::max(0, deformation_tail_waypoints_));
    if (dist_i <= dist_full_shift_end) return 1.0;
    if (dist_end <= dist_full_shift_end) return 1.0;

    const double eta = static_cast<double>(dist_end - dist_i) /
      static_cast<double>(dist_end - dist_full_shift_end);
    return smoothRamp(eta);
  }

  bool windowsOverlap(const DeformationWindow & a, const DeformationWindow & b) const
  {
    if (!a.active() || !b.active()) return false;

    bool overlap = false;
    forEachIndexInForwardRange(a.start, a.end, [&](int i) {
      if (!overlap && indexInForwardRange(i, b.start, b.end)) {
        overlap = true;
      }
    });
    return overlap;
  }

  std::vector<PathDeformation> smoothWithActiveDeformations(
    const std::vector<PathDeformation> & candidate_deformations) const
  {
    if (!hasActiveDeformation() || candidate_deformations.empty()) {
      return candidate_deformations;
    }

    const double alpha = std::clamp(deformation_smoothing_alpha_, 0.0, 1.0);
    if (alpha >= 1.0) {
      return candidate_deformations;
    }

    std::vector<PathDeformation> smoothed = candidate_deformations;
    for (auto & candidate : smoothed) {
      const PathDeformation * best_match = nullptr;
      for (const auto & active : active_deformations_) {
        if (!active.active() || !windowsOverlap(candidate.window, active.window)) {
          continue;
        }
        if (candidate.shift * active.shift < 0.0) {
          continue;
        }
        best_match = &active;
        break;
      }

      if (best_match != nullptr) {
        candidate.shift = alpha * candidate.shift + (1.0 - alpha) * best_match->shift;
      }
    }

    return deformationsCollision(smoothed) ? candidate_deformations : smoothed;
  }

  bool activeDeformationStillRelevant(
    const PathDeformation & deformation,
    int search_start) const
  {
    if (!deformation.active()) return false;
    if (deformation_hold_cycles_ >= 0 && active_clear_cycles_ >= deformation_hold_cycles_) {
      return false;
    }

    if (closed_loop_path_) {
      const int max_hold_distance = std::max(
        0,
        collision_search_lookahead_waypoints_ +
        forward_waypoints_ +
        std::max(0, deformation_tail_waypoints_));
      if (ringDistance(search_start, deformation.window.end) > max_hold_distance) {
        return false;
      }
    } else if (search_start > deformation.window.end) {
      return false;
    }

    return !deformationsCollision({deformation});
  }

  std::vector<PathDeformation> mergeReusableActiveDeformations(
    const std::vector<PathDeformation> & candidate_deformations,
    int search_start) const
  {
    if (!preserve_active_deformations_ || !hasActiveDeformation()) {
      return candidate_deformations;
    }

    std::vector<PathDeformation> merged = candidate_deformations;
    for (const auto & active : active_deformations_) {
      if (!activeDeformationStillRelevant(active, search_start)) {
        continue;
      }

      bool overlaps_candidate = false;
      for (const auto & candidate : candidate_deformations) {
        if (windowsOverlap(active.window, candidate.window)) {
          overlaps_candidate = true;
          break;
        }
      }
      if (overlaps_candidate) {
        continue;
      }

      std::vector<PathDeformation> trial = merged;
      trial.push_back(active);
      if (!deformationsCollision(trial)) {
        merged.push_back(active);
      }
    }

    std::stable_sort(
      merged.begin(),
      merged.end(),
      [this, search_start](const PathDeformation & a, const PathDeformation & b) {
        const int a_rank = closed_loop_path_ ?
          ringDistance(search_start, a.window.start) :
          a.window.start;
        const int b_rank = closed_loop_path_ ?
          ringDistance(search_start, b.window.start) :
          b.window.start;
        return a_rank < b_rank;
      });

    return merged;
  }

  double estimateSideClearance(int is, int ie, int sign) const
  {
    profile_counters_.estimate_side_clearance_calls.fetch_add(1, std::memory_order_relaxed);
    if (sign == 0 || global_path_.empty()) return 0.0;

    const double step = std::max(0.05, effectiveCollisionCheckStep());
    const double limit = std::max(step, std::min(max_shift_, 1.20));
    double clearance_sum = 0.0;
    int sample_count = 0;

    forEachIndexInForwardRange(is, ie, [&](int i) {
      double nx = 0.0;
      double ny = 0.0;
      computeNormal(i, nx, ny);
      if (std::hypot(nx, ny) < 1e-6) return;

      const double yaw = pathYawAtIndex(i);
      double clear_distance = 0.0;
      for (double d = step; d <= limit + 1e-9; d += step) {
        const double x = global_path_[i].x + static_cast<double>(sign) * d * nx;
        const double y = global_path_[i].y + static_cast<double>(sign) * d * ny;
        if (isFootprintCollision(x, y, yaw)) {
          break;
        }
        clear_distance = d;
      }

      clearance_sum += clear_distance;
      ++sample_count;
    });

    return sample_count > 0 ? clearance_sum / static_cast<double>(sample_count) : 0.0;
  }

  int estimateClearanceShiftSign(int is, int ie) const
  {
    const double positive_clearance = estimateSideClearance(is, ie, 1);
    const double negative_clearance = estimateSideClearance(is, ie, -1);
    const double margin = std::max(0.0, shift_direction_clearance_margin_);

    if (positive_clearance > negative_clearance + margin) return 1;
    if (negative_clearance > positive_clearance + margin) return -1;
    return 0;
  }

  int estimateShiftSign(int is, int ie) const
  {
    profile_counters_.estimate_shift_sign_calls.fetch_add(1, std::memory_order_relaxed);
    const int clearance_sign = estimateClearanceShiftSign(is, ie);
    if (clearance_sign != 0) {
      return clearance_sign;
    }

    double scan_signed_sum = 0.0;
    int scan_count = 0;

    if (scan_overlay_trigger_enabled_ && scanOverlayFresh()) {
      forEachIndexInForwardRange(is, ie, [&](int i) {
        double nx = 0.0;
        double ny = 0.0;
        computeNormal(i, nx, ny);
        if (std::hypot(nx, ny) < 1e-6) return;
        const double yaw = pathYawAtIndex(i);

        for (const auto & obstacle : scan_obstacles_) {
          if (!scanObstacleNearPathPoint(obstacle, global_path_[i].x, global_path_[i].y) &&
              !scanObstacleInLocalBox(
              obstacle,
              global_path_[i].x,
              global_path_[i].y,
              yaw,
              scan_overlay_trigger_extra_margin_,
              scan_overlay_trigger_forward_distance_,
              scan_overlay_trigger_lateral_width_))
          {
            continue;
          }
          const double vx = obstacle.x - global_path_[i].x;
          const double vy = obstacle.y - global_path_[i].y;
          scan_signed_sum += vx * nx + vy * ny;
          ++scan_count;
        }
      });
    }

    if (scan_count > 0) {
      return scan_signed_sum > 0.0 ? -1 : 1;
    }

    if (scan_overlay_boundary_enabled_ && scanOverlayFresh()) {
      double boundary_signed_sum = 0.0;
      int boundary_count = 0;
      forEachIndexInForwardRange(is, ie, [&](int i) {
        double nx = 0.0;
        double ny = 0.0;
        computeNormal(i, nx, ny);
        if (std::hypot(nx, ny) < 1e-6) return;
        const double yaw = pathYawAtIndex(i);

        for (const auto & obstacle : scan_obstacles_) {
          if (!scanObstacleInLocalBox(
              obstacle,
              global_path_[i].x,
              global_path_[i].y,
              yaw,
              scan_overlay_boundary_extra_margin_,
              0.0,
              0.0))
          {
            continue;
          }
          const double vx = obstacle.x - global_path_[i].x;
          const double vy = obstacle.y - global_path_[i].y;
          boundary_signed_sum += vx * nx + vy * ny;
          ++boundary_count;
        }
      });

      if (boundary_count > 0) {
        return boundary_signed_sum > 0.0 ? -1 : 1;
      }
    }

    if (!map_collision_enabled_) return 0;

    double map_signed_sum = 0.0;
    int map_count = 0;
    const int r_cell = static_cast<int>(std::ceil(footprintBoundingRadius() / map_.info.resolution));

    forEachIndexInForwardRange(is, ie, [&](int i) {
      int mx = 0;
      int my = 0;
      if (!worldToMap(global_path_[i].x, global_path_[i].y, mx, my)) return;

      double nx = 0.0;
      double ny = 0.0;
      computeNormal(i, nx, ny);

      for (int dx = -r_cell; dx <= r_cell; ++dx) {
        for (int dy = -r_cell; dy <= r_cell; ++dy) {
          if (dx * dx + dy * dy > r_cell * r_cell) continue;
          const int cost = getCost(mx + dx, my + dy);
          if (!isOccupiedCost(cost)) continue;

          const double cell_x = map_.info.origin.position.x +
            (static_cast<double>(mx + dx) + 0.5) * map_.info.resolution;
          const double cell_y = map_.info.origin.position.y +
            (static_cast<double>(my + dy) + 0.5) * map_.info.resolution;
          if (!scanConfirmsMapOccupied(cell_x, cell_y)) continue;

          const double vx = cell_x - global_path_[i].x;
          const double vy = cell_y - global_path_[i].y;
          map_signed_sum += vx * nx + vy * ny;
          ++map_count;
        }
      }
    });

    if (map_count == 0) return 0;
    // Obstacle on +normal side means shift to -normal side.
    return map_signed_sum > 0.0 ? -1 : 1;
  }

  bool pathRangeCollision(
    const std::vector<PathDeformation> & deformations,
    int first,
    int last,
    bool include_closing_segment,
    bool use_deformation_trigger = false) const
  {
    profile_counters_.path_range_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (global_path_.empty()) return false;

    if (!closed_loop_path_) {
      first = std::max(0, first);
      last = std::min(static_cast<int>(global_path_.size()) - 1, last);
      if (first > last) return false;
    } else {
      first = wrapIndex(first);
      last = wrapIndex(last);
    }

    const Waypoint first_wp = shiftedWaypoint(first, deformations);
    Waypoint prev = first_wp;
    int prev_index = first;
    if (use_deformation_trigger) {
      if (isDeformationTriggerCollision(prev.x, prev.y, pathYawAtIndex(prev_index))) return true;
    } else if (isFootprintCollision(prev.x, prev.y, pathYawAtIndex(prev_index))) {
      return true;
    }

    if (closed_loop_path_) {
      int i = wrapIndex(first + 1);
      for (size_t steps = 1; steps < global_path_.size(); ++steps) {
        const Waypoint curr = shiftedWaypoint(i, deformations);
        if (segmentCollision(prev.x, prev.y, curr.x, curr.y, use_deformation_trigger)) return true;
        prev = curr;
        prev_index = i;
        if (i == last) break;
        i = wrapIndex(i + 1);
      }
    } else {
      for (int i = first + 1; i <= last; ++i) {
        const Waypoint curr = shiftedWaypoint(i, deformations);
        if (segmentCollision(prev.x, prev.y, curr.x, curr.y, use_deformation_trigger)) return true;
        prev = curr;
        prev_index = i;
      }
    }

    if (include_closing_segment && prev_index != first) {
      if (segmentCollision(prev.x, prev.y, first_wp.x, first_wp.y, use_deformation_trigger)) {
        return true;
      }
    }

    return false;
  }

  bool correctedWindowCollision(
    const DeformationWindow & window,
    double shift,
    const std::vector<PathDeformation> & existing_deformations = {},
    bool use_deformation_trigger = false) const
  {
    profile_counters_.corrected_window_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (!window.active()) return false;

    std::vector<PathDeformation> deformations = existing_deformations;
    deformations.push_back(PathDeformation{window, shift});

    const int first = closed_loop_path_ ?
      wrapIndex(window.start - 1) :
      std::max(0, window.start - 1);
    const int last = closed_loop_path_ ?
      wrapIndex(window.end + 1) :
      std::min(static_cast<int>(global_path_.size()) - 1, window.end + 1);

    return pathRangeCollision(deformations, first, last, false, use_deformation_trigger);
  }

  bool deformationsCollision(
    const std::vector<PathDeformation> & deformations,
    bool use_deformation_trigger = false) const
  {
    profile_counters_.deformations_collision_calls.fetch_add(1, std::memory_order_relaxed);
    if (deformations.empty()) return false;
    return pathRangeCollision(
      deformations,
      0,
      static_cast<int>(global_path_.size()) - 1,
      closed_loop_path_,
      use_deformation_trigger);
  }

  double refineSafeShift(
    const DeformationWindow & window,
    int sign,
    double unsafe_abs_shift,
    double safe_abs_shift,
    const std::vector<PathDeformation> & existing_deformations) const
  {
    if (!shift_refine_enabled_ || shift_refine_iterations_ <= 0 || sign == 0) {
      return static_cast<double>(sign) * safe_abs_shift;
    }

    double lo = std::max(0.0, unsafe_abs_shift);
    double hi = std::max(lo, safe_abs_shift);
    const double min_shift = std::max(0.0, minimum_deformation_shift_);
    if (min_shift > 0.0 && hi >= min_shift) {
      lo = std::max(lo, std::min(min_shift, hi));
    }

    for (int iter = 0; iter < shift_refine_iterations_; ++iter) {
      const double mid = 0.5 * (lo + hi);
      const double candidate = static_cast<double>(sign) * mid;
      if (correctedWindowCollision(window, candidate, existing_deformations)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const double refined_safe_shift = hi;
    const double extra_shift = std::max(0.0, deformation_extra_shift_);
    if (extra_shift <= 0.0) {
      return static_cast<double>(sign) * refined_safe_shift;
    }

    const double target_shift = std::min(max_shift_, refined_safe_shift + extra_shift);
    if (target_shift <= refined_safe_shift) {
      return static_cast<double>(sign) * refined_safe_shift;
    }

    const double target_candidate = static_cast<double>(sign) * target_shift;
    if (!correctedWindowCollision(window, target_candidate, existing_deformations)) {
      return target_candidate;
    }

    double safe = refined_safe_shift;
    double unsafe = target_shift;
    for (int iter = 0; iter < shift_refine_iterations_; ++iter) {
      const double mid = 0.5 * (safe + unsafe);
      const double candidate = static_cast<double>(sign) * mid;
      if (correctedWindowCollision(window, candidate, existing_deformations)) {
        unsafe = mid;
      } else {
        safe = mid;
      }
    }

    return static_cast<double>(sign) * safe;
  }

  bool findSafeShift(
    const DeformationWindow & window,
    int preferred_sign,
    const std::vector<PathDeformation> & existing_deformations,
    double & shift,
    std::string & failure_reason) const
  {
    profile_counters_.safe_shift_calls.fetch_add(1, std::memory_order_relaxed);
    failure_reason.clear();
    if (shift_step_ <= 0.0) {
      shift = 0.0;
      failure_reason = "invalid_shift_step";
      return false;
    }

    std::vector<int> shift_signs;
    if (preferred_sign != 0) {
      shift_signs.push_back(preferred_sign);
      if (allow_opposite_shift_) {
        shift_signs.push_back(-preferred_sign);
      }
    } else if (allow_opposite_shift_) {
      shift_signs.push_back(1);
      shift_signs.push_back(-1);
    } else {
      shift = 0.0;
      failure_reason = "shift_sign_zero";
      return false;
    }

    const int max_iter = static_cast<int>(std::ceil(max_shift_ / shift_step_));
    for (int k = 1; k <= max_iter; ++k) {
      for (const int sign : shift_signs) {
        const double d = std::clamp(
          static_cast<double>(sign) * static_cast<double>(k) * shift_step_,
          -max_shift_,
          max_shift_);
        profile_counters_.shift_candidates_tested.fetch_add(1, std::memory_order_relaxed);
        if (!correctedWindowCollision(window, d, existing_deformations)) {
          const double safe_abs_shift = std::abs(d);
          const double unsafe_abs_shift = std::min(
            safe_abs_shift,
            static_cast<double>(k - 1) * shift_step_);
          shift = refineSafeShift(window, sign, unsafe_abs_shift, safe_abs_shift, existing_deformations);
          profile_counters_.safe_shift_successes.fetch_add(1, std::memory_order_relaxed);
          return true;
        }
      }
    }

    shift = 0.0;
    failure_reason = allow_opposite_shift_ ?
      "window_collision_all_shift_candidates_both_directions" :
      "window_collision_all_shift_candidates";
    profile_counters_.safe_shift_failures.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  bool dynamicPointcloudFresh() const
  {
    if (!dynamic_acc_enabled_ || dynamic_points_.empty() || dynamic_acc_ttl_ms_ <= 0) {
      return false;
    }

    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - last_dynamic_pointcloud_time_);
    return age.count() <= dynamic_acc_ttl_ms_;
  }

  bool dynamicAccAllowedByObstacleMode() const
  {
    if (!obstacle_mode_control_enabled_) {
      return true;
    }

    switch (obstacle_mode_) {
      case 1:
        return mode1_dynamic_acc_enabled_;
      case 2:
        return mode2_dynamic_acc_enabled_;
      case 3:
        return mode3_dynamic_acc_enabled_;
      default:
        return false;
    }
  }

  bool dynamicPointPassesAccGeometry(const DynamicPoint & point) const
  {
    const double yaw_limit = std::max(0.0, dynamic_acc_front_yaw_limit_);
    const double lateral_width = std::max(0.0, dynamic_acc_lateral_width_);
    const double min_distance = std::max(0.0, dynamic_acc_min_distance_);
    const double max_distance = std::max(min_distance, dynamic_acc_max_distance_);
    const double range = point.range;

    if (!std::isfinite(range) || range <= 0.0 || range > max_distance) {
      return false;
    }

    const double bearing = std::atan2(std::sin(point.relative_yaw), std::cos(point.relative_yaw));
    if (!std::isfinite(bearing) || !std::isfinite(point.relative_speed)) {
      return false;
    }
    if (std::abs(bearing) > yaw_limit) {
      return false;
    }

    const double lateral_offset = std::abs(std::sin(bearing) * range);
    return lateral_width <= 0.0 || lateral_offset <= lateral_width;
  }

  double pointSegmentDistanceSquared(
    double px,
    double py,
    double x0,
    double y0,
    double x1,
    double y1) const
  {
    const double vx = x1 - x0;
    const double vy = y1 - y0;
    const double length2 = vx * vx + vy * vy;
    if (length2 <= 1e-12) {
      const double dx = px - x0;
      const double dy = py - y0;
      return dx * dx + dy * dy;
    }

    const double t = std::clamp(((px - x0) * vx + (py - y0) * vy) / length2, 0.0, 1.0);
    const double cx = x0 + t * vx;
    const double cy = y0 + t * vy;
    const double dx = px - cx;
    const double dy = py - cy;
    return dx * dx + dy * dy;
  }

  bool dynamicPointNearPathWindow(const DynamicPoint & point, int start, int end) const
  {
    if (global_path_.empty()) {
      return false;
    }

    bool near = false;
    forEachIndexInForwardRange(start, end, [&](int i) {
      if (near) {
        return;
      }

      const Waypoint & curr = global_path_[i];
      const double yaw = pathYawAtIndex(i);
      const double margin = std::max(0.0, footprint_margin_ + dynamic_acc_path_extra_margin_);
      const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
      const double footprint_half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
      const double half_width = dynamic_acc_path_lateral_width_ > 0.0 ?
        std::max(footprint_half_width, dynamic_acc_path_lateral_width_) :
        footprint_half_width;
      const double forward_limit = dynamic_acc_path_forward_distance_ > 0.0 ?
        std::max(half_length, dynamic_acc_path_forward_distance_) :
        half_length;
      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);
      const double vx = point.x - curr.x;
      const double vy = point.y - curr.y;
      const double longitudinal = cos_yaw * vx + sin_yaw * vy;
      const double lateral = -sin_yaw * vx + cos_yaw * vy;

      near = longitudinal >= -half_length &&
             longitudinal <= forward_limit &&
             std::abs(lateral) <= half_width;
    });

    return near;
  }

  void updateDynamicAccWindowState(int search_start, int search_end)
  {
    dynamic_acc_path_window_active_ = false;
    dynamic_acc_active_point_indices_.clear();
    if (!dynamic_acc_enabled_ || !dynamicAccAllowedByObstacleMode() || !dynamicPointcloudFresh()) {
      return;
    }

    for (size_t i = 0; i < dynamic_points_.size(); ++i) {
      const auto & point = dynamic_points_[i];
      if (!dynamicPointPassesAccGeometry(point)) {
        continue;
      }
      if (!dynamic_acc_requires_path_window_ ||
          dynamicPointNearPathWindow(point, search_start, search_end))
      {
        dynamic_acc_active_point_indices_.push_back(i);
      }
    }

    dynamic_acc_path_window_active_ = !dynamic_acc_active_point_indices_.empty();
  }

  double applyDynamicAccVelocityLimit(double nominal_speed) const
  {
    if (nominal_speed <= 0.0 ||
        !dynamic_acc_path_window_active_ ||
        !dynamicAccAllowedByObstacleMode() ||
        !dynamicPointcloudFresh())
    {
      return nominal_speed;
    }

    const double min_distance = std::max(0.0, dynamic_acc_min_distance_);
    const double time_headway = std::max(0.0, dynamic_acc_time_headway_);
    const double distance_gain = std::max(0.0, dynamic_acc_distance_gain_);
    const double min_speed = std::max(0.0, dynamic_acc_min_speed_);
    double limited_speed = nominal_speed;

    for (const size_t index : dynamic_acc_active_point_indices_) {
      if (index >= dynamic_points_.size()) {
        continue;
      }

      const auto & point = dynamic_points_[index];
      if (!dynamicPointPassesAccGeometry(point)) {
        continue;
      }

      const double range = point.range;
      const double desired_distance = min_distance + time_headway * nominal_speed;
      const double distance_error = range - desired_distance;
      const double lead_speed = std::max(0.0, point.relative_speed + dynamic_acc_speed_margin_);
      const double acc_speed = lead_speed + distance_gain * distance_error;
      limited_speed = std::min(limited_speed, std::clamp(acc_speed, min_speed, nominal_speed));
    }

    return limited_speed;
  }

  double normalizeAngle(double angle) const
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  double shiftedPathYawAtIndex(int index, const std::vector<Waypoint> & shifted_path) const
  {
    if (shifted_path.empty()) return 0.0;

    const int last_index = static_cast<int>(shifted_path.size()) - 1;
    int i0 = std::max(0, index - 1);
    int i1 = std::min(last_index, index + 1);

    if (closed_loop_path_ && shifted_path.size() > 2) {
      i0 = (index == 0) ? last_index : index - 1;
      i1 = (index == last_index) ? 0 : index + 1;
    }

    return std::atan2(
      shifted_path[i1].y - shifted_path[i0].y,
      shifted_path[i1].x - shifted_path[i0].x);
  }

  double applyCurvatureVelocityLimit(
    int index,
    double nominal_speed,
    const std::vector<Waypoint> & shifted_path) const
  {
    if (!curvature_speed_limit_enabled_ || nominal_speed <= 0.0 || shifted_path.size() < 3) {
      return nominal_speed;
    }

    const double start = std::max(0.0, curvature_yaw_slowdown_start_);
    const double full = std::max(start + 1e-6, curvature_yaw_slowdown_full_);
    const double min_scale = std::clamp(curvature_min_speed_scale_, 0.0, 1.0);
    const double yaw_error = std::abs(normalizeAngle(
      shiftedPathYawAtIndex(index, shifted_path) - pathYawAtIndex(index)));

    if (yaw_error <= start) {
      return nominal_speed;
    }

    const double ratio = std::clamp((yaw_error - start) / (full - start), 0.0, 1.0);
    const double scale = 1.0 - ratio * (1.0 - min_scale);
    const double limited_speed = nominal_speed * scale;
    return std::clamp(limited_speed, std::max(0.0, curvature_min_speed_), nominal_speed);
  }

  nav_msgs::msg::Path makeCorrectedPath(
    const std::vector<PathDeformation> & deformations) const
  {
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = path_frame_id_;
    path.poses.reserve(global_path_.size());

    std::vector<Waypoint> shifted_path;
    shifted_path.reserve(global_path_.size());
    for (int i = 0; i < static_cast<int>(global_path_.size()); ++i) {
      shifted_path.push_back(shiftedWaypoint(i, deformations));
    }

    for (int i = 0; i < static_cast<int>(shifted_path.size()); ++i) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;

      double speed = global_path_[i].v;
      speed = applyCurvatureVelocityLimit(i, speed, shifted_path);
      speed = applyDynamicAccVelocityLimit(speed);

      ps.pose.position.x = shifted_path[i].x;
      ps.pose.position.y = shifted_path[i].y;
      ps.pose.position.z = speed;
      ps.pose.orientation.w = 1.0;
      path.poses.push_back(ps);
    }

    return path;
  }

  void commitDeformationsToDeformedPath(const std::vector<PathDeformation> & deformations)
  {
    if (!commit_deformations_to_global_path_ || deformations.empty()) {
      return;
    }

    std::vector<Waypoint> shifted_path;
    shifted_path.reserve(global_path_.size());
    for (int i = 0; i < static_cast<int>(global_path_.size()); ++i) {
      shifted_path.push_back(shiftedWaypoint(i, deformations));
    }

    if (deformed_global_path_.size() != global_path_.size()) {
      deformed_global_path_ = global_path_;
    }

    for (size_t i = 0; i < deformed_global_path_.size(); ++i) {
      deformed_global_path_[i].x = shifted_path[i].x;
      deformed_global_path_[i].y = shifted_path[i].y;
    }

    global_path_ = deformed_global_path_;
    clearActiveDeformation();
    tracked_closest_index_valid_ = false;
  }

  void publishMarker(const nav_msgs::msg::Path & path)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = path.header;
    marker.ns = "corrected_path";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.color.r = 1.0;
    marker.color.g = 0.3;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    for (const auto & ps : path.poses) {
      geometry_msgs::msg::Point p;
      p.x = ps.pose.position.x;
      p.y = ps.pose.position.y;
      p.z = 0.1;
      marker.points.push_back(p);
    }
    if (closed_loop_path_ && !marker.points.empty()) {
      marker.points.push_back(marker.points.front());
    }
    marker_pub_->publish(marker);
  }

  bool isDynamicAccActivePoint(size_t index) const
  {
    return std::find(
      dynamic_acc_active_point_indices_.begin(),
      dynamic_acc_active_point_indices_.end(),
      index) != dynamic_acc_active_point_indices_.end();
  }

  void publishDynamicAccMarkers(int search_start)
  {
    if (!dynamic_acc_marker_enabled_ || !acc_marker_pub_) {
      return;
    }

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = now();
    clear.header.frame_id = path_frame_id_;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);

    const bool fresh = dynamicPointcloudFresh();
    const bool mode_allowed = dynamicAccAllowedByObstacleMode();
    int id = 1;

    for (size_t i = 0; i < dynamic_points_.size(); ++i) {
      const auto & point = dynamic_points_[i];
      const bool active = fresh && mode_allowed && isDynamicAccActivePoint(i);
      const bool geometry_ok = dynamicPointPassesAccGeometry(point);

      visualization_msgs::msg::Marker marker;
      marker.header = clear.header;
      marker.ns = active ? "dynamic_acc_active" : "dynamic_acc_candidates";
      marker.id = id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = point.x;
      marker.pose.position.y = point.y;
      marker.pose.position.z = active ? 0.35 : 0.25;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = active ? 0.28 : 0.16;
      marker.scale.y = marker.scale.x;
      marker.scale.z = marker.scale.x;
      marker.color.a = geometry_ok ? 0.95 : 0.35;
      if (active) {
        marker.color.r = 1.0;
        marker.color.g = 0.05;
        marker.color.b = 0.02;
      } else {
        marker.color.r = 0.45;
        marker.color.g = 0.45;
        marker.color.b = 0.45;
      }
      markers.markers.push_back(marker);
    }

    visualization_msgs::msg::Marker text;
    text.header = clear.header;
    text.ns = "dynamic_acc_status";
    text.id = id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.orientation.w = 1.0;
    text.pose.position.z = 0.9;
    if (odom_received_) {
      text.pose.position.x = odom_x_;
      text.pose.position.y = odom_y_;
    } else if (!global_path_.empty()) {
      text.pose.position.x = global_path_[wrapIndex(search_start)].x;
      text.pose.position.y = global_path_[wrapIndex(search_start)].y;
    }
    text.scale.z = 0.45;
    text.color.a = 1.0;
    text.color.r = 0.0;
    text.color.g = 0.0;
    text.color.b = 0.0;
    std::ostringstream oss;
    oss
      << "ACC " << (dynamic_acc_path_window_active_ ? "ON" : "OFF")
      << " mode=" << obstacle_mode_
      << " fresh=" << (fresh ? 1 : 0)
      << " active_points=" << dynamic_acc_active_point_indices_.size();
    text.text = oss.str();

    markers.markers.push_back(text);

    acc_marker_pub_->publish(markers);
  }

  void process()
  {
    resetProfiling();
    const auto process_start = std::chrono::steady_clock::now();
    auto add_stage_ms =
      [this](double & target, const std::chrono::steady_clock::time_point & start) {
        target += elapsedProcessMs(start);
      };

    const bool map_ready = map_received_ || !map_collision_enabled_;
    if (!map_ready || !path_received_ || global_path_.size() < 3) {
      logWaitingState();
      writeMetrics(
        "waiting",
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        0,
        0,
        {},
        {},
        elapsedProcessMs(process_start));
      return;
    }
    if (map_collision_enabled_) {
      logFrameMismatch();
    }

    int closest_index = 0;
    int search_start = 0;
    int search_end = static_cast<int>(global_path_.size()) - 1;
    auto stage_start = std::chrono::steady_clock::now();
    getCollisionSearchRange(closest_index, search_start, search_end);
    int dynamic_acc_search_start = search_start;
    int dynamic_acc_search_end = search_end;
    getDynamicAccSearchRange(closest_index, dynamic_acc_search_start, dynamic_acc_search_end);
    updateDynamicAccWindowState(dynamic_acc_search_start, dynamic_acc_search_end);
    add_stage_ms(process_profile_.range_ms, stage_start);

    std::vector<int> collision_indices;
    std::vector<PathDeformation> deformations;
    std::string state = "clear";
    size_t cluster_count = 0;
    size_t failed_clusters = 0;
    std::vector<std::string> failure_details;

    stage_start = std::chrono::steady_clock::now();
    const bool can_hold_active =
      canHoldActiveDeformation(search_start, active_deformation_hold_uses_trigger_);
    add_stage_ms(process_profile_.hold_check_ms, stage_start);

    if (can_hold_active) {
      deformations = active_deformations_;
      active_collision_grace_cycles_ = 0;
      ++active_clear_cycles_;
      state = "holding";
      logHoldState(closest_index, search_start, search_end, "active path remains collision-free");
    } else {
      stage_start = std::chrono::steady_clock::now();
      collision_indices = detectCollisionIndices(search_start, search_end);
      add_stage_ms(process_profile_.detect_collision_ms, stage_start);
    }

    if (state != "holding" && !collision_indices.empty()) {
      stage_start = std::chrono::steady_clock::now();
      const auto clusters = splitLargeCollisionClusters(getCollisionClusters(collision_indices));
      add_stage_ms(process_profile_.cluster_ms, stage_start);
      cluster_count = clusters.size();

      int min_window_start = use_full_path_search_ ? 0 : search_start;

      for (const auto & cluster : clusters) {
        const int ia = cluster.first;
        const int ib = cluster.second;
        DeformationWindow window = getWindow(ia, ib, min_window_start);
        if (!window.active()) {
          ++failed_clusters;
          failure_details.push_back(makeFailureDetail(ia, ib, window, 0, "invalid_window"));
          continue;
        }

        stage_start = std::chrono::steady_clock::now();
        const int shift_sign = estimateShiftSign(ia, ib);
        add_stage_ms(process_profile_.shift_sign_ms, stage_start);
        double d_req = 0.0;
        std::string failure_reason;
        stage_start = std::chrono::steady_clock::now();
        bool safe_shift_found =
          findSafeShift(window, shift_sign, deformations, d_req, failure_reason);
        add_stage_ms(process_profile_.find_safe_shift_ms, stage_start);
        if (safe_shift_found) {
          deformations.push_back(PathDeformation{window, d_req});
          min_window_start = window.end + 1;
        } else {
          ++failed_clusters;
          failure_details.push_back(makeFailureDetail(ia, ib, window, shift_sign, failure_reason));
        }
      }

      stage_start = std::chrono::steady_clock::now();
      const bool keep_active_over_partial =
        !deformations.empty() && shouldKeepActiveOverPartialUpdate(failed_clusters);
      add_stage_ms(process_profile_.partial_update_ms, stage_start);

      if (keep_active_over_partial) {
        deformations = active_deformations_;
        ++active_collision_grace_cycles_;
        ++active_clear_cycles_;
        state = "holding";
        logHoldState(
          closest_index,
          search_start,
          search_end,
          "keeping previous deformation over partial update");
      } else if (!deformations.empty()) {
        stage_start = std::chrono::steady_clock::now();
        deformations = mergeReusableActiveDeformations(deformations, search_start);
        add_stage_ms(process_profile_.merge_ms, stage_start);
        stage_start = std::chrono::steady_clock::now();
        deformations = smoothWithActiveDeformations(deformations);
        add_stage_ms(process_profile_.smooth_ms, stage_start);
        active_deformation_valid_ = true;
        active_deformations_ = deformations;
        active_clear_cycles_ = 0;
        active_collision_grace_cycles_ = 0;
        state = "deforming";
        logDeformationState(
          collision_indices.size(),
          cluster_count,
          deformations.size(),
          failed_clusters,
          closest_index,
          search_start,
          search_end,
          deformations);
      } else {
        stage_start = std::chrono::steady_clock::now();
        const bool can_hold_after_failed_shift =
          canHoldActiveDeformation(search_start, active_deformation_hold_uses_trigger_);
        add_stage_ms(process_profile_.hold_check_ms, stage_start);

        if (can_hold_after_failed_shift) {
          deformations = active_deformations_;
          active_collision_grace_cycles_ = 0;
          ++active_clear_cycles_;
          state = "holding";
          logHoldState(closest_index, search_start, search_end, "safe shift search failed");
        } else {
          state = "unsafe";
          if (hasActiveDeformation()) {
            deformations = active_deformations_;
            ++active_clear_cycles_;
          } else {
            clearActiveDeformation();
          }
          logUnsafeState(
            collision_indices.size(),
            cluster_count,
            closest_index,
            search_start,
            search_end,
            !deformations.empty());
        }
      }
    } else if (state != "holding") {
      stage_start = std::chrono::steady_clock::now();
      const bool can_hold_clear = canHoldActiveDeformation(search_start);
      add_stage_ms(process_profile_.hold_check_ms, stage_start);

      if (can_hold_clear) {
        deformations = active_deformations_;
        active_collision_grace_cycles_ = 0;
        ++active_clear_cycles_;
        state = "holding";
        logHoldState(closest_index, search_start, search_end, "temporary clear detection");
      } else {
        stage_start = std::chrono::steady_clock::now();
        const bool can_release_hold = canReleaseHoldActiveDeformation();
        add_stage_ms(process_profile_.release_check_ms, stage_start);

        if (can_release_hold) {
          deformations = active_deformations_;
          ++active_clear_cycles_;
          state = "holding";
          logHoldState(closest_index, search_start, search_end, "release hysteresis");
        } else {
          clearActiveDeformation();
          logClearState();
        }
      }
    }

    std::vector<PathDeformation> metrics_deformations = deformations;
    if (commit_deformations_to_global_path_ && state == "deforming" && !deformations.empty()) {
      commitDeformationsToDeformedPath(deformations);
      deformations.clear();
      state = "committed";
    }

    stage_start = std::chrono::steady_clock::now();
    const auto corrected_path = makeCorrectedPath(deformations);
    add_stage_ms(process_profile_.make_path_ms, stage_start);
    stage_start = std::chrono::steady_clock::now();
    pub_path_->publish(corrected_path);
    publishMarker(corrected_path);
    publishDynamicAccMarkers(dynamic_acc_search_start);
    add_stage_ms(process_profile_.publish_ms, stage_start);
    writeMetrics(
      state,
      closest_index,
      search_start,
      search_start,
      search_end,
      collision_indices.size(),
      cluster_count,
      metrics_deformations.size(),
      failed_clusters,
      metrics_deformations,
      failure_details,
      elapsedProcessMs(process_start));
  }

  void openMetricsFile()
  {
    if (!metrics_enabled_) return;

    const std::filesystem::path metrics_path = timestampedMetricsPath(metrics_csv_path_);
    if (metrics_path.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(metrics_path.parent_path(), ec);
      if (ec) {
        RCLCPP_WARN(
          get_logger(),
          "failed to create metrics directory: %s (%s)",
          metrics_path.parent_path().string().c_str(),
          ec.message().c_str());
      }
    }

    metrics_actual_csv_path_ = metrics_path.string();
    metrics_file_.open(metrics_actual_csv_path_, std::ios::out | std::ios::trunc);
    if (!metrics_file_.is_open()) {
      RCLCPP_WARN(
        get_logger(),
        "failed to open metrics CSV: %s",
        metrics_actual_csv_path_.c_str());
      return;
    }

    metrics_file_
      << "stamp_sec,state,map_received,path_received,odom_received,path_poses,"
      << "closed_loop_path,use_full_path_search,closest_index,target_index,search_start,search_end,"
      << "collision_points,clusters,applied,failed,hold_cycle,"
      << "window_start,window_end,shift_min,shift_max,"
      << "robot_radius,vehicle_width,vehicle_length,footprint_margin,map_resolution,"
      << "shift_step,max_shift,allow_opposite_shift,collision_check_step,output_poses,"
      << "process_ms,range_ms,hold_check_ms,release_check_ms,detect_collision_ms,cluster_ms,"
      << "shift_sign_ms,find_safe_shift_ms,partial_update_ms,merge_ms,smooth_ms,make_path_ms,"
      << "publish_ms,estimate_shift_sign_calls,estimate_side_clearance_calls,safe_shift_calls,"
      << "safe_shift_successes,safe_shift_failures,shift_candidates_tested,"
      << "corrected_window_collision_calls,deformations_collision_calls,path_range_collision_calls,"
      << "segment_collision_calls,deformation_trigger_collision_calls,footprint_collision_calls,"
      << "scan_trigger_collision_calls,map_trigger_collision_calls,shifted_waypoint_calls,"
      << "deformation_detail,failure_detail\n";
    metrics_file_.flush();

    RCLCPP_INFO(
      get_logger(),
      "metrics CSV enabled: %s",
      metrics_actual_csv_path_.c_str());
  }

  std::filesystem::path timestampedMetricsPath(const std::string & base_path) const
  {
    std::filesystem::path path(base_path);
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&now_time, &tm);

    std::ostringstream stamp;
    stamp << std::put_time(&tm, "%Y%m%d_%H%M%S");

    if (!path.has_extension()) {
      return path / ("global_path_deformer_metrics_" + stamp.str() + ".csv");
    }

    const std::string stem = path.stem().string();
    const std::string extension = path.extension().string();
    return path.parent_path() / (stem + "_" + stamp.str() + extension);
  }

  void writeMetrics(
    const std::string & state,
    int closest_index,
    int target_index,
    int search_start,
    int search_end,
    size_t collision_count,
    size_t cluster_count,
    size_t applied_count,
    size_t failed_count,
    const std::vector<PathDeformation> & deformations,
    const std::vector<std::string> & failure_details,
    double process_ms)
  {
    if (!metrics_enabled_ || !metrics_file_.is_open()) return;

    int window_start = -1;
    int window_end = -1;
    double shift_min = 0.0;
    double shift_max = 0.0;

    if (!deformations.empty()) {
      window_start = deformations.front().window.start;
      window_end = deformations.back().window.end;
      shift_min = deformations.front().shift;
      shift_max = deformations.front().shift;
      for (const auto & deformation : deformations) {
        shift_min = std::min(shift_min, deformation.shift);
        shift_max = std::max(shift_max, deformation.shift);
      }
    }

    metrics_file_
      << now().seconds() << ','
      << state << ','
      << (map_received_ ? 1 : 0) << ','
      << (path_received_ ? 1 : 0) << ','
      << (odom_received_ ? 1 : 0) << ','
      << global_path_.size() << ','
      << (closed_loop_path_ ? 1 : 0) << ','
      << (use_full_path_search_ ? 1 : 0) << ','
      << closest_index << ','
      << target_index << ','
      << search_start << ','
      << search_end << ','
      << collision_count << ','
      << cluster_count << ','
      << applied_count << ','
      << failed_count << ','
      << active_clear_cycles_ << ','
      << window_start << ','
      << window_end << ','
      << shift_min << ','
      << shift_max << ','
      << robot_radius_ << ','
      << vehicle_width_ << ','
      << vehicle_length_ << ','
      << footprint_margin_ << ','
      << map_.info.resolution << ','
      << shift_step_ << ','
      << max_shift_ << ','
      << (allow_opposite_shift_ ? 1 : 0) << ','
      << effectiveCollisionCheckStep() << ','
      << global_path_.size() << ','
      << process_ms << ','
      << process_profile_.range_ms << ','
      << process_profile_.hold_check_ms << ','
      << process_profile_.release_check_ms << ','
      << process_profile_.detect_collision_ms << ','
      << process_profile_.cluster_ms << ','
      << process_profile_.shift_sign_ms << ','
      << process_profile_.find_safe_shift_ms << ','
      << process_profile_.partial_update_ms << ','
      << process_profile_.merge_ms << ','
      << process_profile_.smooth_ms << ','
      << process_profile_.make_path_ms << ','
      << process_profile_.publish_ms << ','
      << profile_counters_.estimate_shift_sign_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.estimate_side_clearance_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.safe_shift_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.safe_shift_successes.load(std::memory_order_relaxed) << ','
      << profile_counters_.safe_shift_failures.load(std::memory_order_relaxed) << ','
      << profile_counters_.shift_candidates_tested.load(std::memory_order_relaxed) << ','
      << profile_counters_.corrected_window_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.deformations_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.path_range_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.segment_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.deformation_trigger_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.footprint_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.scan_trigger_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.map_trigger_collision_calls.load(std::memory_order_relaxed) << ','
      << profile_counters_.shifted_waypoint_calls.load(std::memory_order_relaxed) << ','
      << makeDeformationDetail(deformations) << ','
      << joinDetails(failure_details) << '\n';

    ++metrics_write_count_;
    const int flush_period = std::max(1, metrics_flush_period_);
    if (metrics_write_count_ % static_cast<size_t>(flush_period) == 0) {
      metrics_file_.flush();
    }
  }

  double elapsedProcessMs(const std::chrono::steady_clock::time_point & start) const
  {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
  }

  void resetProfiling()
  {
    process_profile_ = ProcessProfile{};
    profile_counters_.reset();
  }

  std::string makeDeformationDetail(const std::vector<PathDeformation> & deformations) const
  {
    std::ostringstream oss;
    for (size_t i = 0; i < deformations.size(); ++i) {
      if (i > 0) {
        oss << '|';
      }
      const auto & deformation = deformations[i];
      oss
        << deformation.window.start << ':'
        << deformation.window.collision_start << ':'
        << deformation.window.collision_end << ':'
        << deformation.window.end << ':'
        << deformation.shift;
    }
    return oss.str();
  }

  std::string makeFailureDetail(
    int ia,
    int ib,
    const DeformationWindow & window,
    int shift_sign,
    const std::string & reason) const
  {
    std::ostringstream oss;
    oss
      << ia << ':'
      << ib << ':'
      << window.start << ':'
      << window.end << ':'
      << shift_sign << ':'
      << (reason.empty() ? "unknown" : reason);
    return oss.str();
  }

  std::string joinDetails(const std::vector<std::string> & details) const
  {
    std::ostringstream oss;
    for (size_t i = 0; i < details.size(); ++i) {
      if (i > 0) {
        oss << '|';
      }
      oss << details[i];
    }
    return oss.str();
  }

  bool canHoldActiveDeformation(int search_start, bool use_deformation_trigger = false) const
  {
    if (!hasActiveDeformation()) {
      return false;
    }
    if (deformation_hold_cycles_ >= 0 && active_clear_cycles_ >= deformation_hold_cycles_) {
      return false;
    }
    if (closed_loop_path_) {
      const int max_hold_distance = std::max(
        0,
        collision_search_lookahead_waypoints_ +
        forward_waypoints_ +
        std::max(0, deformation_tail_waypoints_));
      if (ringDistance(search_start, active_deformations_.back().window.end) > max_hold_distance) {
        return false;
      }
    } else {
      if (search_start > active_deformations_.back().window.end) {
        return false;
      }
    }
    return !deformationsCollision(active_deformations_, use_deformation_trigger);
  }

  bool canReleaseHoldActiveDeformation() const
  {
    if (!hasActiveDeformation()) {
      return false;
    }
    if (deformation_release_cycles_ < 0) {
      return !deformationsCollision(active_deformations_);
    }
    if (active_clear_cycles_ >= deformation_release_cycles_) {
      return false;
    }
    return !deformationsCollision(active_deformations_);
  }

  bool shouldKeepActiveOverPartialUpdate(size_t failed_clusters) const
  {
    if (!hasActiveDeformation() || failed_clusters == 0 || deformationsCollision(active_deformations_)) {
      return false;
    }
    if (deformation_partial_update_grace_cycles_ < 0) {
      return true;
    }
    return active_collision_grace_cycles_ < deformation_partial_update_grace_cycles_;
  }

  bool hasActiveDeformation() const
  {
    return active_deformation_valid_ && !active_deformations_.empty();
  }

  void clearActiveDeformation()
  {
    active_deformation_valid_ = false;
    active_deformations_.clear();
    active_clear_cycles_ = 0;
    active_collision_grace_cycles_ = 0;
  }

  bool shouldLogStatus()
  {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_status_log_time_ < std::chrono::milliseconds(1000)) {
      return false;
    }

    last_status_log_time_ = now;
    return true;
  }

  void logWaitingState()
  {
    if (!shouldLogStatus()) return;

    RCLCPP_INFO(
      get_logger(),
      "waiting for inputs: map=%s path=%s path_poses=%zu",
      map_received_ ? "ok" : "missing",
      path_received_ ? "ok" : "missing",
      global_path_.size());
  }

  void logFrameMismatch()
  {
    const std::string map_frame_id = map_.header.frame_id.empty() ? "map" : map_.header.frame_id;
    if (!frame_mismatch_logged_ && map_frame_id != path_frame_id_) {
      frame_mismatch_logged_ = true;
      RCLCPP_WARN(
        get_logger(),
        "frame mismatch: map frame=%s global_path frame=%s. This node assumes both coordinates are already in the same frame.",
        map_frame_id.c_str(),
        path_frame_id_.c_str());
    }

    if (!odom_frame_mismatch_logged_ &&
        odom_received_ &&
        !odom_frame_id_.empty() &&
        odom_frame_id_ != path_frame_id_) {
      odom_frame_mismatch_logged_ = true;
      RCLCPP_WARN(
        get_logger(),
        "frame mismatch: odom frame=%s global_path frame=%s. Current path index can be wrong unless odom pose is expressed in the path/map frame.",
        odom_frame_id_.c_str(),
        path_frame_id_.c_str());
    }
  }

  void logDeformationState(
    size_t collision_count,
    size_t cluster_count,
    size_t applied_count,
    size_t failed_count,
    int closest_index,
    int search_start,
    int search_end,
    const std::vector<PathDeformation> & deformations)
  {
    if (deformations.empty()) return;

    const auto & first = deformations.front();
    const auto & last = deformations.back();
    double min_shift = deformations.front().shift;
    double max_shift = deformations.front().shift;
    for (const auto & deformation : deformations) {
      min_shift = std::min(min_shift, deformation.shift);
      max_shift = std::max(max_shift, deformation.shift);
    }

    const bool cluster_changed =
      last_logged_collision_start_ != first.window.collision_start ||
      last_logged_collision_end_ != last.window.collision_end ||
      last_logged_window_start_ != first.window.start ||
      last_logged_window_end_ != last.window.end ||
      last_logged_deformation_count_ != applied_count ||
      last_logged_failed_cluster_count_ != failed_count;
    const bool shift_changed = std::abs(last_logged_shift_ - first.shift) >= 0.02;

    if (!cluster_changed && !shift_changed && !shouldLogStatus()) {
      return;
    }

    last_logged_collision_start_ = first.window.collision_start;
    last_logged_collision_end_ = last.window.collision_end;
    last_logged_window_start_ = first.window.start;
    last_logged_window_end_ = last.window.end;
    last_logged_shift_ = first.shift;
    last_logged_deformation_count_ = applied_count;
    last_logged_failed_cluster_count_ = failed_count;
    last_had_collision_ = true;

    RCLCPP_INFO(
      get_logger(),
      "deforming path: closest=%d target=%d search=[%d,%d] collision_points=%zu clusters=%zu applied=%zu failed=%zu window_span=[%d,%d] shift_range=[%.2f,%.2f] output_poses=%zu",
      closest_index,
      search_start,
      search_start,
      search_end,
      collision_count,
      cluster_count,
      applied_count,
      failed_count,
      first.window.start,
      last.window.end,
      min_shift,
      max_shift,
      global_path_.size());
  }

  void logClearState()
  {
    if (!last_had_collision_ && !shouldLogStatus()) {
      return;
    }

    last_had_collision_ = false;
    last_logged_collision_start_ = -1;
    last_logged_collision_end_ = -1;
    last_logged_window_start_ = -1;
    last_logged_window_end_ = -1;
    last_logged_shift_ = 0.0;
    last_logged_deformation_count_ = 0;
    last_logged_failed_cluster_count_ = 0;

    RCLCPP_INFO(
      get_logger(),
      "path clear: publishing original path output_poses=%zu",
      global_path_.size());
  }

  void logHoldState(int closest_index, int search_start, int search_end, const char * reason)
  {
    if (!shouldLogStatus()) {
      return;
    }

    last_had_collision_ = true;
    if (active_deformations_.empty()) return;

    double min_shift = active_deformations_.front().shift;
    double max_shift = active_deformations_.front().shift;
    for (const auto & deformation : active_deformations_) {
      min_shift = std::min(min_shift, deformation.shift);
      max_shift = std::max(max_shift, deformation.shift);
    }
    const std::string hold_limit = deformation_hold_cycles_ >= 0 ?
      std::to_string(deformation_hold_cycles_) :
      "unlimited";

    RCLCPP_INFO(
      get_logger(),
      "holding deformation: reason=%s closest=%d target=%d search=[%d,%d] applied=%zu window_span=[%d,%d] shift_range=[%.2f,%.2f] hold_cycle=%d/%s",
      reason,
      closest_index,
      search_start,
      search_start,
      search_end,
      active_deformations_.size(),
      active_deformations_.front().window.start,
      active_deformations_.back().window.end,
      min_shift,
      max_shift,
      active_clear_cycles_,
      hold_limit.c_str());
  }

  void logUnsafeState(
    size_t collision_count,
    size_t cluster_count,
    int closest_index,
    int search_start,
    int search_end,
    bool keeping_active_deformation)
  {
    if (!shouldLogStatus()) {
      return;
    }

    last_had_collision_ = true;
    if (keeping_active_deformation && !active_deformations_.empty()) {
      double min_shift = active_deformations_.front().shift;
      double max_shift = active_deformations_.front().shift;
      for (const auto & deformation : active_deformations_) {
        min_shift = std::min(min_shift, deformation.shift);
        max_shift = std::max(max_shift, deformation.shift);
      }

      RCLCPP_WARN(
        get_logger(),
        "no collision-free shift found: closest=%d target=%d search=[%d,%d] collision_points=%zu clusters=%zu max_shift=%.2f check_step=%.3f. Keeping previous deformation applied=%zu window_span=[%d,%d] shift_range=[%.2f,%.2f] instead of reverting to original path.",
        closest_index,
        search_start,
        search_start,
        search_end,
        collision_count,
        cluster_count,
        max_shift_,
        effectiveCollisionCheckStep(),
        active_deformations_.size(),
        active_deformations_.front().window.start,
        active_deformations_.back().window.end,
        min_shift,
        max_shift);
      return;
    }

    RCLCPP_WARN(
      get_logger(),
      "no collision-free shift found: closest=%d target=%d search=[%d,%d] collision_points=%zu clusters=%zu max_shift=%.2f check_step=%.3f. No previous deformation is available, publishing original path.",
      closest_index,
      search_start,
      search_start,
      search_end,
      collision_count,
      cluster_count,
      max_shift_,
      effectiveCollisionCheckStep());
  }

private:
  std::string odom_topic_;
  std::string realtime_map_topic_;
  std::string scan_topic_;
  std::string global_path_topic_;
  std::string output_path_topic_;
  std::string marker_topic_;
  std::string acc_marker_topic_;
  std::string static_obstacle_topic_;
  std::string dynamic_obstacle_topic_;
  std::string obj_flag_topic_;
  std::string obstacle_mode_topic_;
  std::string pcd_static_obstacle_topic_;
  bool metrics_enabled_{true};
  bool latched_input_qos_{true};
  bool obstacle_mode_control_enabled_{true};
  int obstacle_mode_{1};
  bool obstacle_mode_received_{false};
  bool mode1_dynamic_acc_enabled_{false};
  bool mode2_dynamic_acc_enabled_{true};
  bool mode3_dynamic_acc_enabled_{true};
  bool dynamic_acc_enabled_{true};
  bool dynamic_acc_marker_enabled_{true};
  bool dynamic_acc_requires_path_window_{true};
  bool dynamic_acc_path_window_active_{false};
  std::vector<size_t> dynamic_acc_active_point_indices_;
  double dynamic_acc_path_lateral_width_{0.60};
  double dynamic_acc_path_forward_distance_{1.00};
  double dynamic_acc_path_extra_margin_{0.0};
  int dynamic_acc_ttl_ms_{500};
  double dynamic_acc_front_yaw_limit_{0.60};
  double dynamic_acc_lateral_width_{0.60};
  double dynamic_acc_min_distance_{0.80};
  double dynamic_acc_max_distance_{4.00};
  double dynamic_acc_time_headway_{1.00};
  double dynamic_acc_distance_gain_{0.80};
  double dynamic_acc_speed_margin_{0.0};
  double dynamic_acc_min_speed_{0.0};
  bool curvature_speed_limit_enabled_{true};
  double curvature_yaw_slowdown_start_{0.25};
  double curvature_yaw_slowdown_full_{0.80};
  double curvature_min_speed_scale_{0.45};
  double curvature_min_speed_{0.05};
  std::string metrics_csv_path_{
    "/home/rcv/Documents/global_path_deformer/metrics/global_path_deformer_metrics.csv"};
  std::string metrics_actual_csv_path_;
  std::ofstream metrics_file_;
  ProcessProfile process_profile_;
  mutable ProfileCounters profile_counters_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_static_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_dynamic_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_obstacle_mode_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_flag_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcd_static_cloud_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr acc_marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::OccupancyGrid map_;
  std::vector<Waypoint> original_global_path_;
  std::vector<Waypoint> deformed_global_path_;
  std::vector<Waypoint> global_path_;
  bool map_received_{false};
  bool path_received_{false};
  bool odom_received_{false};
  std::string path_frame_id_{"map"};
  std::string odom_frame_id_;
  double odom_x_{0.0};
  double odom_y_{0.0};
  double odom_yaw_{0.0};
  bool tracked_closest_index_valid_{false};
  int tracked_closest_index_{0};
  size_t tracked_closest_path_size_{0};
  std::vector<ScanObstacle> scan_obstacles_;
  std::unordered_map<int64_t, std::vector<size_t>> scan_obstacle_index_;
  std::vector<DynamicPoint> dynamic_points_;
  std::chrono::steady_clock::time_point last_scan_overlay_time_{
    std::chrono::steady_clock::time_point::min()};
  std::chrono::steady_clock::time_point last_dynamic_pointcloud_time_{
    std::chrono::steady_clock::time_point::min()};

  bool map_info_logged_{false};
  uint32_t last_map_width_{0};
  uint32_t last_map_height_{0};
  double last_map_resolution_{0.0};
  size_t last_path_size_{0};
  bool frame_mismatch_logged_{false};
  bool odom_frame_mismatch_logged_{false};

  std::chrono::steady_clock::time_point last_status_log_time_{std::chrono::steady_clock::time_point::min()};
  bool last_had_collision_{false};
  int last_logged_collision_start_{-1};
  int last_logged_collision_end_{-1};
  int last_logged_window_start_{-1};
  int last_logged_window_end_{-1};
  double last_logged_shift_{0.0};
  size_t last_logged_deformation_count_{0};
  size_t last_logged_failed_cluster_count_{0};

  bool active_deformation_valid_{false};
  std::vector<PathDeformation> active_deformations_;
  int active_clear_cycles_{0};
  int active_collision_grace_cycles_{0};

  double robot_radius_{0.14};
  double vehicle_width_{0.28};
  double vehicle_length_{0.512};
  double footprint_margin_{0.10};
  bool map_collision_enabled_{true};
  bool map_occupied_requires_scan_confirmation_{false};
  double map_occupied_scan_confirmation_radius_{0.20};
  bool map_trigger_enabled_{true};
  double map_trigger_extra_margin_{0.15};
  double map_trigger_lateral_width_{0.50};
  double map_trigger_forward_distance_{1.20};
  int occupied_threshold_{65};
  bool unknown_occupied_{true};
  bool scan_overlay_enabled_{true};
  int scan_overlay_ttl_ms_{250};
  double scan_overlay_min_range_{0.10};
  double scan_overlay_max_range_{0.0};
  int scan_overlay_stride_{1};
  double scan_overlay_index_cell_size_{0.25};
  double scan_overlay_extra_margin_{0.05};
  bool scan_overlay_boundary_enabled_{true};
  bool scan_overlay_trigger_enabled_{true};
  double scan_overlay_boundary_extra_margin_{0.05};
  double scan_overlay_trigger_extra_margin_{0.05};
  double scan_overlay_trigger_lateral_width_{0.0};
  double scan_overlay_trigger_forward_distance_{0.0};
  double scan_overlay_near_trigger_radius_{0.0};
  double scan_overlay_sensor_x_{0.0};
  double scan_overlay_sensor_y_{0.0};
  double scan_overlay_sensor_yaw_{0.0};
  int backward_waypoints_{8};
  int forward_waypoints_{20};
  int deformation_tail_waypoints_{0};
  int gap_tolerance_{2};
  int max_collision_cluster_span_{0};
  bool closed_loop_path_{true};
  int closest_index_backward_search_waypoints_{5};
  int closest_index_forward_search_waypoints_{45};
  double closest_index_reset_distance_{1.20};
  bool use_full_path_search_{true};
  int collision_search_start_ahead_waypoints_{8};
  int collision_search_lookahead_waypoints_{160};
  int dynamic_acc_search_start_ahead_waypoints_{3};
  int dynamic_acc_search_lookahead_waypoints_{80};
  int deformation_hold_cycles_{-1};
  int deformation_release_cycles_{60};
  int deformation_partial_update_grace_cycles_{12};
  bool preserve_active_deformations_{true};
  bool active_deformation_hold_uses_trigger_{false};
  bool commit_deformations_to_global_path_{false};
  double shift_step_{0.05};
  double max_shift_{0.50};
  double minimum_deformation_shift_{0.0};
  double deformation_extra_shift_{0.0};
  double deformation_smoothing_alpha_{0.35};
  double shift_direction_clearance_margin_{0.05};
  bool shift_refine_enabled_{true};
  int shift_refine_iterations_{5};
  bool allow_opposite_shift_{true};
  double collision_check_step_{0.0};
  int process_period_ms_{20};
  bool parallel_enabled_{true};
  int parallel_threads_{0};
  int parallel_min_items_{16};
  int metrics_flush_period_{20};
  size_t metrics_write_count_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPathDeformer>());
  rclcpp::shutdown();
  return 0;
}
