#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "visualization_msgs/msg/marker.hpp"

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

class GlobalPathDeformer : public rclcpp::Node
{
public:
  GlobalPathDeformer()
  : Node("global_path_deformer")
  {
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    const auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    realtime_map_topic_ = declare_parameter<std::string>("realtime_map_topic", "/map");
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan_matched_points2");
    global_path_topic_ = declare_parameter<std::string>("global_path_topic", "/global_path");
    output_path_topic_ = declare_parameter<std::string>("output_path_topic", "/Path");
    marker_topic_ = declare_parameter<std::string>("marker_topic", "/local_path");

    static_obstacle_topic_ = declare_parameter<std::string>("static_obstacle_topic", "/static_obstacle");
    dynamic_obstacle_topic_ = declare_parameter<std::string>("dynamic_obstacle_topic", "/dynamic_obstacle");
    obj_flag_topic_ = declare_parameter<std::string>("obj_flag_topic", "/obj_flag");
    pcd_static_obstacle_topic_ = declare_parameter<std::string>("pcd_static_obstacle_topic", "");

    robot_radius_ = declare_parameter<double>("robot_radius", 0.14);
    vehicle_width_ = declare_parameter<double>("vehicle_width", 0.28);
    vehicle_length_ = declare_parameter<double>("vehicle_length", 0.512);
    footprint_margin_ = declare_parameter<double>("footprint_margin", 0.10);
    occupied_threshold_ = declare_parameter<int>("occupied_threshold", 65);
    unknown_occupied_ = declare_parameter<bool>("unknown_occupied", true);
    scan_overlay_enabled_ = declare_parameter<bool>("scan_overlay_enabled", true);
    scan_overlay_ttl_ms_ = declare_parameter<int>("scan_overlay_ttl_ms", 250);
    scan_overlay_min_range_ = declare_parameter<double>("scan_overlay_min_range", 0.10);
    scan_overlay_max_range_ = declare_parameter<double>("scan_overlay_max_range", 0.0);
    scan_overlay_stride_ = declare_parameter<int>("scan_overlay_stride", 1);
    scan_overlay_extra_margin_ = declare_parameter<double>("scan_overlay_extra_margin", 0.05);

    backward_waypoints_ = declare_parameter<int>("backward_waypoints", 8);
    forward_waypoints_ = declare_parameter<int>("forward_waypoints", 20);
    gap_tolerance_ = declare_parameter<int>("gap_tolerance", 2);
    closed_loop_path_ = declare_parameter<bool>("closed_loop_path", true);
    use_full_path_search_ = declare_parameter<bool>("use_full_path_search", true);
    collision_search_start_ahead_waypoints_ =
      declare_parameter<int>("collision_search_start_ahead_waypoints", 8);
    collision_search_lookahead_waypoints_ =
      declare_parameter<int>("collision_search_lookahead_waypoints", 160);
    deformation_hold_cycles_ = declare_parameter<int>("deformation_hold_cycles", -1);

    shift_step_ = declare_parameter<double>("shift_step", 0.05);
    max_shift_ = declare_parameter<double>("max_shift", 0.50);
    allow_opposite_shift_ = declare_parameter<bool>("allow_opposite_shift", true);
    process_period_ms_ = declare_parameter<int>("process_period_ms", 20);
    collision_check_step_ = declare_parameter<double>("collision_check_step", 0.0);
    metrics_enabled_ = declare_parameter<bool>("metrics_enabled", true);
    metrics_csv_path_ = declare_parameter<std::string>(
      "metrics_csv_path",
      "/home/rcv/Documents/global_path_deformer/metrics/global_path_deformer_metrics.csv");
    metrics_flush_period_ = declare_parameter<int>("metrics_flush_period", 20);
    openMetricsFile();

    sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      realtime_map_topic_, latched_qos,
      std::bind(&GlobalPathDeformer::cb_map, this, std::placeholders::_1));

    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      global_path_topic_, latched_qos,
      std::bind(&GlobalPathDeformer::cb_path, this, std::placeholders::_1));

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_odom, this, std::placeholders::_1));

    if (scan_overlay_enabled_) {
      sub_scan_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        scan_topic_, rclcpp::SensorDataQoS(),
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

    sub_dynamic_ = create_subscription<nav_msgs::msg::Odometry>(
      dynamic_obstacle_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_dynamic, this, std::placeholders::_1));

    sub_flag_ = create_subscription<geometry_msgs::msg::PointStamped>(
      obj_flag_topic_, qos,
      std::bind(&GlobalPathDeformer::cb_flag, this, std::placeholders::_1));

    pub_path_ = create_publisher<nav_msgs::msg::Path>(output_path_topic_, 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(marker_topic_, 1);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(process_period_ms_),
      std::bind(&GlobalPathDeformer::process, this));

    RCLCPP_INFO(get_logger(), "global_path_deformer started. realtime_map_topic=%s, scan_topic=%s, global_path_topic=%s, output_path_topic=%s",
      realtime_map_topic_.c_str(), scan_topic_.c_str(), global_path_topic_.c_str(), output_path_topic_.c_str());
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

    path_frame_id_ = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;
    global_path_.clear();
    global_path_.reserve(msg->poses.size());

    for (const auto & ps : msg->poses) {
      Waypoint wp;
      wp.x = ps.pose.position.x;
      wp.y = ps.pose.position.y;
      wp.v = ps.pose.position.z;  // z is velocity
      global_path_.push_back(wp);
    }

    path_received_ = true;

    if (last_path_size_ != global_path_.size()) {
      last_path_size_ = global_path_.size();
      RCLCPP_INFO(
        get_logger(),
        "global path received: poses=%zu frame=%s",
        global_path_.size(),
        path_frame_id_.c_str());
    }
  }

  void cb_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;
    odom_frame_id_ = msg->header.frame_id;
    odom_received_ = true;
  }

  void cb_scan(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!scan_overlay_enabled_) return;

    const int stride = std::max(1, scan_overlay_stride_);
    std::vector<ScanObstacle> obstacles;
    obstacles.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height) /
      static_cast<size_t>(stride) + 1);

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      size_t index = 0;

      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++index) {
        if (index % static_cast<size_t>(stride) != 0) continue;

        const double x = static_cast<double>(*iter_x);
        const double y = static_cast<double>(*iter_y);
        if (!std::isfinite(x) || !std::isfinite(y)) continue;

        if (odom_received_) {
          const double range = std::hypot(x - odom_x_, y - odom_y_);
          if (range < scan_overlay_min_range_) continue;
          if (scan_overlay_max_range_ > 0.0 && range > scan_overlay_max_range_) continue;
        }

        obstacles.push_back(ScanObstacle{x, y});
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
    last_scan_overlay_time_ = std::chrono::steady_clock::now();
  }

  void cb_static(const geometry_msgs::msg::PointStamped::SharedPtr) {}
  void cb_dynamic(const nav_msgs::msg::Odometry::SharedPtr) {}
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

  double footprintBoundingRadius() const
  {
    const double margin = std::max(0.0, footprint_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    return std::hypot(half_length, half_width);
  }

  bool isFootprintCollision(double x, double y, double yaw) const
  {
    if (isScanOverlayCollision(x, y, yaw)) return true;

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

  bool isScanOverlayCollision(double x, double y, double yaw) const
  {
    if (!scanOverlayFresh()) return false;

    const double margin = std::max(0.0, footprint_margin_ + scan_overlay_extra_margin_);
    const double half_length = std::max(vehicle_length_ * 0.5, robot_radius_) + margin;
    const double half_width = std::max(vehicle_width_ * 0.5, robot_radius_) + margin;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    for (const auto & obstacle : scan_obstacles_) {
      const double vx = obstacle.x - x;
      const double vy = obstacle.y - y;
      const double longitudinal = cos_yaw * vx + sin_yaw * vy;
      const double lateral = -sin_yaw * vx + cos_yaw * vy;
      if (std::abs(longitudinal) <= half_length && std::abs(lateral) <= half_width) {
        return true;
      }
    }

    return false;
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

  bool segmentCollision(double x0, double y0, double x1, double y1) const
  {
    const double length = std::hypot(x1 - x0, y1 - y0);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / effectiveCollisionCheckStep())));
    const double yaw = std::atan2(y1 - y0, x1 - x0);

    for (int k = 0; k <= steps; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(steps);
      const double x = x0 + t * (x1 - x0);
      const double y = y0 + t * (y1 - y0);
      if (isFootprintCollision(x, y, yaw)) return true;
    }

    return false;
  }

  int getClosestPathIndex() const
  {
    if (!odom_received_ || global_path_.empty()) return 0;

    int best_index = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();

    for (int i = 0; i < static_cast<int>(global_path_.size()); ++i) {
      const double dx = global_path_[i].x - odom_x_;
      const double dy = global_path_[i].y - odom_y_;
      const double dist2 = dx * dx + dy * dy;
      if (dist2 < best_dist2) {
        best_dist2 = dist2;
        best_index = i;
      }
    }

    return best_index;
  }

  int getUpcomingPathIndex(int closest_index) const
  {
    if (global_path_.empty()) return 0;

    const int last_index = static_cast<int>(global_path_.size()) - 1;
    return std::clamp(
      closest_index + collision_search_start_ahead_waypoints_,
      0,
      last_index);
  }

  void getCollisionSearchRange(int & closest_index, int & start, int & end) const
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
      end = std::min(last_index, start + collision_search_lookahead_waypoints_);
      return;
    }

    closest_index = 0;
    start = 0;
    end = last_index;
  }

  std::vector<int> detectCollisionIndices(int start, int end) const
  {
    std::vector<int> indices;
    start = std::max(0, start);
    end = std::min(static_cast<int>(global_path_.size()) - 1, end);

    auto add_index = [&indices](int index) {
      indices.push_back(index);
    };

    for (int i = start; i <= end; ++i) {
      if (isFootprintCollision(global_path_[i].x, global_path_[i].y, pathYawAtIndex(i))) {
        add_index(i);
      }

      if (i < end && segmentCollision(
          global_path_[i].x,
          global_path_[i].y,
          global_path_[i + 1].x,
          global_path_[i + 1].y))
      {
        add_index(i);
        add_index(i + 1);
      }
    }

    if (closed_loop_path_ && start == 0 && end == static_cast<int>(global_path_.size()) - 1) {
      const int last = static_cast<int>(global_path_.size()) - 1;
      if (segmentCollision(
          global_path_[last].x,
          global_path_[last].y,
          global_path_[0].x,
          global_path_[0].y))
      {
        add_index(last);
        add_index(0);
      }
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
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
      if (collision_indices[k] - ib <= gap_tolerance_) {
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

  void getWindow(int ia, int ib, int min_start, int & is, int & ie) const
  {
    if (closed_loop_path_ && use_full_path_search_) {
      (void)min_start;
      const int last_index = static_cast<int>(global_path_.size()) - 1;
      if (ia == 0 && ib == last_index) {
        is = 0;
        ie = last_index;
        return;
      }
      is = wrapIndex(ia - backward_waypoints_);
      ie = wrapIndex(ib + forward_waypoints_);
      return;
    }

    is = std::max(min_start, ia - backward_waypoints_);
    ie = std::min(static_cast<int>(global_path_.size()) - 1, ib + forward_waypoints_);
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

    if (dist_end <= dist_collision_end) return 1.0;
    const double eta = static_cast<double>(dist_end - dist_i) /
      static_cast<double>(dist_end - dist_collision_end);
    return smoothRamp(eta);
  }

  int estimateShiftSign(int is, int ie) const
  {
    double scan_signed_sum = 0.0;
    int scan_count = 0;

    if (scanOverlayFresh()) {
      const double radius = footprintBoundingRadius() + std::max(0.0, scan_overlay_extra_margin_);
      forEachIndexInForwardRange(is, ie, [&](int i) {
        double nx = 0.0;
        double ny = 0.0;
        computeNormal(i, nx, ny);
        if (std::hypot(nx, ny) < 1e-6) return;

        for (const auto & obstacle : scan_obstacles_) {
          const double vx = obstacle.x - global_path_[i].x;
          const double vy = obstacle.y - global_path_[i].y;
          if (std::hypot(vx, vy) > radius) continue;
          scan_signed_sum += vx * nx + vy * ny;
          ++scan_count;
        }
      });
    }

    if (scan_count > 0) {
      return scan_signed_sum > 0.0 ? -1 : 1;
    }

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
    bool include_closing_segment) const
  {
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
    if (isFootprintCollision(prev.x, prev.y, pathYawAtIndex(prev_index))) return true;

    if (closed_loop_path_) {
      int i = wrapIndex(first + 1);
      for (size_t steps = 1; steps < global_path_.size(); ++steps) {
        const Waypoint curr = shiftedWaypoint(i, deformations);
        if (segmentCollision(prev.x, prev.y, curr.x, curr.y)) return true;
        prev = curr;
        prev_index = i;
        if (i == last) break;
        i = wrapIndex(i + 1);
      }
    } else {
      for (int i = first + 1; i <= last; ++i) {
        const Waypoint curr = shiftedWaypoint(i, deformations);
        if (segmentCollision(prev.x, prev.y, curr.x, curr.y)) return true;
        prev = curr;
        prev_index = i;
      }
    }

    if (include_closing_segment && prev_index != first) {
      if (segmentCollision(prev.x, prev.y, first_wp.x, first_wp.y)) return true;
    }

    return false;
  }

  bool correctedWindowCollision(
    const DeformationWindow & window,
    double shift,
    const std::vector<PathDeformation> & existing_deformations = {}) const
  {
    if (!window.active()) return false;

    std::vector<PathDeformation> deformations = existing_deformations;
    deformations.push_back(PathDeformation{window, shift});

    const int first = closed_loop_path_ ?
      wrapIndex(window.start - 1) :
      std::max(0, window.start - 1);
    const int last = closed_loop_path_ ?
      wrapIndex(window.end + 1) :
      std::min(static_cast<int>(global_path_.size()) - 1, window.end + 1);

    return pathRangeCollision(deformations, first, last, false);
  }

  bool deformationsCollision(const std::vector<PathDeformation> & deformations) const
  {
    if (deformations.empty()) return false;
    return pathRangeCollision(
      deformations,
      0,
      static_cast<int>(global_path_.size()) - 1,
      closed_loop_path_);
  }

  bool findSafeShift(
    const DeformationWindow & window,
    int preferred_sign,
    const std::vector<PathDeformation> & existing_deformations,
    double & shift,
    std::string & failure_reason) const
  {
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
    for (int k = 0; k <= max_iter; ++k) {
      if (k == 0) {
        if (!correctedWindowCollision(window, 0.0, existing_deformations)) {
          shift = 0.0;
          return true;
        }
        continue;
      }

      for (const int sign : shift_signs) {
        const double d = std::clamp(
          static_cast<double>(sign) * static_cast<double>(k) * shift_step_,
          -max_shift_,
          max_shift_);
        if (!correctedWindowCollision(window, d, existing_deformations)) {
          shift = d;
          return true;
        }
      }
    }

    shift = 0.0;
    failure_reason = allow_opposite_shift_ ?
      "no_safe_shift_within_max_shift_both_directions" :
      "no_safe_shift_within_max_shift";
    return false;
  }

  nav_msgs::msg::Path makeCorrectedPath(
    const std::vector<PathDeformation> & deformations) const
  {
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = path_frame_id_;
    path.poses.reserve(global_path_.size());

    for (int i = 0; i < static_cast<int>(global_path_.size()); ++i) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;

      double x = global_path_[i].x;
      double y = global_path_[i].y;

      const Waypoint shifted = shiftedWaypoint(i, deformations);
      x = shifted.x;
      y = shifted.y;

      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = global_path_[i].v;  // preserve velocity in z
      ps.pose.orientation.w = 1.0;
      path.poses.push_back(ps);
    }

    return path;
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

  void process()
  {
    const auto process_start = std::chrono::steady_clock::now();

    if (!map_received_ || !path_received_ || global_path_.size() < 3) {
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
    logFrameMismatch();

    int closest_index = 0;
    int search_start = 0;
    int search_end = static_cast<int>(global_path_.size()) - 1;
    getCollisionSearchRange(closest_index, search_start, search_end);

    const auto collision_indices = detectCollisionIndices(search_start, search_end);

    std::vector<PathDeformation> deformations;
    std::string state = "clear";
    size_t cluster_count = 0;
    size_t failed_clusters = 0;
    std::vector<std::string> failure_details;

    if (!collision_indices.empty()) {
      const auto clusters = getCollisionClusters(collision_indices);
      cluster_count = clusters.size();

      if (canHoldActiveDeformation(search_start)) {
        deformations = active_deformations_;
        ++active_clear_cycles_;
        state = "holding";
        logHoldState(closest_index, search_start, search_end, "active path remains collision-free");
      } else {
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

          const int shift_sign = estimateShiftSign(ia, ib);
          double d_req = 0.0;
          std::string failure_reason;
          const bool safe_shift_found =
            findSafeShift(window, shift_sign, deformations, d_req, failure_reason);
          if (safe_shift_found) {
            deformations.push_back(PathDeformation{window, d_req});
            min_window_start = window.end + 1;
          } else {
            ++failed_clusters;
            failure_details.push_back(makeFailureDetail(ia, ib, window, shift_sign, failure_reason));
          }
        }

        if (!deformations.empty()) {
          active_deformation_valid_ = true;
          active_deformations_ = deformations;
          active_clear_cycles_ = 0;
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
        } else if (canHoldActiveDeformation(search_start)) {
          deformations = active_deformations_;
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
    } else if (canHoldActiveDeformation(search_start)) {
      deformations = active_deformations_;
      ++active_clear_cycles_;
      state = "holding";
      logHoldState(closest_index, search_start, search_end, "temporary clear detection");
    } else {
      clearActiveDeformation();
      logClearState();
    }

    const auto corrected_path = makeCorrectedPath(deformations);
    pub_path_->publish(corrected_path);
    publishMarker(corrected_path);
    writeMetrics(
      state,
      closest_index,
      search_start,
      search_start,
      search_end,
      collision_indices.size(),
      cluster_count,
      deformations.size(),
      failed_clusters,
      deformations,
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
      << "process_ms,deformation_detail,failure_detail\n";
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

  bool canHoldActiveDeformation(int search_start) const
  {
    if (!hasActiveDeformation()) {
      return false;
    }
    if (deformation_hold_cycles_ >= 0 && active_clear_cycles_ >= deformation_hold_cycles_) {
      return false;
    }
    if (search_start > active_deformations_.back().window.end) {
      return false;
    }
    return !deformationsCollision(active_deformations_);
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
  std::string static_obstacle_topic_;
  std::string dynamic_obstacle_topic_;
  std::string obj_flag_topic_;
  std::string pcd_static_obstacle_topic_;
  bool metrics_enabled_{true};
  std::string metrics_csv_path_{
    "/home/rcv/Documents/global_path_deformer/metrics/global_path_deformer_metrics.csv"};
  std::string metrics_actual_csv_path_;
  std::ofstream metrics_file_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_static_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_dynamic_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_flag_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcd_static_cloud_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::OccupancyGrid map_;
  std::vector<Waypoint> global_path_;
  bool map_received_{false};
  bool path_received_{false};
  bool odom_received_{false};
  std::string path_frame_id_{"map"};
  std::string odom_frame_id_;
  double odom_x_{0.0};
  double odom_y_{0.0};
  std::vector<ScanObstacle> scan_obstacles_;
  std::chrono::steady_clock::time_point last_scan_overlay_time_{
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

  double robot_radius_{0.14};
  double vehicle_width_{0.28};
  double vehicle_length_{0.512};
  double footprint_margin_{0.10};
  int occupied_threshold_{65};
  bool unknown_occupied_{true};
  bool scan_overlay_enabled_{true};
  int scan_overlay_ttl_ms_{250};
  double scan_overlay_min_range_{0.10};
  double scan_overlay_max_range_{0.0};
  int scan_overlay_stride_{1};
  double scan_overlay_extra_margin_{0.05};
  int backward_waypoints_{8};
  int forward_waypoints_{20};
  int gap_tolerance_{2};
  bool closed_loop_path_{true};
  bool use_full_path_search_{true};
  int collision_search_start_ahead_waypoints_{8};
  int collision_search_lookahead_waypoints_{160};
  int deformation_hold_cycles_{-1};
  double shift_step_{0.05};
  double max_shift_{0.50};
  bool allow_opposite_shift_{true};
  double collision_check_step_{0.0};
  int process_period_ms_{20};
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
