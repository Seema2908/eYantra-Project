// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from pymoveit2:srv/MoveToPose.idl
// generated code does not contain a copyright notice

#ifndef PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__BUILDER_HPP_
#define PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "pymoveit2/srv/detail/move_to_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace pymoveit2
{

namespace srv
{

namespace builder
{

class Init_MoveToPose_Request_yaw
{
public:
  explicit Init_MoveToPose_Request_yaw(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  ::pymoveit2::srv::MoveToPose_Request yaw(::pymoveit2::srv::MoveToPose_Request::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return std::move(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_pitch
{
public:
  explicit Init_MoveToPose_Request_pitch(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  Init_MoveToPose_Request_yaw pitch(::pymoveit2::srv::MoveToPose_Request::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_MoveToPose_Request_yaw(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_roll
{
public:
  explicit Init_MoveToPose_Request_roll(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  Init_MoveToPose_Request_pitch roll(::pymoveit2::srv::MoveToPose_Request::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_MoveToPose_Request_pitch(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_z
{
public:
  explicit Init_MoveToPose_Request_z(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  Init_MoveToPose_Request_roll z(::pymoveit2::srv::MoveToPose_Request::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_MoveToPose_Request_roll(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_y
{
public:
  explicit Init_MoveToPose_Request_y(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  Init_MoveToPose_Request_z y(::pymoveit2::srv::MoveToPose_Request::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_MoveToPose_Request_z(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_x
{
public:
  explicit Init_MoveToPose_Request_x(::pymoveit2::srv::MoveToPose_Request & msg)
  : msg_(msg)
  {}
  Init_MoveToPose_Request_y x(::pymoveit2::srv::MoveToPose_Request::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_MoveToPose_Request_y(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

class Init_MoveToPose_Request_mode
{
public:
  Init_MoveToPose_Request_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveToPose_Request_x mode(::pymoveit2::srv::MoveToPose_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_MoveToPose_Request_x(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::pymoveit2::srv::MoveToPose_Request>()
{
  return pymoveit2::srv::builder::Init_MoveToPose_Request_mode();
}

}  // namespace pymoveit2


namespace pymoveit2
{

namespace srv
{

namespace builder
{

class Init_MoveToPose_Response_message
{
public:
  explicit Init_MoveToPose_Response_message(::pymoveit2::srv::MoveToPose_Response & msg)
  : msg_(msg)
  {}
  ::pymoveit2::srv::MoveToPose_Response message(::pymoveit2::srv::MoveToPose_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Response msg_;
};

class Init_MoveToPose_Response_success
{
public:
  Init_MoveToPose_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveToPose_Response_message success(::pymoveit2::srv::MoveToPose_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_MoveToPose_Response_message(msg_);
  }

private:
  ::pymoveit2::srv::MoveToPose_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::pymoveit2::srv::MoveToPose_Response>()
{
  return pymoveit2::srv::builder::Init_MoveToPose_Response_success();
}

}  // namespace pymoveit2

#endif  // PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__BUILDER_HPP_
