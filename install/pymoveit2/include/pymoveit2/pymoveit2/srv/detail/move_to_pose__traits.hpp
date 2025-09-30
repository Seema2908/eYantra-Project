// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from pymoveit2:srv/MoveToPose.idl
// generated code does not contain a copyright notice

#ifndef PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__TRAITS_HPP_
#define PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "pymoveit2/srv/detail/move_to_pose__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace pymoveit2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveToPose_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MoveToPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MoveToPose_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace pymoveit2

namespace rosidl_generator_traits
{

[[deprecated("use pymoveit2::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const pymoveit2::srv::MoveToPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  pymoveit2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use pymoveit2::srv::to_yaml() instead")]]
inline std::string to_yaml(const pymoveit2::srv::MoveToPose_Request & msg)
{
  return pymoveit2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<pymoveit2::srv::MoveToPose_Request>()
{
  return "pymoveit2::srv::MoveToPose_Request";
}

template<>
inline const char * name<pymoveit2::srv::MoveToPose_Request>()
{
  return "pymoveit2/srv/MoveToPose_Request";
}

template<>
struct has_fixed_size<pymoveit2::srv::MoveToPose_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<pymoveit2::srv::MoveToPose_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<pymoveit2::srv::MoveToPose_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace pymoveit2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveToPose_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MoveToPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MoveToPose_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace pymoveit2

namespace rosidl_generator_traits
{

[[deprecated("use pymoveit2::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const pymoveit2::srv::MoveToPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  pymoveit2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use pymoveit2::srv::to_yaml() instead")]]
inline std::string to_yaml(const pymoveit2::srv::MoveToPose_Response & msg)
{
  return pymoveit2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<pymoveit2::srv::MoveToPose_Response>()
{
  return "pymoveit2::srv::MoveToPose_Response";
}

template<>
inline const char * name<pymoveit2::srv::MoveToPose_Response>()
{
  return "pymoveit2/srv/MoveToPose_Response";
}

template<>
struct has_fixed_size<pymoveit2::srv::MoveToPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<pymoveit2::srv::MoveToPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<pymoveit2::srv::MoveToPose_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<pymoveit2::srv::MoveToPose>()
{
  return "pymoveit2::srv::MoveToPose";
}

template<>
inline const char * name<pymoveit2::srv::MoveToPose>()
{
  return "pymoveit2/srv/MoveToPose";
}

template<>
struct has_fixed_size<pymoveit2::srv::MoveToPose>
  : std::integral_constant<
    bool,
    has_fixed_size<pymoveit2::srv::MoveToPose_Request>::value &&
    has_fixed_size<pymoveit2::srv::MoveToPose_Response>::value
  >
{
};

template<>
struct has_bounded_size<pymoveit2::srv::MoveToPose>
  : std::integral_constant<
    bool,
    has_bounded_size<pymoveit2::srv::MoveToPose_Request>::value &&
    has_bounded_size<pymoveit2::srv::MoveToPose_Response>::value
  >
{
};

template<>
struct is_service<pymoveit2::srv::MoveToPose>
  : std::true_type
{
};

template<>
struct is_service_request<pymoveit2::srv::MoveToPose_Request>
  : std::true_type
{
};

template<>
struct is_service_response<pymoveit2::srv::MoveToPose_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__TRAITS_HPP_
