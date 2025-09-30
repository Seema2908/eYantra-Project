// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from pymoveit2:srv/MoveToPose.idl
// generated code does not contain a copyright notice

#ifndef PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__STRUCT_H_
#define PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'mode'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/MoveToPose in the package pymoveit2.
typedef struct pymoveit2__srv__MoveToPose_Request
{
  rosidl_runtime_c__String mode;
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
} pymoveit2__srv__MoveToPose_Request;

// Struct for a sequence of pymoveit2__srv__MoveToPose_Request.
typedef struct pymoveit2__srv__MoveToPose_Request__Sequence
{
  pymoveit2__srv__MoveToPose_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} pymoveit2__srv__MoveToPose_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/MoveToPose in the package pymoveit2.
typedef struct pymoveit2__srv__MoveToPose_Response
{
  bool success;
  rosidl_runtime_c__String message;
} pymoveit2__srv__MoveToPose_Response;

// Struct for a sequence of pymoveit2__srv__MoveToPose_Response.
typedef struct pymoveit2__srv__MoveToPose_Response__Sequence
{
  pymoveit2__srv__MoveToPose_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} pymoveit2__srv__MoveToPose_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PYMOVEIT2__SRV__DETAIL__MOVE_TO_POSE__STRUCT_H_
