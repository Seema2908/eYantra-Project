// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from pymoveit2:srv/MoveToPose.idl
// generated code does not contain a copyright notice
#include "pymoveit2/srv/detail/move_to_pose__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `mode`
#include "rosidl_runtime_c/string_functions.h"

bool
pymoveit2__srv__MoveToPose_Request__init(pymoveit2__srv__MoveToPose_Request * msg)
{
  if (!msg) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__init(&msg->mode)) {
    pymoveit2__srv__MoveToPose_Request__fini(msg);
    return false;
  }
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
  return true;
}

void
pymoveit2__srv__MoveToPose_Request__fini(pymoveit2__srv__MoveToPose_Request * msg)
{
  if (!msg) {
    return;
  }
  // mode
  rosidl_runtime_c__String__fini(&msg->mode);
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
}

bool
pymoveit2__srv__MoveToPose_Request__are_equal(const pymoveit2__srv__MoveToPose_Request * lhs, const pymoveit2__srv__MoveToPose_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->mode), &(rhs->mode)))
  {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  return true;
}

bool
pymoveit2__srv__MoveToPose_Request__copy(
  const pymoveit2__srv__MoveToPose_Request * input,
  pymoveit2__srv__MoveToPose_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__copy(
      &(input->mode), &(output->mode)))
  {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  return true;
}

pymoveit2__srv__MoveToPose_Request *
pymoveit2__srv__MoveToPose_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Request * msg = (pymoveit2__srv__MoveToPose_Request *)allocator.allocate(sizeof(pymoveit2__srv__MoveToPose_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(pymoveit2__srv__MoveToPose_Request));
  bool success = pymoveit2__srv__MoveToPose_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
pymoveit2__srv__MoveToPose_Request__destroy(pymoveit2__srv__MoveToPose_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    pymoveit2__srv__MoveToPose_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
pymoveit2__srv__MoveToPose_Request__Sequence__init(pymoveit2__srv__MoveToPose_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Request * data = NULL;

  if (size) {
    data = (pymoveit2__srv__MoveToPose_Request *)allocator.zero_allocate(size, sizeof(pymoveit2__srv__MoveToPose_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = pymoveit2__srv__MoveToPose_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        pymoveit2__srv__MoveToPose_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
pymoveit2__srv__MoveToPose_Request__Sequence__fini(pymoveit2__srv__MoveToPose_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      pymoveit2__srv__MoveToPose_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

pymoveit2__srv__MoveToPose_Request__Sequence *
pymoveit2__srv__MoveToPose_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Request__Sequence * array = (pymoveit2__srv__MoveToPose_Request__Sequence *)allocator.allocate(sizeof(pymoveit2__srv__MoveToPose_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = pymoveit2__srv__MoveToPose_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
pymoveit2__srv__MoveToPose_Request__Sequence__destroy(pymoveit2__srv__MoveToPose_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    pymoveit2__srv__MoveToPose_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
pymoveit2__srv__MoveToPose_Request__Sequence__are_equal(const pymoveit2__srv__MoveToPose_Request__Sequence * lhs, const pymoveit2__srv__MoveToPose_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!pymoveit2__srv__MoveToPose_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
pymoveit2__srv__MoveToPose_Request__Sequence__copy(
  const pymoveit2__srv__MoveToPose_Request__Sequence * input,
  pymoveit2__srv__MoveToPose_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(pymoveit2__srv__MoveToPose_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    pymoveit2__srv__MoveToPose_Request * data =
      (pymoveit2__srv__MoveToPose_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!pymoveit2__srv__MoveToPose_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          pymoveit2__srv__MoveToPose_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!pymoveit2__srv__MoveToPose_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

bool
pymoveit2__srv__MoveToPose_Response__init(pymoveit2__srv__MoveToPose_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    pymoveit2__srv__MoveToPose_Response__fini(msg);
    return false;
  }
  return true;
}

void
pymoveit2__srv__MoveToPose_Response__fini(pymoveit2__srv__MoveToPose_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
pymoveit2__srv__MoveToPose_Response__are_equal(const pymoveit2__srv__MoveToPose_Response * lhs, const pymoveit2__srv__MoveToPose_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
pymoveit2__srv__MoveToPose_Response__copy(
  const pymoveit2__srv__MoveToPose_Response * input,
  pymoveit2__srv__MoveToPose_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

pymoveit2__srv__MoveToPose_Response *
pymoveit2__srv__MoveToPose_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Response * msg = (pymoveit2__srv__MoveToPose_Response *)allocator.allocate(sizeof(pymoveit2__srv__MoveToPose_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(pymoveit2__srv__MoveToPose_Response));
  bool success = pymoveit2__srv__MoveToPose_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
pymoveit2__srv__MoveToPose_Response__destroy(pymoveit2__srv__MoveToPose_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    pymoveit2__srv__MoveToPose_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
pymoveit2__srv__MoveToPose_Response__Sequence__init(pymoveit2__srv__MoveToPose_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Response * data = NULL;

  if (size) {
    data = (pymoveit2__srv__MoveToPose_Response *)allocator.zero_allocate(size, sizeof(pymoveit2__srv__MoveToPose_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = pymoveit2__srv__MoveToPose_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        pymoveit2__srv__MoveToPose_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
pymoveit2__srv__MoveToPose_Response__Sequence__fini(pymoveit2__srv__MoveToPose_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      pymoveit2__srv__MoveToPose_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

pymoveit2__srv__MoveToPose_Response__Sequence *
pymoveit2__srv__MoveToPose_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pymoveit2__srv__MoveToPose_Response__Sequence * array = (pymoveit2__srv__MoveToPose_Response__Sequence *)allocator.allocate(sizeof(pymoveit2__srv__MoveToPose_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = pymoveit2__srv__MoveToPose_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
pymoveit2__srv__MoveToPose_Response__Sequence__destroy(pymoveit2__srv__MoveToPose_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    pymoveit2__srv__MoveToPose_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
pymoveit2__srv__MoveToPose_Response__Sequence__are_equal(const pymoveit2__srv__MoveToPose_Response__Sequence * lhs, const pymoveit2__srv__MoveToPose_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!pymoveit2__srv__MoveToPose_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
pymoveit2__srv__MoveToPose_Response__Sequence__copy(
  const pymoveit2__srv__MoveToPose_Response__Sequence * input,
  pymoveit2__srv__MoveToPose_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(pymoveit2__srv__MoveToPose_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    pymoveit2__srv__MoveToPose_Response * data =
      (pymoveit2__srv__MoveToPose_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!pymoveit2__srv__MoveToPose_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          pymoveit2__srv__MoveToPose_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!pymoveit2__srv__MoveToPose_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
