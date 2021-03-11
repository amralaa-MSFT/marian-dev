#pragma once

#include <string>
#include <functional>
namespace marian {

struct ModelTask {
  virtual ~ModelTask() {}
  virtual void run() = 0;
};

struct ModelCallbackTask {
  virtual ~ModelCallbackTask() {}
  virtual void registerCallback(void (*)(int, const char*, void*), void*) = 0;
  virtual void run(const std::string&) = 0;
};

struct ModelServiceTask {
  virtual ~ModelServiceTask() {}
  virtual std::string run(const std::string&) = 0;
};
}  // namespace marian
