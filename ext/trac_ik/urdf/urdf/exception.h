#ifndef URDF_EXCEPTION_H
#define URDF_EXCEPTION_H

#include <stdexcept>
#include <string>

namespace urdf {
class URDFParseError : public std::exception {
  std::string text;

 public:
  URDFParseError(const std::string& error_msg) : std::exception() {
    text = error_msg;
  }

  virtual const char* what() const noexcept { return text.c_str(); }
};
}  // namespace urdf

#endif
