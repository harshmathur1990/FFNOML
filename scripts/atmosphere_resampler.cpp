#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/interpolators/pchip.hpp>

namespace {

using ProgressCallback = void (*)(std::size_t completed, void *context);
using BoostPchip = boost::math::interpolators::pchip<std::vector<double>>;
using BoostBarycentric = boost::math::interpolators::barycentric_rational<double>;

enum InterpolationMethod {
    METHOD_SKIP = -1,
    METHOD_NEAREST = 0,
    METHOD_LINEAR = 1,
    METHOD_CUBIC = 2,
};

std::size_t product(
    const std::size_t *shape, const int start, const int stop
) {
    std::size_t value = 1;
    for (int index = start; index < stop; ++index) {
        value *= shape[index];
    }
    return value;
}

double input_value(
    const float *values,
    const std::size_t outer,
    const std::size_t coordinate,
    const std::size_t inner,
    const std::size_t coordinate_count,
    const std::size_t inner_count,
    const bool log_space
) {
    const auto offset =
        (outer * coordinate_count + coordinate) * inner_count + inner;
    const double value = static_cast<double>(values[offset]);
    return log_space ? std::log10(value) : value;
}

class SeriesInterpolator {
public:
    SeriesInterpolator(
        const double *coordinates,
        const float *values,
        const std::size_t outer,
        const std::size_t inner,
        const std::size_t count,
        const std::size_t inner_count,
        const int method,
        const bool log_space
    ) : method_(method) {
        x_.reserve(count);
        y_.reserve(count);
        const bool ascending = coordinates[0] < coordinates[count - 1];
        for (std::size_t ordered = 0; ordered < count; ++ordered) {
            const std::size_t source = ascending ? ordered : count - 1 - ordered;
            x_.push_back(coordinates[source]);
            y_.push_back(input_value(
                values, outer, source, inner, count, inner_count, log_space
            ));
        }

        if (method_ == METHOD_CUBIC) {
            if (count >= 4) {
                // Boost PCHIP supplies a shape-preserving piecewise cubic on
                // arbitrary, strictly increasing coordinates.
                pchip_ = std::make_unique<BoostPchip>(
                    std::vector<double>(x_), std::vector<double>(y_)
                );
            } else {
                // Boost PCHIP requires four samples. Boost's barycentric
                // interpolator is the exact low-order fallback for 2/3 points.
                barycentric_ = std::make_unique<BoostBarycentric>(
                    x_.data(), y_.data(), count, count - 1
                );
            }
        }
    }

    double operator()(const double target) const {
        if (method_ == METHOD_CUBIC) {
            return pchip_ ? (*pchip_)(target) : (*barycentric_)(target);
        }

        auto right = std::upper_bound(x_.begin(), x_.end(), target);
        if (right == x_.begin()) {
            right = x_.begin() + 1;
        } else if (right == x_.end()) {
            right = x_.end() - 1;
        }
        const auto right_index = static_cast<std::size_t>(right - x_.begin());
        const auto left_index = right_index - 1;
        if (method_ == METHOD_NEAREST) {
            return target - x_[left_index] > x_[right_index] - target
                ? y_[right_index]
                : y_[left_index];
        }
        const double fraction =
            (target - x_[left_index]) / (x_[right_index] - x_[left_index]);
        return y_[left_index] * (1.0 - fraction) + y_[right_index] * fraction;
    }

private:
    int method_;
    std::vector<double> x_;
    std::vector<double> y_;
    std::unique_ptr<BoostPchip> pchip_;
    std::unique_ptr<BoostBarycentric> barycentric_;
};

void interpolate_axis(
    const float *input,
    float *output,
    const std::size_t *input_shape,
    const int dimensions,
    const int axis,
    const double *coordinates,
    const double *targets,
    const std::size_t target_count,
    const int method,
    const bool log_space,
    std::size_t &completed,
    const std::size_t report_interval,
    std::size_t &next_report,
    const ProgressCallback callback,
    void *context
) {
    const std::size_t coordinate_count = input_shape[axis];
    const std::size_t outer_count = product(input_shape, 0, axis);
    const std::size_t inner_count = product(input_shape, axis + 1, dimensions);
    if (coordinate_count < 2) {
        throw std::invalid_argument("cannot upsample a one-point axis");
    }

    for (std::size_t outer = 0; outer < outer_count; ++outer) {
        for (std::size_t inner = 0; inner < inner_count; ++inner) {
            const SeriesInterpolator interpolate(
                coordinates, input, outer, inner, coordinate_count,
                inner_count, method, log_space
            );
            for (std::size_t target_index = 0;
                 target_index < target_count;
                 ++target_index) {
                double value = interpolate(targets[target_index]);
                if (log_space) {
                    value = std::pow(10.0, value);
                }
                output[(outer * target_count + target_index) * inner_count + inner] =
                    static_cast<float>(value);
            }
            completed += target_count;
            if (callback && completed >= next_report) {
                callback(completed, context);
                next_report = completed + report_interval;
            }
        }
    }
}

}  // namespace

extern "C" int resample_atmosphere_field(
    const float *input,
    const std::size_t *input_shape,
    const int dimensions,
    const double *old_x,
    const double *old_y,
    const double *old_z,
    const double *new_x,
    const double *new_y,
    const double *new_z,
    const std::size_t *target_shape,
    const int *methods,
    const int log_space,
    float *output,
    const std::size_t total_work,
    const ProgressCallback callback,
    void *context
) {
    try {
        if (!input || !input_shape || !target_shape || !methods || !output ||
            dimensions < 3 || dimensions > 4) {
            return 2;
        }
        const double *old_coordinates[3] = {old_x, old_y, old_z};
        const double *new_coordinates[3] = {new_x, new_y, new_z};
        std::size_t current_shape[4] = {1, 1, 1, 1};
        std::copy(input_shape, input_shape + dimensions, current_shape);
        const float *current = input;
        std::vector<float> current_storage;
        std::size_t completed = 0;
        const std::size_t report_interval = total_work / 100 + 1;
        std::size_t next_report = report_interval;

        for (int axis = 0; axis < 3; ++axis) {
            if (methods[axis] == METHOD_SKIP) {
                continue;
            }
            std::size_t next_shape[4];
            std::copy(current_shape, current_shape + dimensions, next_shape);
            next_shape[axis] = target_shape[axis];
            std::vector<float> next(product(next_shape, 0, dimensions));
            interpolate_axis(
                current, next.data(), current_shape, dimensions, axis,
                old_coordinates[axis], new_coordinates[axis], target_shape[axis],
                methods[axis], log_space != 0, completed, report_interval,
                next_report, callback, context
            );
            current_storage = std::move(next);
            current = current_storage.data();
            std::copy(next_shape, next_shape + dimensions, current_shape);
        }

        std::copy(
            current,
            current + product(current_shape, 0, dimensions),
            output
        );
        if (callback) {
            callback(total_work, context);
        }
        return 0;
    } catch (const std::bad_alloc &) {
        return 1;
    } catch (...) {
        return 2;
    }
}
