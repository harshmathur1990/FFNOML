#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double BK = 1.3806488E-16;
constexpr double HH = 6.62606957E-27;
constexpr double CC = 2.99792458E10;
constexpr double AMU = 1.660538921E-24;
constexpr double EV = 1.602176565E-12;
constexpr int NCONTR = 28;
constexpr double PREC = 1.0e-5;

const std::array<double, 99> ABUND_RAW = {{
    -0.04048,-1.07,-10.95,-10.89,-9.44,-3.48,-3.99,-3.11,-7.48,-3.95,
    -5.71,-4.46,-5.57,-4.49,-6.59,-4.83,-6.54,-5.48,-6.82,-5.68,
    -8.94,-7.05,-8.04,-6.37,-6.65,-4.50,-7.12,-5.79,-7.83,-7.44,
    -9.16,-8.63,-9.67,-8.69,-9.41,-8.81,-9.44,-9.14,-9.80,-9.54,
    -10.62,-10.12,-20.00,-10.20,-10.92,-10.35,-11.10,-10.18,-10.58,
    -10.04,-11.04,-9.80,-10.53,-9.81,-10.92,-9.91,-10.82,-10.49,
    -11.33,-10.54,-20.00,-11.04,-11.53,-10.92,-11.94,-10.94,-11.78,
    -11.11,-12.04,-10.96,-11.28,-11.16,-11.91,-10.93,-11.77,-10.59,
    -10.69,-10.24,-11.03,-10.95,-11.14,-10.19,-11.33,-20.00,-20.00,
    -20.00,-20.00,-20.00,-20.00,-11.92,-20.00,-12.51,-20.00,-20.00,
    -20.00,-20.00,-20.00,-20.00,-20.00
}};

const std::array<double, 99> AMASS = {{
    1.008,4.003,6.941,9.012,10.811,12.011,14.007,15.999,18.998,20.179,
    22.990,24.305,26.982,28.086,30.974,32.060,35.453,39.948,39.102,
    40.080,44.956,47.900,50.941,51.996,54.938,55.847,58.933,58.710,
    63.546,65.370,69.720,72.590,74.922,78.960,79.904,83.800,85.468,
    87.620,88.906,91.220,92.906,95.940,98.906,101.070,102.905,106.400,
    107.868,112.400,114.820,118.690,121.750,127.600,126.905,131.300,
    132.905,137.340,138.906,140.120,140.908,144.240,146.000,150.400,
    151.960,157.250,158.925,162.500,164.930,167.260,168.934,170.040,
    174.970,178.490,180.948,183.850,186.200,190.200,192.200,195.090,
    196.967,200.590,204.370,207.190,208.981,210.000,210.000,222.000,
    223.000,226.025,227.000,232.038,230.040,238.029,237.048,242.000,
    242.000,245.000,248.000,252.000,253.000
}};

uint32_t read_u32_be(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Unexpected EOF while reading uint32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

double read_f64_be(std::ifstream& in) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in) throw std::runtime_error("Unexpected EOF while reading double");
    uint64_t u = (uint64_t(b[0]) << 56) | (uint64_t(b[1]) << 48) |
                 (uint64_t(b[2]) << 40) | (uint64_t(b[3]) << 32) |
                 (uint64_t(b[4]) << 24) | (uint64_t(b[5]) << 16) |
                 (uint64_t(b[6]) << 8) | uint64_t(b[7]);
    double d;
    std::memcpy(&d, &u, sizeof(double));
    return d;
}

struct Element {
    uint32_t nstage = 0;
    uint32_t npf = 0;
    std::vector<double> pf;
    std::vector<double> eion;
};

struct GasResult {
    double pg;
    double fe;
};

class WittEOS {
public:
    explicit WittEOS(const char* pf_path) {
        for (size_t i = 0; i < ABUND.size(); ++i) ABUND[i] = std::pow(10.0, ABUND_RAW[i]);
        double abtot = 0.0;
        for (double v : ABUND) abtot += v;
        for (double& v : ABUND) v /= abtot;
        ab_others = 0.0;
        for (size_t i = 1; i < ABUND.size(); ++i) ab_others += ABUND[i];
        ab_others /= ABUND[0];

        avw = 0.0;
        for (size_t i = 0; i < ABUND.size(); ++i) avw += ABUND[i] * AMASS[i];
        const double muH = avw / AMASS[0] / ABUND[0];
        rho_from_H = muH * AMASS[0] * AMU / BK;
        avw *= AMU;

        init_pf_data(pf_path);
    }

    double ne_from_rho_m3(double temp, double rho_kg_m3) const {
        const double rho_cgs = rho_kg_m3 * 1.0e-3;
        const double pgas = pg_from_rho(temp, rho_cgs);
        const double pe = pe_from_pg(temp, pgas);
        return pe / (BK * temp) * 1.0e6;
    }

private:
    std::array<double, 99> ABUND{};
    std::vector<double> tpf;
    std::array<Element, 99> el;
    double avw = 0.0;
    double rho_from_H = 0.0;
    double ab_others = 0.0;

    void init_pf_data(const char* pf_path) {
        std::ifstream in(pf_path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("Could not open PF file: ") + pf_path);
        const uint32_t npf = read_u32_be(in);
        tpf.resize(npf);
        for (uint32_t i = 0; i < npf; ++i) tpf[i] = read_f64_be(in);

        for (size_t ii = 0; ii < el.size(); ++ii) {
            Element e;
            e.npf = read_u32_be(in);
            e.nstage = read_u32_be(in);
            e.pf.resize(size_t(npf) * e.nstage);
            for (double& v : e.pf) v = read_f64_be(in);
            e.eion.resize(e.nstage);
            for (double& v : e.eion) v = read_f64_be(in) * HH * CC / EV;
            el[ii] = std::move(e);
        }
    }

    static double acota(double x, double x0, double x1) {
        if (x < x0) x = x0;
        if (x > x1) x = x1;
        return x;
    }

    static double acotasig(double x, double x0, double x1) {
        if (x < 0.0) return -acota(-x, x0, x1);
        return acota(x, x0, x1);
    }

    static double sign(double a, double b) {
        return std::abs(a) * (b / std::abs(b));
    }

    static void molecb(double X, double y[2]) {
        y[0] = -11.206998 + X * (2.7942767 + X * (7.9196803E-2 - X * 2.4790744E-2));
        y[1] = -12.533505 + X * (4.9251644 + X * (-5.6191273E-2 + X * 3.2687661E-3));
    }

    double itep1(const std::vector<double>& x, const double* y, double xx) const {
        if (xx <= x.front()) return y[0];
        if (xx >= x.back()) return y[x.size() - 1];
        auto it = std::upper_bound(x.begin(), x.end(), xx);
        size_t p0 = size_t(it - x.begin());
        size_t p1 = p0 - 1;
        const double dx = x[p1] - x[p0];
        const double u1 = (xx - x[p0]) / dx;
        const double u0 = 1.0 - u1;
        return u0 * y[p0] + u1 * y[p1];
    }

    void partition_f(int n, double t, int only, double* out, int& count) const {
        const Element& e = el[n];
        count = int(e.nstage);
        if (only > 0) count = std::min(count, only);
        for (int ii = 0; ii < count; ++ii) {
            out[ii] = itep1(tpf, e.pf.data() + size_t(ii) * tpf.size(), t);
        }
        for (int ii = count; ii < only; ++ii) out[ii] = 0.0;
        if (only > 0) count = only;
    }

    double saha(double theta, double eion, double u1, double u2, double pe) const {
        return u2 * std::exp(2.302585093 * (9.0804625434325867 - theta * eion)) /
               (u1 * pe * std::pow(theta, 2.5));
    }

    double init_pe_from_pg(double t, double pg) const {
        const double nu = ABUND[0];
        const double saha_h = std::pow(10.0, -0.4771 + 2.5 * std::log10(t) -
                                             std::log10(pg) - (13.6 * 5040.0 / t));
        const double aaa = 1.0 + saha_h;
        const double bbb = -(nu - 1.0) * saha_h;
        const double ccc = -saha_h * nu;
        const double ybh = (-bbb + std::sqrt(bbb * bbb - 4.0 * aaa * ccc)) / (2.0 * aaa);
        return pg * ybh / (1.0 + ybh);
    }

    double pe_from_pg(double t, double pg) const {
        double dif = 1.1;
        double pe = init_pe_from_pg(t, pg);
        double ope = pe;
        int it = 0;
        double fe = 0.0;
        while ((std::abs(dif) > PREC) && (it < 250)) {
            pe = (ope + pe) * 0.5;
            ope = pe;
            auto out = pe_pg(t, pe, pg);
            pe = out.first;
            fe = out.second;
            (void)fe;
            dif = 2.0 * std::abs(pe - ope) / (pe + ope);
            it += 1;
        }
        return pe;
    }

    std::pair<double, double> pe_pg(double t, double pe_in, double pgas) const {
        double pe = pe_in;
        double g1 = 0.0;
        const double theta = 5040.0 / t;
        double g4 = 0.0, g5 = 0.0;
        if (pe < 0.0) {
            pe = 1.0e-15;
        } else {
            double cmol[2];
            molecb(theta, cmol);
            cmol[0] = acota(cmol[0], -30.0, 30.0);
            cmol[1] = acota(cmol[1], -30.0, 30.0);
            g4 = pe * std::pow(10.0, cmol[0]);
            g5 = pe * std::pow(10.0, cmol[1]);
        }

        double u[8];
        int count = 0;
        partition_f(0, t, 3, u, count);
        const double g2 = saha(theta, el[0].eion[0], u[0], u[1], pe);
        double g3 = saha(theta, 0.754, 1.0, u[0], pe);
        g3 = 1.0 / acota(g3, 1.0e-30, 1.0e30);

        for (int ii = 1; ii < NCONTR; ++ii) {
            const double alfai = ABUND[ii] / ABUND[0];
            partition_f(ii, t, 3, u, count);
            const double aa = saha(theta, el[ii].eion[0], u[0], u[1], pe);
            const double bb = saha(theta, el[ii].eion[1], u[1], u[2], pe);
            const double cc = 1.0 + aa * (1.0 + bb);
            g1 += alfai / cc * aa * (1.0 + 2.0 * bb);
        }

        double a = 1.0 + g2 + g3;
        const double b = 2.0 * (1.0 + g2 / g5 * g4);
        const double c = g5;
        double d = g2 - g3;
        const double e = g2 / g5 * g4;
        a = acotasig(a, 1.0e-15, 1.0e15);
        d = acotasig(d, 1.0e-15, 1.0e15);

        const double c1 = c * b * b + a * d * b - e * a * a;
        const double c2 = 2.0 * a * e - d * b + a * b * g1;
        const double c3 = -(e + b * g1);
        double f1 = 0.5 * c2 / c1;
        f1 = -f1 + sign(1.0, c1) * std::sqrt(f1 * f1 - c3 / c1);
        double f5 = (1.0 - a * f1) / b;
        double f4 = e * f5;
        const double f3 = g3 * f1;
        const double f2 = g2 * f1;
        double fe = acota(f2 - f3 + f4 + g1, 1.0e-30, 1.0e30);
        double phtot = pe / fe;

        if (f5 <= 1.0e-4) {
            double diff = 1.0;
            const double const6 = g5 / pe * f1 * f1;
            const double const7 = f2 - f3 + g1;
            int it = 0;
            while ((diff > 1.0e-5) && (it < 5)) {
                const double of5 = f5;
                f5 = phtot * const6;
                f4 = e * f5;
                fe = const7 + f4;
                phtot = pe / fe;
                diff = 0.5 * std::abs(f5 - of5) / (f5 + of5);
                it += 1;
            }
        }

        pe = pgas / (1.0 + (f1 + f2 + f3 + f4 + f5 + ab_others) / fe);
        if (pe <= 0.0) pe = 1.0e-15;
        return {pe, fe};
    }

    GasResult gasc(double t, double pe) const {
        const double theta = 5040.0 / t;
        double cmol[2];
        molecb(theta, cmol);
        const double g4 = std::pow(10.0, cmol[0]);
        const double g5 = std::pow(10.0, cmol[1]);

        double u[8];
        int count = 0;
        partition_f(0, t, 0, u, count);
        const double g2 = saha(theta, el[0].eion[0], u[0], u[1], pe);
        const double g3 = 1.0 / saha(theta, 0.754, 1.0, u[0], pe);
        double g1 = 0.0;

        for (int ii = 1; ii < NCONTR; ++ii) {
            const double alfai = ABUND[ii] / ABUND[0];
            partition_f(ii, t, 3, u, count);
            const double aa = saha(theta, el[ii].eion[0], u[0], u[1], pe);
            const double bb = saha(theta, el[ii].eion[1], u[1], u[2], pe);
            const double cc = 1.0 + aa * (1.0 + bb);
            const double pp_i = alfai / cc;
            g1 += pp_i * aa * (1.0 + 2.0 * bb);
        }

        const double a = 1.0 + g2 + g3;
        const double e = g2 / g5 * g4;
        const double b = 2.0 * (1.0 + e);
        const double c = g5;
        const double d = g2 - g3;
        const double c1 = c * b * b + a * d * b - e * a * a;
        const double c2 = 2.0 * a * e - d * b + a * b * g1;
        const double c3 = -(e + b * g1);
        double f1 = 0.5 * c2 / c1;
        f1 = -f1 + sign(1.0, c1) * std::sqrt(f1 * f1 - c3 / c1);
        double f5 = (1.0 - a * f1) / b;
        double f4 = e * f5;
        const double f3 = g3 * f1;
        const double f2 = g2 * f1;
        double fe = f2 - f3 + f4 + g1;
        double phtot = pe / fe;

        if (f5 <= 1.0e-5) {
            double diff = 1.0;
            const double const6 = g5 / pe * f1 * f1;
            const double const7 = f2 - f3 + g1;
            int it = 0;
            while ((diff > 1.0e-5) && (it < 5)) {
                const double of5 = f5;
                f5 = phtot * const6;
                f4 = e * f5;
                fe = const7 + f4;
                phtot = pe / fe;
                diff = 0.5 * std::abs(f5 - of5) / (f5 + of5);
                it += 1;
            }
        }

        const double pg = pe * (1.0 + (f1 + f2 + f3 + f4 + f5 + ab_others) / fe);
        return {pg, fe};
    }

    double pg_from_pe(double t, double pe) const {
        return gasc(t, pe).pg;
    }

    std::pair<double, double> pg_from_pe_get_fe(double t, double pe) const {
        const auto out = gasc(t, pe);
        return {out.pg, out.fe};
    }

    double rho_from_pe(double temp, double pe) const {
        const auto out = pg_from_pe_get_fe(temp, pe);
        return pe * rho_from_H / (out.second * temp);
    }

    double pg_from_rho(double temp, double rho) const {
        const double xna = rho / avw;
        double a = 0.001;
        if (temp > 8000.0) a = 0.5;
        else if (temp > 4000.0) a = 0.1;
        else if (temp > 2000.0) a = 0.01;

        const double xne = a * xna / (1.0 - a);
        double pgas = (xna + xne) * BK * temp;
        double pe = pe_from_pg(temp, pgas);
        double irho = rho_from_pe(temp, pe);

        double dif = 1.0;
        int it = 0;
        while ((dif >= PREC) && (it < 100)) {
            pe *= (1.0 + rho / irho) * 0.5;
            irho = rho_from_pe(temp, pe);
            dif = std::abs((irho - rho) / rho);
            it += 1;
        }
        return pg_from_pe(temp, pe);
    }
};

}  // namespace

extern "C" int witt_ne_from_rho(
    const char* pf_path,
    const double* temp,
    const double* rho_kg_m3,
    float* ne_m3,
    std::size_t n,
    int threads
) {
    try {
        WittEOS eos(pf_path);
        unsigned int nthreads = threads > 0
            ? static_cast<unsigned int>(threads)
            : std::thread::hardware_concurrency();
        if (nthreads == 0) nthreads = 1;
        if (n < nthreads) nthreads = static_cast<unsigned int>(std::max<std::size_t>(n, 1));

        auto worker = [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                ne_m3[i] = static_cast<float>(eos.ne_from_rho_m3(temp[i], rho_kg_m3[i]));
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        const std::size_t block = (n + nthreads - 1) / nthreads;
        for (unsigned int t = 0; t < nthreads; ++t) {
            const std::size_t begin = std::min<std::size_t>(t * block, n);
            const std::size_t end = std::min<std::size_t>(begin + block, n);
            if (begin >= end) break;
            pool.emplace_back(worker, begin, end);
        }
        for (auto& thread : pool) {
            thread.join();
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
