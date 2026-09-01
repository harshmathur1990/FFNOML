#include <cmath>
#include <cstddef>
#include "cop.h"

extern "C" void ffno_cop_oracle(
    const double* temp, const double* xna, const double* xne,
    const double* partials, std::size_t cells, const double* wavelengths,
    std::size_t nwave, double* extinction_cm, double* scattering_cm
) {
    constexpr double bk=1.3806488e-16, hh=6.62606957e-27, ev=1.602176565e-12;
    for (std::size_t i=0; i<cells; ++i) {
        const double t=temp[i], tk=bk*t, tkev=tk/ev, hkt=hh/tk;
        const double* n=partials+17*i;
        for (std::size_t j=0; j<nwave; ++j) {
            double wavelength=wavelengths[j], opacity=0.0, scattering=0.0;
            cop(t,tkev,tk,hkt,std::log(t),xna[i],xne[i],&wavelength,
                &opacity,&scattering,n[0],n[1],n[2],n[3],n[4],n[5],n[6],
                n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15],n[16],1,0,0);
            extinction_cm[i*nwave+j]=opacity;
            scattering_cm[i*nwave+j]=scattering;
        }
    }
}

extern "C" void ffno_cop_components(double t, double xne, const double* n,
                                      double wavelength, double* out) {
    constexpr double bk=1.3806488e-16, hh=6.62606957e-27, ev=1.602176565e-12;
    const double tk=bk*t, tkev=tk/ev, hkt=hh/tk, tlog=std::log(t);
    const double f=2.997925e18/wavelength, flog=std::log(f), eh=std::exp(-f*hkt), stim=1-eh;
    HOP(out[0],xne,n[0],n[1],f,flog,t,tlog,tkev,stim,eh);
    H2PLOP(out[1],n[0],n[1],f,flog,f*1e-15,tkev,stim);
    HMINOP(out[2],n[0],n[2],f,t,tkev,xne,eh);
    HE1OP(out[3],n[3],n[4],xne,f,flog,t,tkev,tlog,eh,stim);
    HE2OP(out[4],n[4],n[5],xne,f,flog,t,tkev,tlog,eh,stim);
    HEMIOP(out[5],n[3],f,t,xne);
    out[6]=0; if(t<12000) COOLOP(out[6],n[6],n[12],n[7],n[8],n[14],stim,f,flog,t,tlog,tkev,hkt);
    out[7]=0; if(t<30000) LUKEOP(out[7],n[15],n[16],n[13],n[9],n[11],stim,f,flog,t,tlog,tkev);
    HRAYOP(out[8],n[0],f); HERAOP(out[9],n[3],f); ELECOP(out[10],xne);
    H2RAOP(out[11],n[0],f,t,tkev,tlog);
}
