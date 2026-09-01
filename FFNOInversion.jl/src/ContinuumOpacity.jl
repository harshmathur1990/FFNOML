# Continuum opacity adapted from STiC src/cop.cc, itself adapted from
# N. Piskunov's Fortran/C routines (J. de la Cruz Rodriguez, 2016).
# Only the call graph reachable from cop() is ported here.

const _COP_BK=1.3806488e-16
const _COP_HH=6.62606957e-27
const _COP_EV=1.602176565e-12
const _COP_CC=2.99792458e10
_rowmatrix(v,n)=permutedims(reshape(v,n,:))

struct ContinuumState
    rho_kg_m3::Vector{Float64}
    xna_cm3::Vector{Float64}
    xne_cm3::Vector{Float64}
    partials::Matrix{Float64} # 17 × cells, ordered as WittEOS::background_partials
end

function continuum_state(eos::WittmannEOS,temp::AbstractVector,pgas::AbstractVector)
    length(temp)==length(pgas) || throw(DimensionMismatch("temperature and pressure differ"))
    t=Float64.(temp); p=Float64.(pgas); n=length(t)
    rho=zeros(n); xna=zeros(n); xne=zeros(n); partials=zeros(17,n)
    handle=Libdl.dlopen(eos.library)
    fn=Libdl.dlsym(handle,:witt_continuum_state_from_pgas)
    status=ccall(fn,Cint,(Cstring,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
        eos.partition_functions,t,p,rho,xna,xne,partials,n)
    status==0 || throw(ErrorException("Wittmann continuum-state calculation failed"))
    ContinuumState(rho,xna,xne,partials)
end

@inline _seaton(f0,xsect,power,a,f)=xsect*(a+(1-a)*(f0/f))*(f0/f)^(floor(2power+0.01)*0.5)
const _Z4LOG=(0.,1.20412,1.90849,2.40824,2.79588,3.11261)
const _COULFFA=_rowmatrix(Float64[
5.53,5.49,5.46,5.43,5.40,5.25,5.00,4.69,4.48,4.16,3.85,
4.91,4.87,4.84,4.80,4.77,4.63,4.40,4.13,3.87,3.52,3.27,
4.29,4.25,4.22,4.18,4.15,4.02,3.80,3.57,3.27,2.98,2.70,
3.64,3.61,3.59,3.56,3.54,3.41,3.22,2.97,2.70,2.45,2.20,
3.00,2.98,2.97,2.95,2.94,2.81,2.65,2.44,2.21,2.01,1.81,
2.41,2.41,2.41,2.41,2.41,2.32,2.19,2.02,1.84,1.67,1.50,
1.87,1.89,1.91,1.93,1.95,1.90,1.80,1.68,1.52,1.41,1.30,
1.33,1.39,1.44,1.49,1.55,1.56,1.51,1.42,1.33,1.25,1.17,
0.90,0.95,1.00,1.08,1.17,1.30,1.32,1.30,1.20,1.15,1.11,
0.55,0.58,0.62,0.70,0.85,1.01,1.15,1.18,1.15,1.11,1.08,
0.33,0.36,0.39,0.46,0.59,0.76,0.97,1.09,1.13,1.10,1.08,
0.19,0.21,0.24,0.28,0.38,0.53,0.76,0.96,1.08,1.09,1.09],11)

function _coulff(tlog,flog,z)
    gl=10.39638-tlog/1.15129+_Z4LOG[z]
    ig=clamp(trunc(Int,gl+7),1,10); hl=(flog-tlog)/1.15129-20.63764
    ih=clamp(trunc(Int,hl+9),1,11); p=gl-(ig-7); q=hl-(ih-9)
    (1-p)*((1-q)*_COULFFA[ih,ig]+q*_COULFFA[ih+1,ig])+p*((1-q)*_COULFFA[ih,ig+1]+q*_COULFFA[ih+1,ig+1])
end
const _CXA=(0.9916,1.105,1.101,1.101,1.102,1.0986)
const _CXB=(2.719e3,-2.375e4,-9.863e3,-5.765e3,-3.909e3,-2.704e3)
const _CXC=(-2.268e10,4.077e8,1.035e8,4.593e7,2.371e7,1.229e7)
function _coulx(N,f,z)
    n=(N+1)^2; f<z^2*3.28805e15/n && return 0.0
    f1=f*1e-10; x=0.2815/f1^3/n^2/(N+1)*z^4
    N>=6 ? x : x*(_CXA[N+1]+(_CXB[N+1]+_CXC[N+1]*(z^2/f1))*(z^2/f1))
end
function _hop(xne,h1,h2,f,flog,t,tlog,tkev,stim,eh)
    f3=(f*1e-10)^3; bolt=ntuple(i->exp(-13.595*(1-1/i^2)/tkev)*2i^2*h1,8)
    cont=ntuple(i->_coulx(i-1,f,1.0),8); freet=xne*3.6919e-22/f3*h2/sqrt(t)
    xr=h1/13.595*tkev; bex=exp(-13.427/tkev)*xr; elim=exp(-13.595/tkev)*xr
    f<4.05933e13 && (bex=elim/eh)
    out=(cont[7]*bolt[7]+cont[8]*bolt[8]+(bex-elim)*0.2815/f3+_coulff(tlog,flog,1)*freet)*stim
    out+sum(cont[i]*bolt[i] for i=1:6)*(1-eh)
end
function _h2plop(h1,h2,f,flog,f15,tkev,stim)
    f>3.28805e15 && return 0.0
    fr=-3023.3+(377.97+(-18.2496+(0.39207-0.0031672flog)*flog)*flog)*flog
    es=-0.007342+(-2.409+(1.028+(-0.4230+(0.1224-0.01351*f15)*f15)*f15)*f15)*f15
    exp(-es/tkev+fr)*2h1*h2*stim
end
function _hminop(h1,hmin,f,t,tkev,xne,eh)
    f1=f*1e-10; b=(1.3727e-15+4.3748/f)/f1; c=-2.5993e-7/f1^2
    bf=f<=1.8259e14 ? 0.0 : f>=2.111e14 ? 6.801e-10+(5.358e-3+(1.481e3+(-5.519e7+4.808e11/f1)/f1)/f1)/f1 : 3.695e-6+(-0.1251+1052/f1)/f1
    hm=t<7730 ? hmin : exp(0.7552/tkev)/(2*2.4148e15*t*sqrt(t))*h1*xne
    bf*(1-eh)*hm*1e-10+(b+c/t)*h1*xne*2e-20
end
@inline function _hrayop(h1,f)
    wave=2.997925e18/min(f,2.463e15); ww=wave^2
    (5.799e-13+1.422e-6/ww+2.784/ww^2)/ww^2*h1*2
end
const _HEG=(1.,3.,1.,9.,3.,3.,1.,9.,20.,3.)
const _HEF=(5.9452090e15,1.1528440e15,0.9803331e15,.8761076e15,.8147100e15,.4519048e15,.4030971e15,.8321191e15,.3660215e15,.3627891e15)
const _HECHI=(0.,19.819,20.615,20.964,21.217,22.718,22.920,23.006,23.073,23.086)
function _he1op(he1,he2,xne,f,flog,t,tkev,tlog,eh,stim)
    bolt=ntuple(i->exp(-_HECHI[i]/tkev)*_HEG[i]*he1,10)
    freet=xne*1e-10*he2*1e-10/sqrt(t)*1e-10
    xrlog=log(he1*(2/13.595)*tkev); bex=exp(-23.730/tkev+xrlog); elim=exp(-24.587/tkev+xrlog)
    vals=(33.32-2flog,-390.026+(21.035-0.318flog)*flog,26.83-1.91flog,61.21-2.9flog,81.35-3.5flog,12.69-1.54flog,23.85-1.86flog,49.30-2.60flog,85.20-3.69flog,58.81-2.89flog)
    nmin=findfirst(x->_HEF[x]<=f,1:10); trans=nmin===nothing ? 0.0 : sum(exp(vals[i])*bolt[i] for i=nmin:10)
    ex=f<2.055e14 ? elim/eh : bex; f3=(f*1e-10)^3
    ((ex-elim)*0.2815/f3+trans+_coulff(tlog,flog,1)*freet*3.6919e8/f3)*stim
end
function _he2op(he2,he3,xne,f,flog,t,tkev,tlog,eh,stim)
    # Preserve cop.cc's sqr(N+1) macro expansion: (N+1*N+1) == 2N+1.
    bolt=ntuple(i->begin q=2i-1; exp(-(54.403-54.403/q)/tkev)*2q*he2 end,9)
    cont=ntuple(i->_coulx(i-1,f,2.0),9); xr=he2/13.595*tkev
    bex=exp(-53.859/tkev)*xr; elim=exp(-54.403/tkev)*xr; ex=f<1.31522e14 ? elim/eh : bex
    f3=(f*1e-5)^3; out=((ex-elim)*2.815e14*4/f3+sum(cont[i]*bolt[i] for i=1:9)+_coulff(tlog,flog,2)*3.6919e-7/f3*4*xne*he3/sqrt(t))*stim
    out>=1e-20 ? out : 0.0
end
@inline _hemiop(he1,f,t,xne)=((3.397e-26+(-5.216e-11+7.039e5/f)/f)*t+(-4.116e-22+(1.067e-6+8.135e9/f)/f)+(5.081e-17+(-8.724e-3-5.659e12/f)/f)/t)*xne*he1*1e-20
function _heraop(he1,f)
    ww=(2.997925e3/min(f*1e-15,5.15))^2; a=1+(2.44e5+5.94e10/(ww-2.90e5))/ww
    5.484e-14/ww^2*a^2*he1
end

const _MGPEACH=_rowmatrix(Float64[
-42.474,-42.350,-42.109,-41.795,-41.467,-41.159,-40.883,-41.808,-41.735,-41.582,-41.363,-41.115,-40.866,-40.631,-41.273,-41.223,-41.114,-40.951,-40.755,-40.549,-40.347,-45.583,-44.008,-42.957,-42.205,-41.639,-41.198,-40.841,-44.324,-42.747,-41.694,-40.939,-40.370,-39.925,-39.566,-50.969,-48.388,-46.630,-45.344,-44.355,-43.568,-42.924,-50.633,-48.026,-46.220,-44.859,-43.803,-42.957,-42.264,-53.028,-49.643,-47.367,-45.729,-44.491,-43.520,-42.736,-51.785,-48.352,-46.050,-44.393,-43.140,-42.157,-41.363,-52.285,-48.797,-46.453,-44.765,-43.486,-42.480,-41.668,-52.028,-48.540,-46.196,-44.507,-43.227,-42.222,-41.408,-52.384,-48.876,-46.513,-44.806,-43.509,-42.488,-41.660,-52.363,-48.856,-46.493,-44.786,-43.489,-42.467,-41.639,-54.704,-50.772,-48.107,-46.176,-44.707,-43.549,-42.611,-54.359,-50.349,-47.643,-45.685,-44.198,-43.027,-42.418],7)
const _MGF=(1.9341452e15,1.8488510e15,1.1925797e15,7.9804046e14,4.5772110e14,4.1440977e14,4.1113514e14)
const _MGFL=(35.32123,35.19844,35.15334,34.71490,34.31318,33.75728,33.65788,33.64994,33.43947)
const _MGTL=(8.29405,8.51719,8.69951,8.85367,8.98720,9.10498,9.21034)
function _mg1op(f,fl,t,tl)
    nt=clamp(floor(Int,t/1000)-3,1,6); dt=(tl-_MGTL[nt])/(_MGTL[nt+1]-_MGTL[nt])
    n=something(findfirst(x->f>x,_MGF),length(_MGF)+1); d=(fl-_MGFL[n])/(_MGFL[n+1]-_MGFL[n]); n>2 && (n=2n-2)
    x1=_MGPEACH[n+1,nt]*d+_MGPEACH[n,nt]*(1-d); x2=_MGPEACH[n+1,nt+1]*d+_MGPEACH[n,nt+1]*(1-d)
    exp(x1*(1-dt)+x2*dt)
end
function _c1op(f,tkev)
    x1100=f>=2.7254e15 ? _seaton(2.7254e15,1.219e-17,2.,3.317,f) : 0.0
    x1240=f>=2.4196e15 ? _seaton(2.4196e15,1.030e-17,1.5,2.789,f) : 0.0
    x1444=f>=2.0761e15 ? _seaton(2.0761e15,9.590e-18,1.5,3.501,f) : 0.0
    9x1100+x1240*5exp(-1.264/tkev)+x1444*exp(-2.683/tkev)
end
@inline _al1op(f)=f>1.443e15 ? 2.1e-17*(1.443e15/f)^3*6 : 0.0

const _SI1PEACH=_rowmatrix(Float64[
38.136,38.138,38.140,38.141,38.143,38.144,38.144,38.145,38.145,37.834,37.839,37.843,37.847,37.850,37.853,37.855,37.857,37.858,37.898,37.898,37.897,37.897,37.897,37.896,37.895,37.895,37.894,40.737,40.319,40.047,39.855,39.714,39.604,39.517,39.445,39.385,40.581,40.164,39.893,39.702,39.561,39.452,39.366,39.295,39.235,45.521,44.456,43.753,43.254,42.878,42.580,42.332,42.119,41.930,45.520,44.455,43.752,43.251,42.871,42.569,42.315,42.094,41.896,55.068,51.783,49.553,47.942,46.723,45.768,44.997,44.360,43.823,53.868,50.369,48.031,46.355,45.092,44.104,43.308,42.652,42.100,54.133,50.597,48.233,46.539,45.261,44.262,43.456,42.790,42.230,54.051,50.514,48.150,46.454,45.176,44.175,43.368,42.702,42.141,54.442,50.854,48.455,46.733,45.433,44.415,43.592,42.912,42.340,54.320,50.722,48.313,46.583,45.277,44.251,43.423,42.738,42.160,55.691,51.965,49.444,47.615,46.221,45.119,44.223,43.478,42.848,55.661,51.933,49.412,47.582,46.188,45.085,44.189,43.445,42.813,55.973,52.193,49.630,47.769,46.349,45.226,44.314,43.555,42.913,55.922,52.141,49.577,47.715,46.295,45.172,44.259,43.500,42.858,56.828,52.821,50.110,48.146,46.654,45.477,44.522,43.730,43.061,56.657,52.653,49.944,47.983,46.491,45.315,44.360,43.569,42.901],9)
const _SI1F=(2.1413750e15,1.97231650e15,1.7879689e15,1.5152920e15,.55723927e15,5.3295914e14,4.7886458e14,4.72164220e14,4.6185133e14)
const _SI1FL=(35.45438,35.30022,35.21799,35.11986,34.95438,33.95402,33.90947,33.80244,33.78835,33.76626,33.70518)
const _SI1TL=(8.29405,8.51719,8.69951,8.85367,8.98720,9.10498,9.21034,9.30565,9.39266)
function _si1op(f,fl,t,tl)
    nt=clamp(floor(Int,t/1000)-3,1,8); dt=(tl-_SI1TL[nt])/(_SI1TL[nt+1]-_SI1TL[nt])
    n=something(findfirst(x->f>x,_SI1F),length(_SI1F)+1); d=(fl-_SI1FL[n])/(_SI1FL[n+1]-_SI1FL[n]); n>2 && (n=2n-2)
    x1=_SI1PEACH[n+1,nt]*d+_SI1PEACH[n,nt]*(1-d); x2=_SI1PEACH[n+1,nt+1]*d+_SI1PEACH[n,nt+1]*(1-d)
    9exp(-(x1*(1-dt)+x2*dt))
end
const _FEG=(25.,35.,21.,15.,9.,35.,33.,21.,27.,49.,9.,21.,27.,9.,9.,25.,33.,15.,35.,3.,5.,11.,15.,13.,15.,9.,21.,15.,21.,25.,35.,9.,5.,45.,27.,21.,15.,21.,15.,25.,21.,35.,5.,15.,45.,35.,55.,25.)
const _FEE=(500.,7500.,12500.,17500.,19000.,19500.,19500.,21000.,22000.,23000.,23000.,24000.,24000.,24500.,24500.,26000.,26500.,26500.,27000.,27500.,28500.,29000.,29500.,29500.,29500.,30000.,31500.,31500.,33500.,33500.,34000.,34500.,34500.,35000.,35500.,37000.,37000.,37000.,38500.,40000.,40000.,41000.,41000.,43000.,43000.,43000.,43000.,44000.)
const _FEW=(63500.,58500.,53500.,59500.,45000.,44500.,44500.,43000.,58000.,41000.,54000.,40000.,40000.,57500.,55500.,38000.,57500.,57500.,37000.,54500.,53500.,55000.,34500.,34500.,34500.,34000.,32500.,32500.,32500.,32500.,32000.,29500.,29500.,31000.,30500.,29000.,27000.,54000.,27500.,24000.,47000.,23000.,44000.,42000.,42000.,21000.,42000.,42000.)
function _fe1op(f,hkt)
    wn=f/_COP_CC; wn<21000 && return 0.0
    sum(_FEW[i]<wn ? 3e-18/(1+((_FEW[i]+3000-wn)/_FEW[i]/0.1)^4)*_FEG[i]*exp(-_FEE[i]*_COP_CC*hkt) : 0.0 for i=eachindex(_FEG))
end
@inline _coolop(n,stim,f,fl,t,tl,tkev,hkt)=(_c1op(f,tkev)*n[7]+_mg1op(f,fl,t,tl)*n[13]+_al1op(f)*n[8]+_si1op(f,fl,t,tl)*n[9]+_fe1op(f,hkt)*n[15])*stim

function _n1op(f,tkev)
    x853=f>=3.517915e15 ? _seaton(3.517915e15,1.142e-17,2.,4.29,f) : 0.; x1020=f>=2.941534e15 ? _seaton(2.941534e15,4.410e-18,1.5,3.85,f) : 0.; x1130=f>=2.653317e15 ? _seaton(2.653317e15,4.2e-18,1.5,4.34,f) : 0.
    4x853+x1020*10exp(-2.384/tkev)+x1130*6exp(-3.575/tkev)
end
@inline _o1op(f)=f>=3.28805e15 ? 9*_seaton(3.28805e15,2.94e-18,1.,2.66,f) : 0.
function _mg2op(f,tkev)
    x824=f>=3.635492e15 ? _seaton(3.635492e15,1.4e-19,4.,6.7,f) : 0.; x1169=f>=2.564306e15 ? 5.11e-19*(2.564306e15/f)^3 : 0.
    2x824+x1169*6exp(-4.43/tkev)
end
const _SI2PEACH=_rowmatrix(Float64[-43.8941,-43.8941,-43.8941,-43.8941,-43.8941,-43.8941,-42.2444,-42.2444,-42.2444,-42.2444,-42.2444,-42.2444,-40.6054,-40.6054,-40.6054,-40.6054,-40.6054,-40.6054,-54.2389,-52.2906,-50.8799,-49.8033,-48.9485,-48.2490,-50.4108,-48.4892,-47.1090,-46.0672,-45.2510,-44.5933,-52.0936,-50.0741,-48.5999,-47.4676,-46.5649,-45.8246,-51.9548,-49.9371,-48.4647,-47.3340,-46.4333,-45.6947,-54.2407,-51.7319,-49.9178,-48.5395,-47.4529,-46.5709,-52.7355,-50.2218,-48.4059,-47.0267,-45.9402,-45.0592,-53.5387,-50.9189,-49.0200,-47.5750,-46.4341,-45.5082,-53.2417,-50.6234,-48.7252,-47.2810,-46.1410,-45.2153,-53.5097,-50.8535,-48.9263,-47.4586,-46.2994,-45.3581,-54.0561,-51.2365,-49.1980,-47.6497,-46.4302,-45.4414,-53.8469,-51.0256,-48.9860,-47.4368,-46.2162,-45.2266],6)
const _SI2F=(4.9965417e15,3.9466738e15,1.5736321e15,1.5171539e15,9.2378947e14,8.3825004e14,7.6869872e14)
const _SI2FL=(36.32984,36.14752,35.91165,34.99216,34.95561,34.45941,34.36234,34.27572,34.20161)
const _SI2TL=(9.21034,9.39266,9.54681,9.68034,9.79813,9.90349)
function _si2op(f,fl,t,tl)
    nt=clamp(floor(Int,t/2000)-4,1,5); dt=(tl-_SI2TL[nt])/(_SI2TL[nt+1]-_SI2TL[nt])
    n=something(findfirst(x->f>x,_SI2F),length(_SI2F)+1); d=(fl-_SI2FL[n])/(_SI2FL[n+1]-_SI2FL[n]); n>2 && (n=2n-3)
    x1=_SI2PEACH[n+1,nt]*d+_SI2PEACH[n,nt]*(1-d); x2=_SI2PEACH[n+1,nt+1]*d+_SI2PEACH[n,nt+1]*(1-d)
    6exp(x1*(1-dt)+x2*dt)
end
function _ca2op(f,tkev)
    x1044=f>=2.870454e15 ? 1.08e-19*(2.870454e15/f)^3 : 0.; x1218=f>=2.460127e15 ? 1.64e-17*sqrt(2.460127e15/f) : 0.; x1420=f>=2.110779e15 ? _seaton(2.110779e15,4.13e-18,3.,.69,f) : 0.
    x1044+x1218*10exp(-1.697/tkev)+x1420*6exp(-3.142/tkev)
end
@inline _lukeop(n,stim,f,fl,t,tl,tkev)=(_n1op(f,tkev)*n[16]+_o1op(f)*n[17]+_mg2op(f,tkev)*n[14]+_si2op(f,fl,t,tl)*n[10]+_ca2op(f,tkev)*n[12])*stim
@inline _elecop(xne)=0.6653e-24*xne
function _h2raop(h1,f,t,tkev,tlog)
    ww=(2.997925e18/min(f,2.922e15))^2; sig=(8.14e-13+1.28e-6/ww+1.61/ww^2)/ww^2
    arg=4.477/tkev-46.628+(1.8031e-3+(-5.023e-7+(8.1424e-11-5.0501e-15t)*t)*t)*t-1.5tlog
    arg>-80 ? exp(arg)*(2h1)^2*sig : 0.0
end

"""STiC-compatible total continuum extinction and scattering in cm⁻¹."""
function continuum_extinction_cm(t::Real,wavelength_angstrom::Real,xna::Real,xne::Real,n::AbstractVector)
    length(n)==17 || throw(DimensionMismatch("continuum state needs 17 partial densities"))
    tk=t*_COP_BK; tkev=tk/_COP_EV; hkt=_COP_HH/tk; tl=log(t)
    f=2.997925e18/wavelength_angstrom; fl=log(f); eh=exp(-f*hkt); stim=1-eh
    absorption=_hop(xne,n[1],n[2],f,fl,t,tl,tkev,stim,eh)+_h2plop(n[1],n[2],f,fl,f*1e-15,tkev,stim)+_hminop(n[1],n[3],f,t,tkev,xne,eh)+_he1op(n[4],n[5],xne,f,fl,t,tkev,tl,eh,stim)+_he2op(n[5],n[6],xne,f,fl,t,tkev,tl,eh,stim)+_hemiop(n[4],f,t,xne)
    t<12000 && (absorption+=_coolop(n,stim,f,fl,t,tl,tkev,hkt))
    t<30000 && (absorption+=_lukeop(n,stim,f,fl,t,tl,tkev))
    scattering=_hrayop(n[1],f)+_heraop(n[4],f)+_elecop(xne)+_h2raop(n[1],f,t,tkev,tl)
    absorption+scattering,scattering
end

function continuum_extinction_m(state::ContinuumState,temp::AbstractVector,wavelength_angstrom::Real)
    length(temp)==length(state.xne_cm3) || throw(DimensionMismatch("continuum state and temperature differ"))
    [100*first(continuum_extinction_cm(temp[i],wavelength_angstrom,state.xna_cm3[i],state.xne_cm3[i],view(state.partials,:,i))) for i=eachindex(temp)]
end
