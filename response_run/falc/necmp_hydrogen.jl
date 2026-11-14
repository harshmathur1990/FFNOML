# Neutral H and protons of the FALC nₑ-comparison EOS runs, as their synthesis used them: Wittmann at (T, P) for LTE, the NLTE table
# for NLTE. Writes falc_necmp_<run>_hydrogen.h5 (h_neutral, h_proton on the atmosphere file's layout).
using FFNOInversion, HDF5, Libdl
import HSEMHS
const R="/mn/stornext/d9/data/harshm/FFNOML/response_run"
eos=WittmannEOS(joinpath(R,"inputs","wittmann","libwitt_ffno.$(Libdl.dlext)"),joinpath(R,"inputs","pf_Kurucz.input"))
states=Dict("lte"=>WittmannKuruczState(eos),
    "nlte"=>NLTEHydrogenState(HSEMHS.load_table("/mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS/cache/eos_table_nlte.h5";model="nlte")))
for m in ("lte","nlte"), a in ("m","z")
    T,P=h5open(f->(read(f,"temperature"),read(f,"pgas")),joinpath(R,"falc","falc_necmp_$(m)$(a)_atmosphere.h5"))
    t=vec(Float64.(T)); p=vec(Float64.(P)); nh=zeros(length(t))
    FFNOInversion.eos_neutral_hydrogen!(nh,states[m],t,p); np=FFNOInversion.eos_protons(states[m],t,p)
    h5open(joinpath(R,"falc","falc_necmp_$(m)$(a)_hydrogen.h5"),"w") do f
        f["h_neutral"]=reshape(nh,size(T)); f["h_proton"]=reshape(np,size(T))
    end
    println(m,a," nHI ",extrema(nh)," np ",extrema(np))
end
