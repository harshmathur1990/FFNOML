from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import MODEL, MODEL_DIR, MULTI3D_PRED_DATA, PRED_DIR


def _read_mesh_dx_dy_megameters(mesh_path):
    mesh = np.fromfile(mesh_path, sep=" ", dtype=np.float32)
    offset = 0
    nx = int(mesh[offset])
    offset += 1
    x = mesh[offset:offset + nx]
    offset += nx
    ny = int(mesh[offset])
    offset += 1
    y = mesh[offset:offset + ny]

    if x.size < 2 or y.size < 2:
        raise ValueError(f"Mesh {mesh_path!r} must contain at least two x/y points")

    # Mesh coordinates are in cm. One megametre is 1e8 cm.
    dx = abs(float(np.median(np.diff(x)))) * 1e-8
    dy = abs(float(np.median(np.diff(y)))) * 1e-8

    return dx, dy


orig_json = {
    "checkpoint_path": "training_FFNO3D_zscale_expand_new_26_05_2026/3D_sim_train_s123.pt",
    "val_h5": "IO/3D_sim_test_ch012006_849__ch012012_hion_984__en024048_hion_700__nw012023_940__qs006003_sap_900_patch40_stride20.hdf5",
    "baseline_loss": 0.029075954169546895,
    "baseline_components": {
        "data": 0.029075954168230896,
        "source": 0.07403859930396404,
        "gradient": 0.0035591060319086552,
        "source_per_atom": [
            0.07403859930396404
        ],
        "rmse": 0.0671059270374338,
        "rel_rmse": 0.07737396093802701,
        "std_ratio": 0.9853863671644134,
        "p95_err": 0.13642323666923212
    },
    "baseline_stats": {
        "layer0.spec_norm": 0.39068763292354086,
        "layer0.vertical_norm": 0.24313242251504652,
        "layer0.fuse_norm": 0.3900955155528851,
        "layer0.pw_norm": 0.8084716209380524,
        "layer0.mlp_norm": 0.591036772135622,
        "layer0.spec_weight": 1.0,
        "layer0.vertical_weight": 1.0,
        "layer0.spec_weight_effective": 1.0,
        "layer0.vertical_weight_effective": 1.0,
        "layer0.res_fused": 0.8983318209648132,
        "layer0.res_pw": 0.16526420414447784,
        "layer0.res_mlp": 0.2513953149318695,
        "layer0.fuse_contrib": 0.35043521466462507,
        "layer0.pw_contrib": 0.13361141905466223,
        "layer0.mlp_contrib": 0.14858387520039304,
        "layer0.fused_update": 0.35043521486361573,
        "layer0.pw_update": 0.1336114191039471,
        "layer0.mlp_update": 0.14858387524805824,
        "layer1.spec_norm": 0.34243395961589695,
        "layer1.vertical_norm": 0.32412762546946544,
        "layer1.fuse_norm": 0.26341195964850256,
        "layer1.pw_norm": 0.7997797686250313,
        "layer1.mlp_norm": 0.5995995081988921,
        "layer1.spec_weight": 1.0,
        "layer1.vertical_weight": 1.0,
        "layer1.spec_weight_effective": 1.0,
        "layer1.vertical_weight_effective": 1.0,
        "layer1.res_fused": 0.6356682181358337,
        "layer1.res_pw": 0.11841071397066116,
        "layer1.res_mlp": 0.21978987753391266,
        "layer1.fuse_contrib": 0.16744261126409962,
        "layer1.pw_contrib": 0.09470249342575947,
        "layer1.mlp_contrib": 0.131785902370531,
        "layer1.fused_update": 0.16744261122823503,
        "layer1.pw_update": 0.09470249339151457,
        "layer1.mlp_update": 0.1317859022763575,
        "layer2.spec_norm": 0.3721225045686183,
        "layer2.vertical_norm": 0.2127986170150201,
        "layer2.fuse_norm": 0.29974927883525815,
        "layer2.pw_norm": 0.7551998722349635,
        "layer2.mlp_norm": 0.603307693243397,
        "layer2.spec_weight": 1.0,
        "layer2.vertical_weight": 1.0,
        "layer2.spec_weight_effective": 1.0,
        "layer2.vertical_weight_effective": 1.0,
        "layer2.res_fused": 0.7997440099716187,
        "layer2.res_pw": 0.11721750348806381,
        "layer2.res_mlp": 0.21938785910606384,
        "layer2.fuse_contrib": 0.23972269026574142,
        "layer2.pw_contrib": 0.08852264369681755,
        "layer2.mlp_contrib": 0.13235838339959058,
        "layer2.fused_update": 0.2397226902939703,
        "layer2.pw_update": 0.08852264366349819,
        "layer2.mlp_update": 0.13235838340306133,
        "layer3.spec_norm": 0.31350781918543835,
        "layer3.vertical_norm": 0.35903461221813787,
        "layer3.fuse_norm": 0.3198751297241412,
        "layer3.pw_norm": 0.7068699244090489,
        "layer3.mlp_norm": 0.6012971958073771,
        "layer3.spec_weight": 1.0,
        "layer3.vertical_weight": 1.0,
        "layer3.spec_weight_effective": 1.0,
        "layer3.vertical_weight_effective": 1.0,
        "layer3.res_fused": 0.8101842999458313,
        "layer3.res_pw": 0.14528216421604156,
        "layer3.res_mlp": 0.28850048780441284,
        "layer3.fuse_contrib": 0.25915780790288995,
        "layer3.pw_contrib": 0.10269559250693329,
        "layer3.mlp_contrib": 0.17347453415949152,
        "layer3.fused_update": 0.25915780802830035,
        "layer3.pw_update": 0.10269559246620961,
        "layer3.mlp_update": 0.17347453419882689,
        "mean.spec_norm": 0.3546879790733736,
        "mean.vertical_norm": 0.28477331930441746,
        "mean.fuse_norm": 0.31828297094019675,
        "mean.pw_norm": 0.7675802965517741,
        "mean.mlp_norm": 0.598810292346322,
        "mean.spec_weight": 1.0,
        "mean.vertical_weight": 1.0,
        "mean.spec_weight_effective": 1.0,
        "mean.vertical_weight_effective": 1.0,
        "mean.res_fused": 0.7859820872545242,
        "mean.res_pw": 0.1365436464548111,
        "mean.res_mlp": 0.2447683848440647,
        "mean.fuse_contrib": 0.25418958102433903,
        "mean.pw_contrib": 0.10488303717104314,
        "mean.mlp_contrib": 0.14655067378250153,
        "mean.fused_update": 0.25418958110353035,
        "mean.pw_update": 0.10488303715629237,
        "mean.mlp_update": 0.146550673781576
    },
    "validation_metadata": {
        "strategy": "distributed_full_h_slab",
        "distributed_inference_enabled": True,
        "distributed_initialized": True,
        "fsdp_enabled": False,
        "world_size": 6,
        "num_items": 805,
        "dataset_items": 805,
        "visited_all_dataset_items_once": True,
        "num_columns": 1288000,
        "local_columns": 225400,
        "rank": 0
    },
    "branch_importance": {
        "no_spec": 2.960750749155182,
        "no_vertical": 0.16061384758251135,
        "no_pw": 0.07821637122453659,
        "no_mlp": 0.18686849435298236
    },
    "ablations": {
        "full": {
            "branch_mask": {
                "spec": 1.0,
                "vertical": 1.0,
                "pw": 1.0,
                "mlp": 1.0
            },
            "loss": 0.029075954169546895,
            "loss_delta_vs_full": 0.0,
            "components": {
                "data": 0.029075954168230896,
                "source": 0.07403859930396404,
                "gradient": 0.0035591060319086552,
                "source_per_atom": [
                    0.07403859930396404
                ],
                "rmse": 0.0671059270374338,
                "rel_rmse": 0.07737396093802701,
                "std_ratio": 0.9853863671644134,
                "p95_err": 0.13642323666923212
            },
            "stats": {
                "layer0.spec_norm": 0.39068763292354086,
                "layer0.vertical_norm": 0.24313242251504652,
                "layer0.fuse_norm": 0.3900955155528851,
                "layer0.pw_norm": 0.8084716209380524,
                "layer0.mlp_norm": 0.591036772135622,
                "layer0.spec_weight": 1.0,
                "layer0.vertical_weight": 1.0,
                "layer0.spec_weight_effective": 1.0,
                "layer0.vertical_weight_effective": 1.0,
                "layer0.res_fused": 0.8983318209648132,
                "layer0.res_pw": 0.16526420414447784,
                "layer0.res_mlp": 0.2513953149318695,
                "layer0.fuse_contrib": 0.35043521466462507,
                "layer0.pw_contrib": 0.13361141905466223,
                "layer0.mlp_contrib": 0.14858387520039304,
                "layer0.fused_update": 0.35043521486361573,
                "layer0.pw_update": 0.1336114191039471,
                "layer0.mlp_update": 0.14858387524805824,
                "layer1.spec_norm": 0.34243395961589695,
                "layer1.vertical_norm": 0.32412762546946544,
                "layer1.fuse_norm": 0.26341195964850256,
                "layer1.pw_norm": 0.7997797686250313,
                "layer1.mlp_norm": 0.5995995081988921,
                "layer1.spec_weight": 1.0,
                "layer1.vertical_weight": 1.0,
                "layer1.spec_weight_effective": 1.0,
                "layer1.vertical_weight_effective": 1.0,
                "layer1.res_fused": 0.6356682181358337,
                "layer1.res_pw": 0.11841071397066116,
                "layer1.res_mlp": 0.21978987753391266,
                "layer1.fuse_contrib": 0.16744261126409962,
                "layer1.pw_contrib": 0.09470249342575947,
                "layer1.mlp_contrib": 0.131785902370531,
                "layer1.fused_update": 0.16744261122823503,
                "layer1.pw_update": 0.09470249339151457,
                "layer1.mlp_update": 0.1317859022763575,
                "layer2.spec_norm": 0.3721225045686183,
                "layer2.vertical_norm": 0.2127986170150201,
                "layer2.fuse_norm": 0.29974927883525815,
                "layer2.pw_norm": 0.7551998722349635,
                "layer2.mlp_norm": 0.603307693243397,
                "layer2.spec_weight": 1.0,
                "layer2.vertical_weight": 1.0,
                "layer2.spec_weight_effective": 1.0,
                "layer2.vertical_weight_effective": 1.0,
                "layer2.res_fused": 0.7997440099716187,
                "layer2.res_pw": 0.11721750348806381,
                "layer2.res_mlp": 0.21938785910606384,
                "layer2.fuse_contrib": 0.23972269026574142,
                "layer2.pw_contrib": 0.08852264369681755,
                "layer2.mlp_contrib": 0.13235838339959058,
                "layer2.fused_update": 0.2397226902939703,
                "layer2.pw_update": 0.08852264366349819,
                "layer2.mlp_update": 0.13235838340306133,
                "layer3.spec_norm": 0.31350781918543835,
                "layer3.vertical_norm": 0.35903461221813787,
                "layer3.fuse_norm": 0.3198751297241412,
                "layer3.pw_norm": 0.7068699244090489,
                "layer3.mlp_norm": 0.6012971958073771,
                "layer3.spec_weight": 1.0,
                "layer3.vertical_weight": 1.0,
                "layer3.spec_weight_effective": 1.0,
                "layer3.vertical_weight_effective": 1.0,
                "layer3.res_fused": 0.8101842999458313,
                "layer3.res_pw": 0.14528216421604156,
                "layer3.res_mlp": 0.28850048780441284,
                "layer3.fuse_contrib": 0.25915780790288995,
                "layer3.pw_contrib": 0.10269559250693329,
                "layer3.mlp_contrib": 0.17347453415949152,
                "layer3.fused_update": 0.25915780802830035,
                "layer3.pw_update": 0.10269559246620961,
                "layer3.mlp_update": 0.17347453419882689,
                "mean.spec_norm": 0.3546879790733736,
                "mean.vertical_norm": 0.28477331930441746,
                "mean.fuse_norm": 0.31828297094019675,
                "mean.pw_norm": 0.7675802965517741,
                "mean.mlp_norm": 0.598810292346322,
                "mean.spec_weight": 1.0,
                "mean.vertical_weight": 1.0,
                "mean.spec_weight_effective": 1.0,
                "mean.vertical_weight_effective": 1.0,
                "mean.res_fused": 0.7859820872545242,
                "mean.res_pw": 0.1365436464548111,
                "mean.res_mlp": 0.2447683848440647,
                "mean.fuse_contrib": 0.25418958102433903,
                "mean.pw_contrib": 0.10488303717104314,
                "mean.mlp_contrib": 0.14655067378250153,
                "mean.fused_update": 0.25418958110353035,
                "mean.pw_update": 0.10488303715629237,
                "mean.mlp_update": 0.146550673781576
            }
        },
        "no_spec": {
            "branch_mask": {
                "spec": 0.0,
                "vertical": 1.0,
                "pw": 1.0,
                "mlp": 1.0
            },
            "loss": 2.989826703324729,
            "loss_delta_vs_full": 2.960750749155182,
            "components": {
                "data": 2.9898267024283456,
                "source": 5.065049237508951,
                "gradient": 0.3095755340473744,
                "source_per_atom": [
                    5.065049237508951
                ],
                "rmse": 0.8198651525276418,
                "rel_rmse": 0.8599328060392637,
                "std_ratio": 0.43524466154738245,
                "p95_err": 1.6036711202608132
            },
            "stats": {
                "layer0.spec_norm": 0.39068763292354086,
                "layer0.vertical_norm": 0.24313242251504652,
                "layer0.fuse_norm": 0.34142940508282704,
                "layer0.pw_norm": 0.7880349329761837,
                "layer0.mlp_norm": 0.58770950816618,
                "layer0.spec_weight": 0.0,
                "layer0.vertical_weight": 1.0,
                "layer0.spec_weight_effective": 0.0,
                "layer0.vertical_weight_effective": 1.0,
                "layer0.res_fused": 0.8983318209648132,
                "layer0.res_pw": 0.16526420414447784,
                "layer0.res_mlp": 0.2513953149318695,
                "layer0.fuse_contrib": 0.30671689959416476,
                "layer0.pw_contrib": 0.13023396589116465,
                "layer0.mlp_contrib": 0.1477474171159245,
                "layer0.fused_update": 0.3067168996154522,
                "layer0.pw_update": 0.1302339658261456,
                "layer0.mlp_update": 0.1477474170747381,
                "layer1.spec_norm": 0.3402207839817549,
                "layer1.vertical_norm": 0.35526400159882465,
                "layer1.fuse_norm": 0.25766945188691526,
                "layer1.pw_norm": 0.8009343618802403,
                "layer1.mlp_norm": 0.5781383952442903,
                "layer1.spec_weight": 0.0,
                "layer1.vertical_weight": 1.0,
                "layer1.spec_weight_effective": 0.0,
                "layer1.vertical_weight_effective": 1.0,
                "layer1.res_fused": 0.6356682181358337,
                "layer1.res_pw": 0.11841071397066116,
                "layer1.res_mlp": 0.21978987753391266,
                "layer1.fuse_contrib": 0.1637922814157531,
                "layer1.pw_contrib": 0.09483920971498541,
                "layer1.mlp_contrib": 0.12706896707531273,
                "layer1.fused_update": 0.1637922813967796,
                "layer1.pw_update": 0.09483920964742114,
                "layer1.mlp_update": 0.1270689670741558,
                "layer2.spec_norm": 0.373925635064981,
                "layer2.vertical_norm": 0.23480085093967662,
                "layer2.fuse_norm": 0.3076534202033133,
                "layer2.pw_norm": 0.7710854414142437,
                "layer2.mlp_norm": 0.5959838085940906,
                "layer2.spec_weight": 0.0,
                "layer2.vertical_weight": 1.0,
                "layer2.spec_weight_effective": 0.0,
                "layer2.vertical_weight_effective": 1.0,
                "layer2.res_fused": 0.7997440099716187,
                "layer2.res_pw": 0.11721750348806381,
                "layer2.res_mlp": 0.21938785910606384,
                "layer2.fuse_contrib": 0.2460439797523229,
                "layer2.pw_contrib": 0.09038471032813285,
                "layer2.mlp_contrib": 0.13075161204690705,
                "layer2.fused_update": 0.2460439797911955,
                "layer2.pw_update": 0.09038471028046764,
                "layer2.mlp_update": 0.13075161207212796,
                "layer3.spec_norm": 0.3451714581001249,
                "layer3.vertical_norm": 0.38512787226842055,
                "layer3.fuse_norm": 0.355550217884853,
                "layer3.pw_norm": 0.7624173040475164,
                "layer3.mlp_norm": 0.6109065947565974,
                "layer3.spec_weight": 0.0,
                "layer3.vertical_weight": 1.0,
                "layer3.spec_weight_effective": 0.0,
                "layer3.vertical_weight_effective": 1.0,
                "layer3.res_fused": 0.8101842999458313,
                "layer3.res_pw": 0.14528216421604156,
                "layer3.res_mlp": 0.28850048780441284,
                "layer3.fuse_contrib": 0.2880612042620315,
                "layer3.pw_contrib": 0.11076563578075874,
                "layer3.mlp_contrib": 0.17624685058358663,
                "layer3.fused_update": 0.288061204028796,
                "layer3.pw_update": 0.1107656357754369,
                "layer3.mlp_update": 0.1762468507182524,
                "mean.spec_norm": 0.3625013775176004,
                "mean.vertical_norm": 0.30458128683049207,
                "mean.fuse_norm": 0.3155756237644772,
                "mean.pw_norm": 0.780618010079546,
                "mean.mlp_norm": 0.5931845766902896,
                "mean.spec_weight": 0.0,
                "mean.vertical_weight": 1.0,
                "mean.spec_weight_effective": 0.0,
                "mean.vertical_weight_effective": 1.0,
                "mean.res_fused": 0.7859820872545242,
                "mean.res_pw": 0.1365436464548111,
                "mean.res_mlp": 0.2447683848440647,
                "mean.fuse_contrib": 0.25115359125606807,
                "mean.pw_contrib": 0.10655588042876041,
                "mean.mlp_contrib": 0.14545371170543273,
                "mean.fused_update": 0.2511535912080558,
                "mean.pw_update": 0.10655588038236782,
                "mean.mlp_update": 0.14545371173481858
            }
        },
        "no_vertical": {
            "branch_mask": {
                "spec": 1.0,
                "vertical": 0.0,
                "pw": 1.0,
                "mlp": 1.0
            },
            "loss": 0.18968980175205824,
            "loss_delta_vs_full": 0.16061384758251135,
            "components": {
                "data": 0.18968980172487057,
                "source": 1.3697254286266818,
                "gradient": 0.02843427079341082,
                "source_per_atom": [
                    1.3697254286266818
                ],
                "rmse": 0.3252627964373331,
                "rel_rmse": 0.6495630949221968,
                "std_ratio": 1.3589549391850921,
                "p95_err": 0.5831994385512904
            },
            "stats": {
                "layer0.spec_norm": 0.39068763292354086,
                "layer0.vertical_norm": 0.24313242251504652,
                "layer0.fuse_norm": 0.39786298345242227,
                "layer0.pw_norm": 0.8094285687952308,
                "layer0.mlp_norm": 0.5806568448884146,
                "layer0.spec_weight": 1.0,
                "layer0.vertical_weight": 0.0,
                "layer0.spec_weight_effective": 1.0,
                "layer0.vertical_weight_effective": 0.0,
                "layer0.res_fused": 0.8983318209648132,
                "layer0.res_pw": 0.16526420414447784,
                "layer0.res_mlp": 0.2513953149318695,
                "layer0.fuse_contrib": 0.3574129781497191,
                "layer0.pw_contrib": 0.13376956824676037,
                "layer0.mlp_contrib": 0.14597441033037922,
                "layer0.fused_update": 0.35741297800348415,
                "layer0.pw_update": 0.13376956819493022,
                "layer0.mlp_update": 0.1459744103891509,
                "layer1.spec_norm": 0.33814310064782266,
                "layer1.vertical_norm": 0.3258282010581182,
                "layer1.fuse_norm": 0.3385822400381291,
                "layer1.pw_norm": 0.787097340070683,
                "layer1.mlp_norm": 0.6030347107008377,
                "layer1.spec_weight": 1.0,
                "layer1.vertical_weight": 0.0,
                "layer1.spec_weight_effective": 1.0,
                "layer1.vertical_weight_effective": 0.0,
                "layer1.res_fused": 0.6356682181358337,
                "layer1.res_pw": 0.11841071397066116,
                "layer1.res_mlp": 0.21978987753391266,
                "layer1.fuse_contrib": 0.2152259690916131,
                "layer1.pw_contrib": 0.09320075811135659,
                "layer1.mlp_contrib": 0.13254092528259162,
                "layer1.fused_update": 0.2152259689814741,
                "layer1.pw_update": 0.09320075805189076,
                "layer1.mlp_update": 0.13254092530040823,
                "layer2.spec_norm": 0.3869951247261918,
                "layer2.vertical_norm": 0.23731930247328667,
                "layer2.fuse_norm": 0.31106984142190924,
                "layer2.pw_norm": 0.7773713948267588,
                "layer2.mlp_norm": 0.5613230308520127,
                "layer2.spec_weight": 1.0,
                "layer2.vertical_weight": 0.0,
                "layer2.spec_weight_effective": 1.0,
                "layer2.vertical_weight_effective": 0.0,
                "layer2.res_fused": 0.7997440099716187,
                "layer2.res_pw": 0.11721750348806381,
                "layer2.res_mlp": 0.21938785910606384,
                "layer2.fuse_contrib": 0.24877624259231995,
                "layer2.pw_contrib": 0.09112153416163989,
                "layer2.mlp_contrib": 0.12314745799331747,
                "layer2.fused_update": 0.2487762426316553,
                "layer2.pw_update": 0.09112153416487927,
                "layer2.mlp_update": 0.12314745799563132,
                "layer3.spec_norm": 0.27589083164467576,
                "layer3.vertical_norm": 0.3480960677240206,
                "layer3.fuse_norm": 0.2762611812848297,
                "layer3.pw_norm": 0.7109014287313319,
                "layer3.mlp_norm": 0.5971184254303482,
                "layer3.spec_weight": 1.0,
                "layer3.vertical_weight": 0.0,
                "layer3.spec_weight_effective": 1.0,
                "layer3.vertical_weight_effective": 0.0,
                "layer3.res_fused": 0.8101842999458313,
                "layer3.res_pw": 0.14528216421604156,
                "layer3.res_mlp": 0.28850048780441284,
                "layer3.fuse_contrib": 0.22382247167344418,
                "layer3.pw_contrib": 0.10328129810113344,
                "layer3.mlp_contrib": 0.17226895713241575,
                "layer3.fused_update": 0.22382247149805476,
                "layer3.pw_update": 0.1032812980733673,
                "layer3.mlp_update": 0.1722689570884527,
                "mean.spec_norm": 0.34792917248555777,
                "mean.vertical_norm": 0.288593998442618,
                "mean.fuse_norm": 0.33094406154932254,
                "mean.pw_norm": 0.7711996831060011,
                "mean.mlp_norm": 0.5855332529679034,
                "mean.spec_weight": 1.0,
                "mean.vertical_weight": 0.0,
                "mean.spec_weight_effective": 1.0,
                "mean.vertical_weight_effective": 0.0,
                "mean.res_fused": 0.7859820872545242,
                "mean.res_pw": 0.1365436464548111,
                "mean.res_mlp": 0.2447683848440647,
                "mean.fuse_contrib": 0.2613094153767741,
                "mean.pw_contrib": 0.10534328965522258,
                "mean.mlp_contrib": 0.14348293768467602,
                "mean.fused_update": 0.2613094152786671,
                "mean.pw_update": 0.10534328962126689,
                "mean.mlp_update": 0.14348293769341078
            }
        },
        "no_pw": {
            "branch_mask": {
                "spec": 1.0,
                "vertical": 1.0,
                "pw": 0.0,
                "mlp": 1.0
            },
            "loss": 0.10729232539408348,
            "loss_delta_vs_full": 0.07821637122453659,
            "components": {
                "data": 0.107292325448777,
                "source": 0.42571069268648576,
                "gradient": 0.01357766495876218,
                "source_per_atom": [
                    0.42571069268648576
                ],
                "rmse": 0.14906035323054617,
                "rel_rmse": 0.17146774045729674,
                "std_ratio": 0.9123259079437819,
                "p95_err": 0.31831789619121587
            },
            "stats": {
                "layer0.spec_norm": 0.39068763292354086,
                "layer0.vertical_norm": 0.24313242251504652,
                "layer0.fuse_norm": 0.3900955155528851,
                "layer0.pw_norm": 0.8084716209380524,
                "layer0.mlp_norm": 0.5920235847020001,
                "layer0.spec_weight": 1.0,
                "layer0.vertical_weight": 1.0,
                "layer0.spec_weight_effective": 1.0,
                "layer0.vertical_weight_effective": 1.0,
                "layer0.res_fused": 0.8983318209648132,
                "layer0.res_pw": 0.16526420414447784,
                "layer0.res_mlp": 0.2513953149318695,
                "layer0.fuse_contrib": 0.35043521466462507,
                "layer0.pw_contrib": 0.0,
                "layer0.mlp_contrib": 0.1488319555730183,
                "layer0.fused_update": 0.35043521486361573,
                "layer0.pw_update": 0.0,
                "layer0.mlp_update": 0.14883195559754506,
                "layer1.spec_norm": 0.3444001824431908,
                "layer1.vertical_norm": 0.3219376716022351,
                "layer1.fuse_norm": 0.2604727828766433,
                "layer1.pw_norm": 0.7940056698274168,
                "layer1.mlp_norm": 0.5942089964458661,
                "layer1.spec_weight": 1.0,
                "layer1.vertical_weight": 1.0,
                "layer1.spec_weight_effective": 1.0,
                "layer1.vertical_weight_effective": 1.0,
                "layer1.res_fused": 0.6356682181358337,
                "layer1.res_pw": 0.11841071397066116,
                "layer1.res_mlp": 0.21978987753391266,
                "layer1.fuse_contrib": 0.16557426983224494,
                "layer1.pw_contrib": 0.0,
                "layer1.mlp_contrib": 0.130601122629661,
                "layer1.fused_update": 0.16557426982900555,
                "layer1.pw_update": 0.0,
                "layer1.mlp_update": 0.1306011226532622,
                "layer2.spec_norm": 0.3779192139199061,
                "layer2.vertical_norm": 0.21540113677426895,
                "layer2.fuse_norm": 0.3043846393788453,
                "layer2.pw_norm": 0.756734797873112,
                "layer2.mlp_norm": 0.6063226643352775,
                "layer2.spec_weight": 1.0,
                "layer2.vertical_weight": 1.0,
                "layer2.spec_weight_effective": 1.0,
                "layer2.vertical_weight_effective": 1.0,
                "layer2.res_fused": 0.7997440099716187,
                "layer2.res_pw": 0.11721750348806381,
                "layer2.res_mlp": 0.21938785910606384,
                "layer2.fuse_contrib": 0.2434297919893487,
                "layer2.pw_contrib": 0.0,
                "layer2.mlp_contrib": 0.13301983142283208,
                "layer2.fused_update": 0.24342979193335365,
                "layer2.pw_update": 0.0,
                "layer2.mlp_update": 0.13301983139645424,
                "layer3.spec_norm": 0.33325527362988233,
                "layer3.vertical_norm": 0.3566490599304808,
                "layer3.fuse_norm": 0.3324590774676444,
                "layer3.pw_norm": 0.7166164155239644,
                "layer3.mlp_norm": 0.6119186772674508,
                "layer3.spec_weight": 1.0,
                "layer3.vertical_weight": 1.0,
                "layer3.spec_weight_effective": 1.0,
                "layer3.vertical_weight_effective": 1.0,
                "layer3.res_fused": 0.8101842999458313,
                "layer3.res_pw": 0.14528216421604156,
                "layer3.res_mlp": 0.28850048780441284,
                "layer3.fuse_contrib": 0.2693531250736173,
                "layer3.pw_contrib": 0.0,
                "layer3.mlp_contrib": 0.17653883697037,
                "layer3.fused_update": 0.26935312514766035,
                "layer3.pw_update": 0.0,
                "layer3.mlp_update": 0.1765388370601472,
                "mean.spec_norm": 0.36156557572913006,
                "mean.vertical_norm": 0.2842800727055078,
                "mean.fuse_norm": 0.3218530038190045,
                "mean.pw_norm": 0.7689571260406364,
                "mean.mlp_norm": 0.6011184806876486,
                "mean.spec_weight": 1.0,
                "mean.vertical_weight": 1.0,
                "mean.spec_weight_effective": 1.0,
                "mean.vertical_weight_effective": 1.0,
                "mean.res_fused": 0.7859820872545242,
                "mean.res_pw": 0.1365436464548111,
                "mean.res_mlp": 0.2447683848440647,
                "mean.fuse_contrib": 0.257198100389959,
                "mean.pw_contrib": 0.0,
                "mean.mlp_contrib": 0.14724793664897035,
                "mean.fused_update": 0.25719810044340885,
                "mean.pw_update": 0.0,
                "mean.mlp_update": 0.14724793667685218
            }
        },
        "no_mlp": {
            "branch_mask": {
                "spec": 1.0,
                "vertical": 1.0,
                "pw": 1.0,
                "mlp": 0.0
            },
            "loss": 0.21594444852252925,
            "loss_delta_vs_full": 0.18686849435298236,
            "components": {
                "data": 0.21594444851107572,
                "source": 1.3259028011865868,
                "gradient": 0.023764434749628305,
                "source_per_atom": [
                    1.3259028011865868
                ],
                "rmse": 0.2260568565410209,
                "rel_rmse": 0.26401370478805547,
                "std_ratio": 0.8110873066546014,
                "p95_err": 0.4591106219909021
            },
            "stats": {
                "layer0.spec_norm": 0.39068763292354086,
                "layer0.vertical_norm": 0.24313242251504652,
                "layer0.fuse_norm": 0.3900955155528851,
                "layer0.pw_norm": 0.8084716209380524,
                "layer0.mlp_norm": 0.591036772135622,
                "layer0.spec_weight": 1.0,
                "layer0.vertical_weight": 1.0,
                "layer0.spec_weight_effective": 1.0,
                "layer0.vertical_weight_effective": 1.0,
                "layer0.res_fused": 0.8983318209648132,
                "layer0.res_pw": 0.16526420414447784,
                "layer0.res_mlp": 0.2513953149318695,
                "layer0.fuse_contrib": 0.35043521466462507,
                "layer0.pw_contrib": 0.13361141905466223,
                "layer0.mlp_contrib": 0.0,
                "layer0.fused_update": 0.35043521486361573,
                "layer0.pw_update": 0.1336114191039471,
                "layer0.mlp_update": 0.0,
                "layer1.spec_norm": 0.34091448326381096,
                "layer1.vertical_norm": 0.322731938421356,
                "layer1.fuse_norm": 0.26360097440757924,
                "layer1.pw_norm": 0.7961351759352299,
                "layer1.mlp_norm": 0.59379646902499,
                "layer1.spec_weight": 1.0,
                "layer1.vertical_weight": 1.0,
                "layer1.spec_weight_effective": 1.0,
                "layer1.vertical_weight_effective": 1.0,
                "layer1.res_fused": 0.6356682181358337,
                "layer1.res_pw": 0.11841071397066116,
                "layer1.res_mlp": 0.21978987753391266,
                "layer1.fuse_contrib": 0.16756276173158463,
                "layer1.pw_contrib": 0.09427093467079334,
                "layer1.mlp_contrib": 0.0,
                "layer1.fused_update": 0.167562761746856,
                "layer1.pw_update": 0.09427093460762538,
                "layer1.mlp_update": 0.0,
                "layer2.spec_norm": 0.3659915588176028,
                "layer2.vertical_norm": 0.22316315274500514,
                "layer2.fuse_norm": 0.3019277237966564,
                "layer2.pw_norm": 0.7564075772510552,
                "layer2.mlp_norm": 0.5733396648981186,
                "layer2.spec_weight": 1.0,
                "layer2.vertical_weight": 1.0,
                "layer2.spec_weight_effective": 1.0,
                "layer2.vertical_weight_effective": 1.0,
                "layer2.res_fused": 0.7997440099716187,
                "layer2.res_pw": 0.11721750348806381,
                "layer2.res_mlp": 0.21938785910606384,
                "layer2.fuse_contrib": 0.2414648884696805,
                "layer2.pw_contrib": 0.08866420778775623,
                "layer2.mlp_contrib": 0.0,
                "layer2.fused_update": 0.24146488854603737,
                "layer2.pw_update": 0.08866420785647741,
                "layer2.mlp_update": 0.0,
                "layer3.spec_norm": 0.2850163442266654,
                "layer3.vertical_norm": 0.3547032907591288,
                "layer3.fuse_norm": 0.32559723477278435,
                "layer3.pw_norm": 0.7105706846121675,
                "layer3.mlp_norm": 0.5974886277392044,
                "layer3.spec_weight": 1.0,
                "layer3.vertical_weight": 1.0,
                "layer3.spec_weight_effective": 1.0,
                "layer3.vertical_weight_effective": 1.0,
                "layer3.res_fused": 0.8101842999458313,
                "layer3.res_pw": 0.14528216421604156,
                "layer3.res_mlp": 0.28850048780441284,
                "layer3.fuse_contrib": 0.26379376748194977,
                "layer3.pw_contrib": 0.10323324671188681,
                "layer3.mlp_contrib": 0.0,
                "layer3.fused_update": 0.2637937676175411,
                "layer3.pw_update": 0.10323324675307326,
                "layer3.mlp_update": 0.0,
                "mean.spec_norm": 0.345652504807905,
                "mean.vertical_norm": 0.28593270111013414,
                "mean.fuse_norm": 0.3203053621324763,
                "mean.pw_norm": 0.7678962646841263,
                "mean.mlp_norm": 0.5889153834494838,
                "mean.spec_weight": 1.0,
                "mean.vertical_weight": 1.0,
                "mean.spec_weight_effective": 1.0,
                "mean.vertical_weight_effective": 1.0,
                "mean.res_fused": 0.7859820872545242,
                "mean.res_pw": 0.1365436464548111,
                "mean.res_mlp": 0.2447683848440647,
                "mean.fuse_contrib": 0.25581415808696,
                "mean.pw_contrib": 0.10494495205627466,
                "mean.mlp_contrib": 0.0,
                "mean.fused_update": 0.25581415819351255,
                "mean.pw_update": 0.10494495208028079,
                "mean.mlp_update": 0.0
            }
        }
    }
}

def make_branch_importance_plots():

    # ==========================
    # Branch importance
    # ==========================

    branches = ["Spec", "Vertical", "MLP", "PW"]
    delta_loss = [
        orig_json["branch_importance"]["no_spec"],
        orig_json["branch_importance"]["no_vertical"],
        orig_json["branch_importance"]["no_mlp"],
        orig_json["branch_importance"]["no_pw"]
    ]

    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    bars = ax.bar(branches, delta_loss)

    ax.set_ylabel("Δ Validation Loss")
    ax.set_title("Branch Importance from masking")

    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            0.9 * h,
            f"{h:.3f}",
            ha="center",
            va="bottom"
        )

    plt.subplots_adjust(left=0.15, right=0.99, top=0.85, bottom=0.15)
    plt.savefig('figures/branch_importance.pdf', dpi=300, format='pdf')


def get_data_for_line_core_intensity_plots():
    valid_names = [
        "en024048_hion_385",
        "nw012023_1050",
        "ch024031_by200bz005_450",
        "en024031_by100_helium_109"
    ]

    fnoml_dir = Path(PRED_DIR) / "FFNOML"
    ml_intensity_dir = fnoml_dir / MODEL_DIR
    bifrost_intensity_dir = fnoml_dir / "IO"
    datasets_by_name = {dataset["NAME"]: dataset for dataset in MULTI3D_PRED_DATA}
    plot_data = []

    for name in valid_names:
        if name not in datasets_by_name:
            raise KeyError(f"No configured prediction dataset named {name!r}")

        ml_path = ml_intensity_dir / f"intensity_ml_{name}_{MODEL}.h5"
        bifrost_path = bifrost_intensity_dir / f"intensity_bifrost_{name}.h5"

        with h5py.File(ml_path, "r") as ml_file:
            ml_intensity = np.asarray(ml_file["H"]["intensity"][:, :, 51])

        with h5py.File(bifrost_path, "r") as bifrost_file:
            bifrost_intensity = np.asarray(
                bifrost_file["H"]["intensity"][:, :, 51]
            )

        dx, dy = _read_mesh_dx_dy_megameters(datasets_by_name[name]["MESH"])

        if ml_intensity.shape != bifrost_intensity.shape:
            raise ValueError(
                f"Intensity shape mismatch for {name!r}: "
                f"ML {ml_intensity.shape}, Bifrost {bifrost_intensity.shape}"
            )

        plot_data.append({
            "name": name,
            "ml": ml_intensity,
            "bifrost": bifrost_intensity,
            "dx": dx,
            "dy": dy,
        })

    return plot_data


def make_line_core_intensity_compare_plots():
    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(7, 3.5), constrained_layout=True)
    plot_data = get_data_for_line_core_intensity_plots()
    image = None

    for index, data in enumerate(plot_data):
        row = index // 2
        first_column = 2 * (index % 2)
        panel_label = f"{chr(ord('a') + index)})"

        for column, source, title in (
            (first_column, "ml", "ML"),
            (first_column + 1, "bifrost", "Bifrost"),
        ):
            intensity = data[source]
            extent = (
                0.0,
                intensity.shape[0] * data["dx"],
                0.0,
                intensity.shape[1] * data["dy"],
            )
            ax = axs[row, column]
            image = ax.imshow(
                intensity,
                origin="lower",
                extent=extent,
                cmap="gray",
                vmin=0.0,
                vmax=12.0,
                aspect="equal",
            )
            ax.set_title(title)
            ax.set_xlabel("x [Mm]")
            if column in (0, 2):
                ax.set_ylabel("y [Mm]")
            if source == "ml":
                ax.text(
                    0.03,
                    0.97,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="white",
                    fontweight="bold",
                )

    if image is not None:
        fig.colorbar(image, ax=axs, label="Line-core intensity", shrink=0.8)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "line_core_intensity_comparison.pdf",
        dpi=300,
        format="pdf",
    )


def get_data_for_zero_shot_super_resolution_plots():
    sel_names = [
        "en024048_hion_385",
        "nw012023_1050",
        "ch024031_by200bz005_450",
        "en024031_by100_helium_109"
    ]

    sel_names_super_resolution = [
        "en024048_hion_504_385",
        "nw012023_512_1050",
        "ch024031_by200bz005_768_450",
        "en024031_by100_helium_768_109"
    ]

    fnoml_dir = Path(PRED_DIR) / "FFNOML"
    ml_intensity_dir = fnoml_dir / MODEL_DIR
    datasets_by_name = {dataset["NAME"]: dataset for dataset in MULTI3D_PRED_DATA}
    plot_data = []

    for lowres_name, highres_name in zip(sel_names, sel_names_super_resolution):
        if lowres_name not in datasets_by_name:
            raise KeyError(f"No configured prediction dataset named {lowres_name!r}")
        if highres_name not in datasets_by_name:
            raise KeyError(f"No configured prediction dataset named {highres_name!r}")

        lowres_path = ml_intensity_dir / f"intensity_ml_{lowres_name}_{MODEL}.h5"
        highres_path = ml_intensity_dir / f"intensity_ml_{highres_name}_{MODEL}.h5"

        with h5py.File(lowres_path, "r") as lowres_file:
            lowres_intensity = np.asarray(
                lowres_file["H"]["intensity"][:, :, 51]
            )

        with h5py.File(highres_path, "r") as highres_file:
            highres_intensity = np.asarray(
                highres_file["H"]["intensity"][:, :, 51]
            )

        lowres_dx, lowres_dy = _read_mesh_dx_dy_megameters(
            datasets_by_name[lowres_name]["MESH"]
        )
        highres_dx, highres_dy = _read_mesh_dx_dy_megameters(
            datasets_by_name[highres_name]["MESH"]
        )

        plot_data.append({
            "name": lowres_name,
            "super_resolution_name": highres_name,
            "lowres": lowres_intensity,
            "highres": highres_intensity,
            "lowres_dx": lowres_dx,
            "lowres_dy": lowres_dy,
            "highres_dx": highres_dx,
            "highres_dy": highres_dy,
        })

    return plot_data


def make_zero_shot_super_resolution_plots():
    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(7, 3.5), constrained_layout=True)
    plot_data = get_data_for_zero_shot_super_resolution_plots()
    image = None

    for index, data in enumerate(plot_data):
        row = index // 2
        first_column = 2 * (index % 2)
        panel_label = f"{chr(ord('a') + index)})"

        for column, source, title in (
            (first_column, "lowres", "coarse"),
            (first_column + 1, "highres", "Fine"),
        ):
            intensity = data[source]
            dx = data[f"{source}_dx"]
            dy = data[f"{source}_dy"]
            extent = (
                0.0,
                intensity.shape[0] * dx,
                0.0,
                intensity.shape[1] * dy,
            )
            ax = axs[row, column]
            image = ax.imshow(
                intensity,
                origin="lower",
                extent=extent,
                cmap="gray",
                vmin=0.0,
                vmax=12.0,
                aspect="equal",
            )
            ax.set_title(title)
            ax.set_xlabel("x [Mm]")
            if column in (0, 2):
                ax.set_ylabel("y [Mm]")
            if source == "lowres":
                ax.text(
                    0.03,
                    0.97,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="white",
                    fontweight="bold",
                )

    if image is not None:
        fig.colorbar(image, ax=axs, label="Line-core intensity", shrink=0.8)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "zero_shot_super_resolution.pdf",
        dpi=300,
        format="pdf",
    )


if __name__ == '__main__':
    # make_branch_importance_plots()
    make_line_core_intensity_compare_plots()
    make_zero_shot_super_resolution_plots()
