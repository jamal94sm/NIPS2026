"""
Parameter configurations for each palmprint database.
Translated from MATLAB params/*.m files.

Each entry includes a 'meta' key with dataset-level info (image extension, etc.)
that main.py uses automatically — no need to touch main.py when switching datasets.
"""


def _base_params():
    """Shared defaults across all databases."""
    return {
        'meta': {
            'image_ext': 'bmp',
        },
        'segm': {
            'resizeF'              : 1,
            'colorSpaceTrans'      : 'rgb2ycbcr',
            'th_em'                : 0.85,
            'thSegmRed'            : 0.0,
            'numCCbinar'           : 10,
            'thVessModVe'          : 0.0,
            'thVessModHo'          : 0.0,
            'indKirschVer'         : [1, 2],
            'indKirschHor'         : [5, 6],
            'edgeVeSearchAngle'    : 5,
            'edgeHoSearchAngle'    : 2,
            'useEdgeAdd'           : True,
            'subEdgesVer'          : True,
            'useRefl'              : True,
            'reflEdgeMult'         : 1.0,
            'thArea'               : 300,
            'fGauss_size'          : (5, 5),
            'fGauss_sigma'         : 2,
            'typeStrel'            : 'disk',
            'sizeStrel_small'      : 1,
            'sizeStrel_medium'     : 3,
            'sizeStrel_large'      : 5,
            'typeStrelEdge'        : 'line',
            'sizeStrelEdge_small'  : 3,
            'sizeStrelEdge_medium' : 5,
            'sizeStrelEdge_large'  : 15,
            'sizeStrelEdge_huge'   : 20,
            'sizeShapeInterp'      : 3000,
            'smoothShapeSize'      : 50,
            'smoothShapeOrder'     : 2,
            'normalizza'           : 0,
        },
        'peakFind': {
            'numIterCentroid' : 1,
            'smoothF'         : 150,
            'minPeakDistance' : 3000 // 60,
            'meanPksMult'     : 1.2,
        },
        'localsearch': {
            'beta'        : 20,
            'alpha'       : 20,
            'mu'          : 20,
            'offset'      : 300,
            'maxDistance' : 50,
            'stepSearch'  : 5,
            'stepAngle'   : 5,
        },
        'rejectPoints': {
            'thAngle'              : 35,
            'thDiffAngle'          : 10,
            'percBlackPixels'      : 0.05,
            'percCheckExtArea'     : 0.3,
            'numStepsCheckExtArea' : 10,
            'minDistanceBorder'    : 5,
        },
        'ROIsize': {
            'useRefineGrad'  : True,
            'sizeROI'        : (150, 150),
            'multX'          : 1.4,
            'multY'          : 1 + 2 / 5,
            'multOffset'     : 1 / 5,
            'percBlackPixels': 0.30,
        },
        'prepProc': {
            'colorSpaceTransSingle' : 'rgb2gray',
            'colorSpaceTransMulti'  : 'rgb2hsv',
            'padSize'               : 5,
            'alphaLaplacian'        : 0.2,
            'sigmaGauss'            : 1.0,
        },
    }


# ---------------------------------------------------------------------------
# CASIA Palmprint Database V1
# ---------------------------------------------------------------------------
def get_params_casia():
    p = _base_params()
    p['meta']['image_ext'] = 'bmp'
    p['segm'].update({
        'colorSpaceTrans' : 'rgb2ycbcr',
        'th_em'           : 0.85,
        'thSegmRed'       : 0.0,
        'numCCbinar'      : 10,
        'fGauss_size'     : (5, 5),
        'fGauss_sigma'    : 2,
        'sizeStrel_large' : 5,
        'sizeStrelEdge_huge': 20,
    })
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


# ---------------------------------------------------------------------------
# IIT Delhi Touchless Palmprint Database v1.0
# ---------------------------------------------------------------------------
def get_params_iitd():
    p = _base_params()
    p['meta']['image_ext'] = 'bmp'
    p['segm'].update({
        'colorSpaceTrans'    : 'rgb2ycbcr',
        'th_em'              : 0.90,
        'numCCbinar'         : 1,
        'thVessModVe'        : 0.03,
        'edgeVeSearchAngle'  : 2,
        'fGauss_size'        : (15, 15),
        'fGauss_sigma'       : 3,
        'sizeStrel_large'    : 10,
        'sizeStrelEdge_small'  : 1,
        'sizeStrelEdge_medium' : 5,
        'sizeStrelEdge_large'  : 10,
        'sizeStrelEdge_huge'   : 15,
    })
    p['localsearch'].update({'beta': 10, 'alpha': 10, 'mu': 10})
    return p


# ---------------------------------------------------------------------------
# REST Hand Database 2016
# ---------------------------------------------------------------------------
def get_params_rest():
    p = _base_params()
    p['meta']['image_ext'] = 'jpg'
    p['segm'].update({
        'resizeF'         : 0.5,
        'colorSpaceTrans' : 'rgb2ycbcr',
        'th_em'           : 0.85,
    })
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


# ---------------------------------------------------------------------------
# Tongji Contactless Palmprint Dataset
# ---------------------------------------------------------------------------
def get_params_tongji():
    p = _base_params()
    p['meta']['image_ext'] = 'bmp'
    p['segm'].update({
        'colorSpaceTrans'    : 'rgb2hsv',
        'th_em'              : 0.85,
        'thSegmRed'          : -0.10,
        'numCCbinar'         : 30,
        'thVessModVe'        : 0.07,
        'thVessModHo'        : 0.07,
        'edgeVeSearchAngle'  : 2,
        'edgeHoSearchAngle'  : 2,
        'useEdgeAdd'         : False,
        'subEdgesVer'        : False,
        'useRefl'            : False,
        'normalizza'         : 1,
    })
    p['rejectPoints']['minDistanceBorder'] = 50
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


# ---------------------------------------------------------------------------
# MPDv2  –  Mobile Palmprint Database v2
#
# Characteristics:
#   • Colour JPEG images from Huawei / Xiaomi smartphones
#   • Very high resolution (full camera sensor)
#   • Complex, varied backgrounds  (no enclosure / uniform backdrop)
#   • Variable illumination across sessions
#   • Unconstrained hand pose and orientation
#
# Strategy:
#   • HSV colour space — V channel + normalisation handles exposure variation
#   • resizeF 0.5 — reduces memory use and suppresses high-freq background noise
#   • Lower th_em — complex backgrounds reduce Otsu separability
#   • Full edge pipeline (useEdgeAdd + subEdgesVer + useRefl) to recover
#     palm boundary on busy backgrounds
#   • Relaxed rejection thresholds — more background leakage tolerated
# ---------------------------------------------------------------------------
def get_params_mpdv2():
    p = _base_params()
    p['meta']['image_ext'] = 'jpg'
    p['segm'].update({
        'colorSpaceTrans'    : 'rgb2hsv',   # V channel handles skin vs clutter
        'resizeF'            : 0.5,          # high-res phone images; halve first
        'th_em'              : 0.80,         # complex bg lowers Otsu separability
        'thSegmRed'          : -0.05,        # slightly softer binarisation
        'numCCbinar'         : 30,
        'thVessModVe'        : 0.00,
        'thVessModHo'        : 0.00,
        'edgeVeSearchAngle'  : 5,
        'edgeHoSearchAngle'  : 2,
        'useEdgeAdd'         : True,         # needed on complex backgrounds
        'subEdgesVer'        : True,
        'useRefl'            : True,
        'fGauss_size'        : (7, 7),       # more smoothing for noisy backgrounds
        'fGauss_sigma'       : 3,
        'normalizza'         : 1,            # compensates variable phone exposure
    })
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    p['rejectPoints'].update({
        'percBlackPixels'   : 0.10,          # more bg inside triangle tolerated
        'minDistanceBorder' : 10,            # hand can appear anywhere in frame
    })
    p['ROIsize']['percBlackPixels'] = 0.40   # more bg leakage tolerated in ROI
    return p


# ---------------------------------------------------------------------------
# Registry  –  maps the DATABASE string in main.py to a param function
# ---------------------------------------------------------------------------
DB_PARAMS = {
    'CASIA'  : get_params_casia,
    'IITD'   : get_params_iitd,
    'REST'   : get_params_rest,
    'Tongji' : get_params_tongji,
    'MPDv2'  : get_params_mpdv2,
}


def get_params(db_name):
    fn = DB_PARAMS.get(db_name)
    if fn is None:
        raise ValueError(
            f"Unknown database: '{db_name}'. "
            f"Available options: {list(DB_PARAMS)}"
        )
    return fn()
