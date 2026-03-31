"""
Parameter configurations for each palmprint database.
Translated from MATLAB params/*.m files.
"""


def _base_params():
    """Shared defaults across databases."""
    return {
        'segm': {
            'resizeF': 1,
            'colorSpaceTrans': 'rgb2ycbcr',
            'th_em': 0.85,
            'thSegmRed': 0.0,
            'numCCbinar': 10,
            'thVessModVe': 0.0,
            'thVessModHo': 0.0,
            'indKirschVer': [1, 2],   # 1-indexed (matches MATLAB)
            'indKirschHor': [5, 6],   # 1-indexed
            'edgeVeSearchAngle': 5,
            'edgeHoSearchAngle': 2,
            'useEdgeAdd': True,
            'subEdgesVer': True,
            'useRefl': True,
            'reflEdgeMult': 1.0,
            'thArea': 300,
            'fGauss_size': (5, 5),
            'fGauss_sigma': 2,
            'typeStrel': 'disk',
            'sizeStrel_small': 1,
            'sizeStrel_medium': 3,
            'sizeStrel_large': 5,
            'typeStrelEdge': 'line',
            'sizeStrelEdge_small': 3,
            'sizeStrelEdge_medium': 5,
            'sizeStrelEdge_large': 15,
            'sizeStrelEdge_huge': 20,
            'sizeShapeInterp': 3000,
            'smoothShapeSize': 50,
            'smoothShapeOrder': 2,
            'normalizza': 0,
        },
        'peakFind': {
            'numIterCentroid': 1,
            'smoothF': 150,
            'minPeakDistance': 3000 // 60,  # sizeShapeInterp / 60
            'meanPksMult': 1.2,
        },
        'localsearch': {
            'beta': 20,
            'alpha': 20,
            'mu': 20,
            'offset': 300,
            'maxDistance': 50,
            'stepSearch': 5,
            'stepAngle': 5,
        },
        'rejectPoints': {
            'thAngle': 35,
            'thDiffAngle': 10,
            'percBlackPixels': 0.05,
            'percCheckExtArea': 0.3,
            'numStepsCheckExtArea': 10,
            'minDistanceBorder': 5,
        },
        'ROIsize': {
            'useRefineGrad': True,
            'sizeROI': (150, 150),
            'multX': 1.4,
            'multY': 1 + 2 / 5,
            'multOffset': 1 / 5,
            'percBlackPixels': 0.30,
        },
        'prepProc': {
            'colorSpaceTransSingle': 'rgb2gray',
            'colorSpaceTransMulti': 'rgb2hsv',
            'padSize': 5,
            'alphaLaplacian': 0.2,
            'sigmaGauss': 1.0,
        },
    }


def get_params_casia():
    p = _base_params()
    p['segm'].update({
        'resizeF': 1,
        'colorSpaceTrans': 'rgb2ycbcr',
        'th_em': 0.85,
        'thSegmRed': 0.0,
        'numCCbinar': 10,
        'fGauss_size': (5, 5),
        'fGauss_sigma': 2,
        'sizeStrel_large': 5,
        'sizeStrelEdge_huge': 20,
    })
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


def get_params_iitd():
    p = _base_params()
    p['segm'].update({
        'resizeF': 1,
        'colorSpaceTrans': 'rgb2ycbcr',
        'th_em': 0.90,
        'numCCbinar': 1,
        'thVessModVe': 0.03,
        'edgeVeSearchAngle': 2,
        'fGauss_size': (15, 15),
        'fGauss_sigma': 3,
        'sizeStrel_large': 10,
        'sizeStrelEdge_small': 1,
        'sizeStrelEdge_medium': 5,
        'sizeStrelEdge_large': 10,
        'sizeStrelEdge_huge': 15,
    })
    p['localsearch'].update({'beta': 10, 'alpha': 10, 'mu': 10})
    return p


def get_params_rest():
    p = _base_params()
    p['segm'].update({
        'resizeF': 0.5,
        'colorSpaceTrans': 'rgb2ycbcr',
        'th_em': 0.85,
    })
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


def get_params_tongji():
    p = _base_params()
    p['segm'].update({
        'resizeF': 1,
        'colorSpaceTrans': 'rgb2hsv',
        'th_em': 0.85,
        'thSegmRed': -0.10,
        'numCCbinar': 30,
        'thVessModVe': 0.07,
        'thVessModHo': 0.07,
        'edgeVeSearchAngle': 2,
        'edgeHoSearchAngle': 2,
        'useEdgeAdd': False,
        'subEdgesVer': False,
        'useRefl': False,
        'normalizza': 1,
    })
    p['rejectPoints']['minDistanceBorder'] = 50
    p['localsearch'].update({'beta': 20, 'alpha': 20, 'mu': 20})
    return p


DB_PARAMS = {
    'CASIA': get_params_casia,
    'IITD': get_params_iitd,
    'REST': get_params_rest,
    'Tongji': get_params_tongji,
}


def get_params(db_name='Tongji'):
    fn = DB_PARAMS.get(db_name)
    if fn is None:
        raise ValueError(f"Unknown database: {db_name}. Choose from {list(DB_PARAMS)}")
    return fn()
