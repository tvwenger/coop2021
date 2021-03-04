import numpy as np

x = np.array(
    [ 0.52848393,  0.52241911,  0.50951003,  0.68051412,  0.55079817,
  1.31866569,  0.94199302,  1.02373628,  1.64661405,  4.4266107 ,
  2.2530227 ,  3.20335548,  2.37259132,  1.77249446,  2.00126823,
  3.65614944,  1.59598175,  4.29274698,  5.2588271 ,  4.87063974,
  2.39585762,  6.91442986,  6.48599696,  8.1383898 ,  5.75329861,
  3.89882719, 12.7942293 ,  2.62628549,  1.51412745,  2.70453862,
  2.3401526 ,  2.9860919 ,  3.10629653,  2.46047618,  1.45495123,
 -0.29705464, -0.30817625,  0.0160902 ,  0.30693676,  0.41084152,
  2.65344589,  0.22671741,  0.5293325 ,  0.24907593,  0.69527461,
  0.52768125,  0.64658902,  0.55210512,  0.54721684,  0.56412736,
  0.89793454,  0.94402077,  0.27718161,  0.46277915,  0.51980708,
  1.0230826 ,  0.60674836,  0.45292341,  0.62904476,  1.08916837,
  1.90672517,  1.64869273,  1.90850874,  1.28196526,  1.72237524,
  2.14547242,  2.05877415,  2.41121554,  2.40149023,  2.41219853,
  3.33713699,  1.56832418,  1.75042024,  6.82457239,  2.8219139 ,
  1.95196715,  2.86175861,  2.67970209,  4.07454999,  4.22892258,
  1.66213221,  1.49718082,  1.33469372,  1.26391131,  1.39848677,
  5.17568375,  1.14204618,  2.55079468,  2.17691588,  8.31284965,
  5.26486239,  5.83210908,  9.43060102,  5.24971815,  7.60132788,
  4.16917943,  5.1732754 ,  5.48823365,  5.98905468,  4.9524976 ,
  5.23319371,  5.96860892,  8.06647113,  4.23924638,  4.60408848,
  3.8419812 ,  6.70553623,  3.14802139,  4.10950693,  3.49323205,
  4.78155059,  3.50710839,  3.1286739 ,  3.19005247,  2.85980007,
  1.61014778,  1.86635629,  3.57232044,  7.19834393,  2.3073509 ,
  6.02775692,  4.44685253,  3.42391737,  1.52782331,  8.95361718,
  3.40091165,  3.71398458,  1.26211403,  3.27056846,  1.33493391,
  1.58745336,  1.59126226,  1.43709463,  1.48589009,  1.28220032,
  0.67374485,  5.84520287,  1.62719369,  4.50810377,  4.83301031,
  7.44237189,  3.39700223,  0.84118967,  0.7376802 ,  0.86596795,
  4.18448831,  2.37154381,  3.06583519,  0.76532   ,  3.32817966,
  2.5446013 ,  0.79333427,  1.8292477 ,  2.34642281,  1.9785933 ,
  1.40595973,  1.72217148,  4.20855322,  1.38975653,  1.10598902,
  0.3050625 ,  0.19092567,  0.21636571,  0.05855199, -0.1029212 ,
 -0.32665947, -0.31845395, -0.36297425, -1.16990831, -0.18891791,
 -2.19160768, -0.47170796, -3.75932768, -3.49178123, -1.33307819,
 -2.56570634, -1.00227515, -4.64580523, -3.2685687 , -0.71632416,
 -0.66164665, -0.19780827, -0.01793087]
)

y = np.array(
    [4.92364912,   5.31311558,   5.46638387,   5.93654582,   6.38225185,
     4.35355264,   5.50117487,   5.28975234,   4.17774697,  -1.01257039,
     4.2572754 ,   2.62890125,   4.1045428 ,   5.13892356,   4.77293067,
     2.04519849,   5.50597512,   1.50601364,   0.02026149,   0.792754  ,
     4.89761027,  -0.83652166,  -0.10909025,  -1.30628078,   3.25287322,
     4.85023093,   4.42856765,   7.45643312,   7.86283118,   8.56314094,
     8.96865352,   9.27971325,   9.38844023,   9.15263738,   9.733154  ,
     14.5337487,  10.17344442,   5.25745539,   5.20320964,   4.73357774,
   -12.05356557,   6.6356576 ,   5.48559642,   6.95687882,   4.88714766,
     5.83616595,   5.33853923,   5.76855315,   5.79227906,   5.7199621 ,
     4.50146031,   4.35924606,   7.09646939,   6.40939919,   6.24610297,
     4.74625309,   6.26296169,   6.75686832,   6.2838269 ,   4.9953893 ,
     3.69155832,   4.33589552,   3.76872863,   5.25496555,   4.4252865 ,
     4.19758714,   4.37325063,   3.8012442 ,   3.82421883,   3.99578552,
     2.561575  ,   5.54511397,   5.26499785,  -3.06827716,   3.53661733,
     4.98497318,   3.52616435,   3.90038201,   1.92919749,   1.82630718,
     5.75489935,   6.02656214,   6.27695947,   6.38948243,   6.19896856,
     1.00320041,   6.68918105,   4.93044709,   5.6127934 ,  -1.57769286,
     2.15805349,   1.52553642,  -2.27977577,   2.55926353,   0.077252  ,
     3.83309397,   2.80382578,   2.70672916,   2.28664646,   3.31331178,
     3.09306856,   2.46905381,   1.07134481,   4.494969  ,   4.18524017,
     4.86392177,   2.40655441,   5.47831203,   4.67010971,   5.20833818,
     4.45900627,   5.64356457,   6.02066453,   6.03730563,   6.44765535,
     7.23207939,   7.09436452,   6.10492479,   4.12126603,   7.32055278,
     6.00884827,   6.67723161,   7.03721485,   7.74428685,   5.83170359,
     7.31831655,   7.24032288,   7.87558874,   7.5388807 ,   7.93960588,
     7.89790838,   7.92362533,   7.95017876,   7.9659684 ,   7.99820917,
     8.18381771,   8.27527514,   8.25721696,   8.54436924,   8.62945368,
     9.16493589,   8.80361763,   8.41320994,   8.41104958,   8.46576135,
     9.55773611,   8.97141556,   9.20557383,   8.45794934,   9.47602694,
     9.37107672,   8.66366742,   9.32527305,   9.70921967,   9.46965303,
     9.53656408,   9.88092749,  12.43071024,  12.03011602,  13.41271682,
     10.03612893,   9.85240746,  10.31173027,   9.14296378,   9.76466306,
     10.25637491,   9.65929693,   9.80518203,  12.14275346,   8.52221495,
     11.74468722,   8.88870541,  13.02821721,  11.15613611,   9.19979887,
     9.85921506,   8.77530763,  10.82960529,   5.87562547,   6.22561998,
     4.86902446,   6.86644794,   5.51478809]
)

# coord_obs = np.array(
#   [[ 0.52848393,  4.92364912],
#    [ 0.52241911,  5.31311558],
#    [ 0.68051412,  5.93654582],
#    [ 0.55079817,  6.38225185],
#    [ 0.94199302,  5.50117487],
#    [ 1.64661405,  4.17774697],
#    [ 2.39585762,  4.89761027],
#    [ 3.89882719,  4.85023093],
#    [ 2.62628549,  7.45643312],
#    [ 3.10629653,  9.38844023],
#    [ 2.46047618,  9.15263738],
#    [ 1.45495123,  9.733154  ],
#    [-0.29705464, 14.5337487 ],
#    [-0.30817625, 10.17344442],
#    [ 0.0160902 ,  5.25745539],
#    [ 0.30693676,  5.20320964],
#    [ 0.41084152,  4.73357774],
#    [ 0.22671741,  6.6356576 ],
#    [ 0.5293325 ,  5.48559642],
#    [ 0.24907593,  6.95687882],
#    [ 0.69527461,  4.88714766],
#    [ 0.64658902,  5.33853923],
#    [ 0.55210512,  5.76855315],
#    [ 0.54721684,  5.79227906],
#    [ 0.56412736,  5.7199621 ],
#    [ 0.89793454,  4.50146031],
#    [ 0.94402077,  4.35924606],
#    [ 0.27718161,  7.09646939],
#    [ 0.46277915,  6.40939919],
#    [ 0.51980708,  6.24610297],
#    [ 1.0230826 ,  4.74625309],
#    [ 0.60674836,  6.26296169],
#    [ 0.45292341,  6.75686832],
#    [ 0.62904476,  6.2838269 ],
#    [ 1.08916837,  4.9953893 ],
#    [ 1.90672517,  3.69155832],
#    [ 1.64869273,  4.33589552],
#    [ 1.90850874,  3.76872863],
#    [ 1.28196526,  5.25496555],
#    [ 2.14547242,  4.19758714],
#    [ 2.05877415,  4.37325063],
#    [ 2.41121554,  3.8012442 ],
#    [ 3.33713699,  2.561575  ],
#    [ 1.75042024,  5.26499785],
#    [ 6.82457239, -3.06827716],
#    [ 2.8219139 ,  3.53661733],
#    [ 2.86175861,  3.52616435],
#    [ 2.67970209,  3.90038201],
#    [ 1.49718082,  6.02656214],
#    [ 1.33469372,  6.27695947],
#    [ 1.26391131,  6.38948243],
#    [ 1.39848677,  6.19896856],
#    [ 5.17568375,  1.00320041],
#    [ 1.14204618,  6.68918105],
#    [ 2.55079468,  4.93044709],
#    [ 2.17691588,  5.6127934 ],
#    [ 5.26486239,  2.15805349],
#    [ 5.83210908,  1.52553642],
#    [ 5.24971815,  2.55926353],
#    [ 7.60132788,  0.077252  ],
#    [ 5.48823365,  2.70672916],
#    [ 4.9524976 ,  3.31331178],
#    [ 5.23319371,  3.09306856],
#    [ 5.96860892,  2.46905381],
#    [ 8.06647113,  1.07134481],
#    [ 4.23924638,  4.494969  ],
#    [ 4.60408848,  4.18524017],
#    [ 3.8419812 ,  4.86392177],
#    [ 6.70553623,  2.40655441],
#    [ 3.14802139,  5.47831203],
#    [ 4.10950693,  4.67010971],
#    [ 3.49323205,  5.20833818],
#    [ 4.78155059,  4.45900627],
#    [ 3.50710839,  5.64356457],
#    [ 3.1286739 ,  6.02066453],
#    [ 3.19005247,  6.03730563],
#    [ 2.85980007,  6.44765535],
#    [ 1.61014778,  7.23207939],
#    [ 1.86635629,  7.09436452],
#    [ 3.57232044,  6.10492479],
#    [ 7.19834393,  4.12126603],
#    [ 2.3073509 ,  7.32055278],
#    [ 6.02775692,  6.00884827],
#    [ 4.44685253,  6.67723161],
#    [ 3.42391737,  7.03721485],
#    [ 1.52782331,  7.74428685],
#    [ 3.40091165,  7.31831655],
#    [ 3.71398458,  7.24032288],
#    [ 1.33493391,  7.93960588],
#    [ 1.58745336,  7.89790838],
#    [ 1.59126226,  7.92362533],
#    [ 1.43709463,  7.95017876],
#    [ 1.48589009,  7.9659684 ],
#    [ 1.28220032,  7.99820917],
#    [ 0.67374485,  8.18381771],
#    [ 1.62719369,  8.25721696],
#    [ 4.83301031,  8.62945368],
#    [ 7.44237189,  9.16493589],
#    [ 3.39700223,  8.80361763],
#    [ 0.84118967,  8.41320994],
#    [ 0.7376802 ,  8.41104958],
#    [ 0.86596795,  8.46576135],
#    [ 4.18448831,  9.55773611],
#    [ 2.37154381,  8.97141556],
#    [ 3.06583519,  9.20557383],
#    [ 0.76532   ,  8.45794934],
#    [ 3.32817966,  9.47602694],
#    [ 2.5446013 ,  9.37107672],
#    [ 0.79333427,  8.66366742],
#    [ 1.8292477 ,  9.32527305],
#    [ 1.9785933 ,  9.46965303],
#    [ 1.40595973,  9.53656408],
#    [ 1.72217148,  9.88092749],
#    [ 4.20855322, 12.43071024],
#    [ 1.38975653, 12.03011602],
#    [ 1.10598902, 13.41271682],
#    [ 0.3050625 , 10.03612893],
#    [ 0.19092567,  9.85240746],
#    [ 0.21636571, 10.31173027],
#    [ 0.05855199,  9.14296378],
#    [-0.1029212 ,  9.76466306],
#    [-0.32665947, 10.25637491],
#    [-0.31845395,  9.65929693],
#    [-0.36297425,  9.80518203],
#    [-1.16990831, 12.14275346],
#    [-0.18891791,  8.52221495],
#    [-2.19160768, 11.74468722],
#    [-0.47170796,  8.88870541],
#    [-3.75932768, 13.02821721],
#    [-3.49178123, 11.15613611],
#    [-1.33307819,  9.19979887],
#    [-2.56570634,  9.85921506],
#    [-1.00227515,  8.77530763],
#    [-4.64580523, 10.82960529],
#    [-3.2685687 ,  5.87562547],
#    [-0.71632416,  6.22561998],
#    [-0.66164665,  4.86902446],
#    [-0.19780827,  6.86644794],
#    [-0.01793087,  5.51478809]]
# )

# Upec_good = np.array(
#   [ 20.18355779,  -6.67225625,  -6.57607902,   4.2000377 ,  -2.93899109,
#   25.89094442,  -4.52511784,   0.73370822,  -2.83111961,  32.68975895,
#   17.89369676,   7.53793282,  11.32956242,  14.20989142,  17.41991785,
#   -2.74776503,   3.02948033,  -2.26508969,   8.75538996,  -0.75857301,
#    2.45156726,   1.89877773,  14.4757572 ,  16.20324128,  16.4691059 ,
#    4.76215892,   7.60173252,  11.13272519,   2.24063214,   0.40765283,
#   21.17304858,  20.40637146,   7.46848207,   5.27474329,  -7.22304519,
#   15.40465985,  20.39350614,   9.12262145,  33.20050096,  12.16744682,
#   -2.7392967 ,  11.83684225, -13.22733147,  24.61450473, -16.45726849,
#    6.39225781,  -3.33890512,  14.4827658 ,  -6.69843273,  11.71074256,
#   -4.16997489,   2.84992572,   3.11944887,   7.45434299,  25.72733066,
#    6.24296021,  -6.27521382, -10.35638682,  22.53793003,  -4.34153262,
#    9.48549083,  -8.67253258,  -1.00814941,   7.2475087 ,  -4.95556097,
#    0.40724713,  27.70908822,   8.38032262,  14.86594333,  11.75454672,
#   -8.94346259,  15.32852912,   5.71525921,  10.48830992,  25.1882342 ,
#   17.69715857,  15.41196898,  14.60313684,  -0.51162659,  16.30296115,
#   17.54757534,   6.5120495 ,  11.02541716,   6.32408232,  -4.5062177 ,
#    7.62634163,  -0.05192267,  -4.12803192,   1.23924613,   8.49947151,
#   -3.0180938 ,   6.85434609,  -0.62909728,  -1.11252452,  -4.79165286,
#  -17.00901466,  -7.55166034,   9.67825334,  12.81364589,   7.75288574,
#    4.32034847,  -2.07193877, -11.16170318,  12.29317425,  20.14426843,
#    2.81189181,   1.06238629,  16.15820636,   9.56820964,  29.96751715,
#    7.29434256,  20.53210071,   3.00680804,  14.63624601, -11.64345667,
#    5.23695758,  10.95683554,   8.46563152,  -1.7432229 ,  15.00882178,
#   -0.93246002,  -0.25977681,   2.72111659,  -0.06096162,   4.14341847,
#   -0.82552185,  -5.00208561,  -6.33870864,  13.3222905 ,  13.4160818 ,
#    1.62920323,  -7.9231827 ,  -0.61744261,  -7.63792626,  -4.91597903,
#   -6.50673082,  13.98534604,   0.1208676 ,  21.87509731]
# )

# Vpec_good = np.array(
#   [ -3.13660859, -23.58136981, -10.50298905,  -8.2866297 , -16.93772422,
#  -13.12478866,   6.14447393,  -5.41153916, -14.76577414,   4.31874766,
#  -23.84873124, -18.57726424,   0.75620205, -19.80152708,  -6.71101356,
#  -13.7183526 ,  -2.67840572,  -9.44814569,  -7.27826375,   9.50340312,
#    8.42952789,  13.0167603 ,  -4.60677107,  -6.15953849,  -6.25533856,
#   -5.74655299,  -8.35824925,   3.07001128,  -3.2063974 ,   3.76732313,
#   -8.20752294,   5.03050758,   3.25086721,  -1.93323971,  -1.53700485,
#  -17.53128829,  -1.91349383, -16.29736502,   7.9972358 ,  -0.05440143,
#    0.25297802,  -8.72016508, -15.3815636 ,   1.69090597,  -7.31086433,
#   13.76046046,   2.18336992,   3.52057395,   3.87794904,   4.09677884,
#   -8.16982944,  -2.59542262, -16.66841755,   3.32580867,  -6.81553129,
#   14.10019688, -13.97270252,  -5.7622628 ,  -5.98895292, -12.72993593,
#    1.39644101, -10.94167698,   0.73426267, -21.0304591 ,  13.0737745 ,
#    4.40827141, -19.83149011,   4.70182425, -27.78389309,   8.17972464,
#   -3.54478259,   1.49863691,  -9.33601155, -10.48748775, -12.78014775,
#  -20.78541662,  -6.9809948 ,  -5.4451151 ,  -6.14142246,  -3.66502659,
#   13.32184063,  -8.45119181, -15.46964431,   1.21442124,  -5.31701925,
#   -9.11244992, -17.76813201,  -6.48112162, -12.25955831, -13.88645591,
#  -11.47969038, -11.20728373, -10.68022389,  -0.70943807,  -8.06969269,
#   -7.52816223,  -5.10814247,   6.91005019,  -2.91849337,  -2.62299488,
#   -8.84381144,  -8.13796992, -13.34643462, -25.21988201, -12.51105694,
#   -2.40726834,  -1.04537112,   7.21767773, -11.30673054,  -6.06192128,
#    4.38573089, -11.8808445 , -13.07423799,   8.41106442, -16.64527498,
#   -6.11731257, -11.91333153,  -5.67946895,  10.86633946, -15.75837206,
#    1.94346346,  -4.53497842,  -4.67289639,   6.37175119,  -5.72857724,
#   -3.79742577,  -4.27322082,   4.86927642,  -9.92200509,  -9.02854072,
#   -3.30190403,  -0.09246264, -10.08428383,   0.70542043,  -5.06797262,
#    1.77465458, -25.15735207,   1.38799339,   1.6087242 ]
# )


coord_obs = np.array(
    [[ 0.52848393,  4.92364912], [ 0.52241911,  5.31311558],
    [ 0.68051412,  5.93654582], [ 0.55079817,  6.38225185],
    [ 0.94199302,  5.50117487], [ 1.64661405,  4.17774697],
    [ 2.39585762,  4.89761027], [ 3.89882719,  4.85023093],
    [ 2.62628549,  7.45643312], [ 3.10629653,  9.38844023],
    [ 2.46047618,  9.15263738], [ 1.45495123,  9.733154  ],
    [-0.29705464, 14.5337487 ], [-0.30817625, 10.17344442],
    [ 0.0160902 ,  5.25745539], [ 0.30693676,  5.20320964],
    [ 0.41084152,  4.73357774], [ 0.22671741,  6.6356576 ],
    [ 0.5293325 ,  5.48559642], [ 0.24907593,  6.95687882],
    [ 0.69527461,  4.88714766], [ 0.64658902,  5.33853923],
    [ 0.55210512,  5.76855315], [ 0.54721684,  5.79227906],
    [ 0.56412736,  5.7199621 ], [ 0.89793454,  4.50146031],
    [ 0.94402077,  4.35924606], [ 0.27718161,  7.09646939],
    [ 0.46277915,  6.40939919], [ 0.51980708,  6.24610297],
    [ 1.0230826 ,  4.74625309], [ 0.60674836,  6.26296169],
    [ 0.45292341,  6.75686832], [ 0.62904476,  6.2838269 ],
    [ 1.08916837,  4.9953893 ], [ 1.90672517,  3.69155832],
    [ 1.64869273,  4.33589552], [ 1.90850874,  3.76872863],
    [ 1.28196526,  5.25496555], [ 2.14547242,  4.19758714],
    [ 2.05877415,  4.37325063], [ 2.41121554,  3.8012442 ],
    [ 3.33713699,  2.561575  ], [ 1.75042024,  5.26499785],
    [ 6.82457239, -3.06827716], [ 2.8219139 ,  3.53661733],
    [ 2.86175861,  3.52616435], [ 2.67970209,  3.90038201],
    [ 1.49718082,  6.02656214], [ 1.33469372,  6.27695947],
    [ 1.26391131,  6.38948243], [ 1.39848677,  6.19896856],
    [ 5.17568375,  1.00320041], [ 1.14204618,  6.68918105],
    [ 2.55079468,  4.93044709], [ 2.17691588,  5.6127934 ],
    [ 5.26486239,  2.15805349], [ 5.83210908,  1.52553642],
    [ 5.24971815,  2.55926353], [ 7.60132788,  0.077252  ],
    [ 5.48823365,  2.70672916], [ 4.9524976 ,  3.31331178],
    [ 5.23319371,  3.09306856], [ 5.96860892,  2.46905381],
    [ 8.06647113,  1.07134481], [ 4.23924638,  4.494969  ],
    [ 4.60408848,  4.18524017], [ 3.8419812 ,  4.86392177],
    [ 6.70553623,  2.40655441], [ 3.14802139,  5.47831203],
    [ 4.10950693,  4.67010971], [ 3.49323205,  5.20833818],
    [ 4.78155059,  4.45900627], [ 3.50710839,  5.64356457],
    [ 3.1286739 ,  6.02066453], [ 3.19005247,  6.03730563],
    [ 2.85980007,  6.44765535], [ 1.61014778,  7.23207939],
    [ 1.86635629,  7.09436452], [ 3.57232044,  6.10492479],
    [ 7.19834393,  4.12126603], [ 2.3073509 ,  7.32055278],
    [ 6.02775692,  6.00884827], [ 4.44685253,  6.67723161],
    [ 3.42391737,  7.03721485], [ 1.52782331,  7.74428685],
    [ 3.40091165,  7.31831655], [ 3.71398458,  7.24032288],
    [ 1.33493391,  7.93960588], [ 1.58745336,  7.89790838],
    [ 1.59126226,  7.92362533], [ 1.43709463,  7.95017876],
    [ 1.48589009,  7.9659684 ], [ 1.28220032,  7.99820917],
    [ 0.67374485,  8.18381771], [ 1.62719369,  8.25721696],
    [ 4.83301031,  8.62945368], [ 7.44237189,  9.16493589],
    [ 3.39700223,  8.80361763], [ 0.84118967,  8.41320994],
    [ 0.7376802 ,  8.41104958], [ 0.86596795,  8.46576135],
    [ 4.18448831,  9.55773611], [ 2.37154381,  8.97141556],
    [ 3.06583519,  9.20557383], [ 0.76532   ,  8.45794934],
    [ 3.32817966,  9.47602694], [ 2.5446013 ,  9.37107672],
    [ 0.79333427,  8.66366742], [ 1.8292477 ,  9.32527305],
    [ 1.9785933 ,  9.46965303], [ 1.40595973,  9.53656408],
    [ 1.72217148,  9.88092749], [ 4.20855322, 12.43071024],
    [ 1.38975653, 12.03011602], [ 1.10598902, 13.41271682],
    [ 0.3050625 , 10.03612893], [ 0.19092567,  9.85240746],
    [ 0.21636571, 10.31173027], [ 0.05855199,  9.14296378],
    [-0.1029212 ,  9.76466306], [-0.32665947, 10.25637491],
    [-0.31845395,  9.65929693], [-0.36297425,  9.80518203],
    [-1.16990831, 12.14275346], [-0.18891791,  8.52221495],
    [-2.19160768, 11.74468722], [-0.47170796,  8.88870541],
    [-3.75932768, 13.02821721], [-3.49178123, 11.15613611],
    [-1.33307819,  9.19979887], [-2.56570634,  9.85921506],
    [-1.00227515,  8.77530763], [-4.64580523, 10.82960529],
    [-3.2685687 ,  5.87562547], [-0.71632416,  6.22561998],
    [-0.66164665,  4.86902446], [-0.19780827,  6.86644794],
    [-0.01793087,  5.51478809]]
)

Upec_good = np.array(
    [ 20.18355779,  -6.67225625,  -6.57607902,   4.2000377 ,  -2.93899109,
    25.89094442,  -4.52511784,   0.73370822,  -2.83111961,  32.68975895,
    17.89369676,   7.53793282,  11.32956242,  14.20989142,  17.41991785,
    -2.74776503,   3.02948033,  -2.26508969,   8.75538996,  -0.75857301,
    2.45156726,   1.89877773,  14.4757572 ,  16.20324128,  16.4691059 ,
    4.76215892,   7.60173252,  11.13272519,   2.24063214,   0.40765283,
    21.17304858,  20.40637146,   7.46848207,   5.27474329,  -7.22304519,
    15.40465985,  20.39350614,   9.12262145,  33.20050096,  12.16744682,
    -2.7392967 ,  11.83684225, -13.22733147,  24.61450473, -16.45726849,
    6.39225781,  -3.33890512,  14.4827658 ,  -6.69843273,  11.71074256,
    -4.16997489,   2.84992572,   3.11944887,   7.45434299,  25.72733066,
    6.24296021,  -6.27521382, -10.35638682,  22.53793003,  -4.34153262,
    9.48549083,  -8.67253258,  -1.00814941,   7.2475087 ,  -4.95556097,
    0.40724713,  27.70908822,   8.38032262,  14.86594333,  11.75454672,
    -8.94346259,  15.32852912,   5.71525921,  10.48830992,  25.1882342 ,
    17.69715857,  15.41196898,  14.60313684,  -0.51162659,  16.30296115,
    17.54757534,   6.5120495 ,  11.02541716,   6.32408232,  -4.5062177 ,
    7.62634163,  -0.05192267,  -4.12803192,   1.23924613,   8.49947151,
    -3.0180938 ,   6.85434609,  -0.62909728,  -1.11252452,  -4.79165286,
    -17.00901466,  -7.55166034,   9.67825334,  12.81364589,   7.75288574,
    4.32034847,  -2.07193877, -11.16170318,  12.29317425,  20.14426843,
    2.81189181,   1.06238629,  16.15820636,   9.56820964,  29.96751715,
    7.29434256,  20.53210071,   3.00680804,  14.63624601, -11.64345667,
    5.23695758,  10.95683554,   8.46563152,  -1.7432229 ,  15.00882178,
    -0.93246002,  -0.25977681,   2.72111659,  -0.06096162,   4.14341847,
    -0.82552185,  -5.00208561,  -6.33870864,  13.3222905 ,  13.4160818 ,
    1.62920323,  -7.9231827 ,  -0.61744261,  -7.63792626,  -4.91597903,
    -6.50673082,  13.98534604,   0.1208676 ,  21.87509731]
)

Vpec_good = np.array(
    [ -3.13660859, -23.58136981, -10.50298905,  -8.2866297 , -16.93772422,
    -13.12478866,   6.14447393,  -5.41153916, -14.76577414,   4.31874766,
    -23.84873124, -18.57726424,   0.75620205, -19.80152708,  -6.71101356,
    -13.7183526 ,  -2.67840572,  -9.44814569,  -7.27826375,   9.50340312,
    8.42952789,  13.0167603 ,  -4.60677107,  -6.15953849,  -6.25533856,
    -5.74655299,  -8.35824925,   3.07001128,  -3.2063974 ,   3.76732313,
    -8.20752294,   5.03050758,   3.25086721,  -1.93323971,  -1.53700485,
    -17.53128829,  -1.91349383, -16.29736502,   7.9972358 ,  -0.05440143,
    0.25297802,  -8.72016508, -15.3815636 ,   1.69090597,  -7.31086433,
    13.76046046,   2.18336992,   3.52057395,   3.87794904,   4.09677884,
    -8.16982944,  -2.59542262, -16.66841755,   3.32580867,  -6.81553129,
    14.10019688, -13.97270252,  -5.7622628 ,  -5.98895292, -12.72993593,
    1.39644101, -10.94167698,   0.73426267, -21.0304591 ,  13.0737745 ,
    4.40827141, -19.83149011,   4.70182425, -27.78389309,   8.17972464,
    -3.54478259,   1.49863691,  -9.33601155, -10.48748775, -12.78014775,
    -20.78541662,  -6.9809948 ,  -5.4451151 ,  -6.14142246,  -3.66502659,
    13.32184063,  -8.45119181, -15.46964431,   1.21442124,  -5.31701925,
    -9.11244992, -17.76813201,  -6.48112162, -12.25955831, -13.88645591,
    -11.47969038, -11.20728373, -10.68022389,  -0.70943807,  -8.06969269,
    -7.52816223,  -5.10814247,   6.91005019,  -2.91849337,  -2.62299488,
    -8.84381144,  -8.13796992, -13.34643462, -25.21988201, -12.51105694,
    -2.40726834,  -1.04537112,   7.21767773, -11.30673054,  -6.06192128,
    4.38573089, -11.8808445 , -13.07423799,   8.41106442, -16.64527498,
    -6.11731257, -11.91333153,  -5.67946895,  10.86633946, -15.75837206,
    1.94346346,  -4.53497842,  -4.67289639,   6.37175119,  -5.72857724,
    -3.79742577,  -4.27322082,   4.86927642,  -9.92200509,  -9.02854072,
    -3.30190403,  -0.09246264, -10.08428383,   0.70542043,  -5.06797262,
    1.77465458, -25.15735207,   1.38799339,   1.6087242 ]
)

print(coord_obs.shape, Upec_good.shape, Vpec_good.shape)
