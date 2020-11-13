import numpy as np

import unittest

from centroid_assignment import Solution


class TestSolution(unittest.TestCase):

    def test_case_1(self):
        points = np.array([[15.676239431663356, 7.044473002084628], [19.520511773669188, 2.873824457785794],
                           [1.7971462318033637, 10.953276501214628], [7.255145215356087, 17.86454530686228],
                           [14.644271863416837, 4.377123624525012], [16.599258110140887, 1.6204727156055765],
                           [7.0776745091752495, 19.641901093891367], [7.9759631987335355, 0.1362854982900097],
                           [10.50236760065356, 19.270067979055824], [4.7032421883797415, 4.157750743542086],
                           [5.56477106548453, 8.008088463632731], [10.181458078059702, 18.215311160093428],
                           [12.658102569584102, 16.862759542903568], [2.1220864917023796, 6.274491842155565],
                           [12.240542103932725, 18.96454121868744], [9.316518916765094, 8.996582647398064],
                           [17.245161686627842, 15.23631380593298], [0.0839633031430842, 0.1987921831592443],
                           [0.8284491868096566, 3.0959402211894838], [1.832483219579888, 15.56549743421966],
                           [10.073335957605883, 9.195226244555347], [14.460561743957324, 4.898958645580445],
                           [14.594954718692817, 9.465664618969555], [12.605869071220912, 11.86359269777645],
                           [0.7168734597936699, 10.316014989651608], [3.0646175552586863, 12.01699294051685],
                           [14.217031825866442, 13.0862961193627], [11.232437545345256, 15.546233904751752],
                           [9.561670647330837, 0.14833761390052214], [15.723875010254321, 4.43149170050199],
                           [2.057079178975365, 4.4646568142482845], [15.775797039047681, 18.10224902248002],
                           [17.645315271123085, 10.981283444923314], [0.24357985953696426, 19.59321384318264],
                           [16.748683413895485, 4.338702298920934], [19.263570819979115, 19.69866730928855],
                           [14.002171053501417, 5.660188630855176], [14.386374631614501, 7.4264303213348075],
                           [11.271808316091592, 11.04096778677761], [17.148165986627347, 11.265650540354848],
                           [3.2814965947531816, 6.701840634659768], [1.4877519388208604, 0.04786170445773008],
                           [2.52791768407975, 18.412749409009272], [18.53497026737887, 17.11630352790132],
                           [19.22895164478972, 6.162667654735989], [6.421816058981024, 2.0993043717874604],
                           [9.906132582984117, 11.884550530173533], [16.802623021820263, 3.5657069924503704],
                           [3.76025551885679, 15.250176142506302], [6.562466785488644, 5.958940020890687],
                           [4.798303225287905, 12.140455187642562], [6.345440636874795, 16.69152942021671],
                           [13.902433480769565, 16.019756378552103], [19.956023624024624, 9.997983518994037],
                           [9.105636135458806, 6.733289233018267], [14.014014728678353, 13.55385186366748],
                           [14.810607884278648, 6.854553731032484], [0.8866810562357452, 4.6135217151250885],
                           [4.238276232262573, 14.918857822332267], [7.500189182394525, 6.719525789661618],
                           [9.424028208080902, 7.668551051570212], [17.83185737866736, 9.118134943797067],
                           [12.326031345113597, 1.0943819928976528], [9.887223692429183, 12.513732666420909],
                           [7.191588594061466, 10.333380346411865], [7.882868769020384, 19.03140472119847],
                           [0.1453243704238627, 2.4209991959205657], [3.4418422620616007, 1.7246818476029913],
                           [18.916691769619945, 16.813658507546222], [2.6362377189585118, 10.304006725085635],
                           [18.63972089002805, 5.554498242346504], [1.5706539520512686, 1.7263783473594096],
                           [10.248923804111243, 7.742804467794782], [1.3302550644570732, 7.449138994181796],
                           [12.217910076241658, 1.4037863124388528], [18.433300865871093, 5.486707782749896],
                           [16.01751208222645, 10.379193006447096], [9.842165784414172, 18.269517268397585],
                           [15.269209864072707, 10.835899789492139], [17.21457873429022, 18.386667584531658],
                           [16.483651583859352, 6.037182528251812], [14.579353364662184, 11.148229740062284],
                           [13.730216581770591, 4.958135250939528], [19.406335587309613, 13.183516803405597],
                           [13.053798123282602, 16.543020583991932], [17.117952816789124, 19.030260611604948],
                           [13.176290033600191, 19.63704442458871], [1.7490011578959241, 9.075260325953455],
                           [18.66661585386967, 8.837811841526026], [8.569973034815481, 19.3550222774238],
                           [11.319194081695958, 8.13993591497627], [11.752702510224333, 3.3351233791876855],
                           [2.231811763839928, 15.416054102050818], [15.0794460963355, 1.4466224439148068],
                           [10.152497149415005, 2.774599764177914], [8.93982429596413, 4.118408413037886],
                           [14.856731616969705, 8.361067575665153], [8.857563829984127, 4.464456209972991],
                           [19.503746039018, 3.7542727264123354], [14.185834724184188, 16.394200563259183],
                           [15.955698399595226, 8.262223446949656], [10.30758524134651, 14.324340965799458],
                           [4.67247244205049, 2.6910204598227816], [9.764662785743301, 10.113080469012184],
                           [1.1402988103008793, 0.5112111404773567], [12.620644842124733, 2.2415195830102763],
                           [1.1937041316427233, 17.725788894694656], [9.296196645927106, 1.2313870713212371],
                           [19.518589026618574, 4.19415982102681], [15.784016490529321, 4.339487417796876],
                           [11.794372771685746, 14.601123666006302], [14.615982594557659, 9.937720922657467],
                           [4.949563571797022, 5.6054895377021445], [6.909588907318969, 9.937733814564503],
                           [4.01382310419878, 3.2188954695280625], [12.792631587491265, 1.5516418472816174],
                           [0.9345835170137895, 16.787099224810305], [17.264805138146993, 12.423456630080569],
                           [8.060896178217316, 5.956449188583594], [19.298668299988513, 17.901283651000366],
                           [4.555533073449116, 15.161072861819163], [9.948920313026512, 8.517683085776957],
                           [10.997549155074434, 7.340880123975316], [6.758595882545508, 0.15894134705442875],
                           [17.561495624144065, 1.5508672979300187], [18.686911041856757, 18.586839260695264],
                           [4.582399326976237, 9.988425464352337], [15.632687287016179, 11.143140683914528],
                           [6.5611807916996945, 6.813777273745463], [11.95490151013362, 12.082882729798321],
                           [16.432498541064785, 11.149210656752222], [19.028372213802893, 4.440837999100009],
                           [2.7446438237720217, 12.019265258886659], [3.1360537395750687, 8.671925291482289],
                           [4.2415359569435545, 17.94866312497416], [11.063513772901333, 10.80750466648209],
                           [4.90891586142554, 6.5443003100563635], [3.7852971000107116, 10.960984673237865],
                           [9.48942843834972, 16.02134640192478], [7.8879339434920315, 17.17169925369673],
                           [10.864438119478317, 3.276518874011507], [4.3237404805704704, 13.538128532346613],
                           [15.599390910337918, 2.0334704519553526], [19.06401955544679, 12.176645839894382],
                           [9.75635426216538, 0.3901954311392841], [4.526812662588542, 1.434569586157246],
                           [5.7792680883794905, 5.430413484426526], [11.570755498344106, 10.265080449235198],
                           [14.181735230472107, 12.798641430829443], [9.05371246111039, 1.4909208429211307],
                           [10.28480211727376, 6.663403463171433], [12.164211601348377, 12.496351163581895],
                           [6.429152581980475, 15.20750834943182], [8.062965494732033, 16.865177410122264],
                           [7.469184096266554, 1.8153476903787613], [4.4960206844704675, 4.932578299533674],
                           [8.176434382882924, 15.593899981543252], [9.435680586435964, 7.744309438770447],
                           [1.4883368278545928, 16.919850265498326], [9.076389060166711, 6.316955906382622],
                           [12.045246595699785, 8.770921995917742], [14.168777528782906, 19.97680218021721],
                           [1.4110406500674455, 3.9010870636916017], [10.017217995035159, 19.444892644045527],
                           [2.728010741470157, 3.3180281227909547], [6.134087855584878, 5.131516408655184],
                           [9.778716158329981, 18.753431754352498], [2.4035359373690057, 8.348973535413622],
                           [7.711736983310525, 5.26723313611404], [13.206280502008081, 19.710245908604026],
                           [1.0827459167066933, 9.862453327250748], [16.560161542434603, 8.040164861081628],
                           [2.1007030109457814, 9.67672925185892], [4.3835384049508335, 9.780810593830688],
                           [13.681401124174794, 19.452820275675943], [1.21912318202825, 13.8366510218836],
                           [1.9725272009505979, 17.96822450973131], [10.38867845898932, 13.316384744906983],
                           [2.7187756860194923, 18.695454973526992], [18.852113066235646, 3.382615779418008],
                           [8.257640119569519, 18.756624804249153], [4.554903297192785, 9.55555863424448],
                           [7.41298606656132, 7.288776419753374], [19.041948751941483, 7.160078536009813],
                           [13.985033670257586, 18.787396990622717], [5.230156387124028, 0.3768678760140709],
                           [14.122002246395802, 8.997890963867812], [18.576357878221415, 16.3106623985628],
                           [3.8729196107905506, 14.293010753449664], [12.452206863987879, 17.12805332123277],
                           [9.817518163849154, 11.991251533809812], [5.692848048616548, 12.748468796860234],
                           [17.488450798671874, 2.567035301402265], [7.447167851362457, 2.7493789070898833],
                           [9.740594309770628, 10.9293591325019], [0.4190757945618695, 5.37291945091444],
                           [2.3855211801589205, 17.7575210090885], [9.45143369128906, 1.883461768427579],
                           [6.804781834933456, 15.911495900285718], [8.862907935842802, 12.802489483904882]])
        centroids = np.array([[3.495736945708241, 16.0307521725109], [13.704268851231214, 14.991998942116423],
                              [7.53357204941739, 3.552528657614926], [1.6941766788353596, 14.21129052647284],
                              [8.31299163680024, 3.5947892724351305], [2.900438791810418, 6.9308982271973125],
                              [13.71275241518517, 9.335290154881424], [17.759978843430176, 19.68171614937291],
                              [7.322271219243655, 16.30338412870792], [18.715924244988443, 6.646525903731955]])
        result = Solution.solution(points, centroids)
        self.assertEqual(result,
                         [6, 9, 3, 8, 9, 9, 8, 2, 8, 2, 5, 8, 1, 5, 1, 6, 1, 5, 5, 3, 6, 6, 6, 6, 3, 3, 1, 1, 4, 9, 5,
                          7, 6, 0, 9, 7, 6, 6, 6, 6, 5, 2, 0, 7, 9, 2, 6, 9, 0, 2, 3, 8, 1, 9, 4, 1, 6, 5, 0, 2, 4, 9,
                          4, 1, 5, 8, 5, 2, 7, 5, 9, 5, 6, 5, 4, 9, 6, 8, 6, 7, 9, 6, 6, 1, 1, 7, 7, 5, 9, 8, 6, 4, 3,
                          9, 4, 4, 6, 4, 9, 1, 6, 1, 2, 6, 5, 4, 0, 4, 9, 9, 1, 6, 5, 5, 2, 4, 0, 1, 4, 7, 0, 6, 6, 2,
                          9, 7, 5, 6, 2, 6, 6, 9, 3, 5, 0, 6, 5, 3, 8, 8, 4, 0, 9, 9, 4, 2, 2, 6, 1, 4, 4, 1, 8, 8, 2,
                          5, 8, 4, 0, 4, 6, 7, 5, 8, 5, 2, 8, 5, 2, 7, 5, 9, 5, 5, 7, 3, 0, 1, 0, 9, 8, 5, 2, 9, 1, 2,
                          6, 7, 0, 1, 6, 8, 9, 2, 6, 5, 0, 4, 8, 8])
