# This file contains a set of numerical integration schemes for
# general-purpose use. To this end, the integrators are implemented
# with a consistent call signature. In addition, they can handle 
# any number of coordinates, as long as a function returning the
# derivative (RHS) is provided for each of them.

# Changelog: 
#     2017-08-25: File created. 
#                 Explicit Euler method, Heun's method, 
#                 Kutta's method and the classical RK4 method added.
#                 Added spaces (whitespace) everywhere slopes are 
#                 estimated, in order to make the differences between
#                 the various methods more apparent.
#
#     2017-08-30: Clarified the function signature for the derivative
#                 functions.
#
#     2017-09-01: Changed function signature of the integrators as
#                 well as the derivative functions, from
#                 f(x,t) --> f(t,x) in accordance with the literature.
#
#                 Added implementation of the Cash-Karp, Fehlberg and
#                 Dormand-Prince automatic step size integrators.
#                 Changed return variables of the fixed-stepsize 
#                 integrators, so that they are consistent with
#                 the return variables from their automatic stepsize
#                 siblings.
#
#                 Added a not fully functioning implementation
#                 of the Bogacki-Shampine 4(5) scheme.
# 
#     2017-09-07: Corrected several errors in terms of the matrix
#                 elements of the Cash-Karp 4(5) scheme. The scheme
#                 works as intended, now.
#
#                 Performed several similar corrections to the
#                 Bogacki-Shampine 4(5) scheme, which now works as
#                 intended.
#
#                 Added a working implementation of the
#                 Bogacki-Shampine 2(3) automatic step size integrator
#
#                 Added a working implementation of the Dormand-Prince
#                 7(8) automatic step size integrator
#
# Written by Arne M. T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.                

#--------------------------------------------------------------------#

import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Methods with fixed stepsize   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Explicit Euler method (also known as the simplest, one-stage
#                        Runge-Kutta method, 1st order accurate)
def euler(t, # Current time
          x, # Coordinates, as an array
          h, # Time step
          f  # Function handle for the derivatives (RHS),
             # function signature: f = f(t, x)
         ):
   # This function performs a single time step forwards, using the 
   # explicit Euler scheme, and returns the new coordinates at the
   # new time level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + k1*h
   
   return _t, _x, h



# Heun's method (also known as a two-stage Runge-Kutta method, 
#                2nd order accurate)
def rk2(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # Heun scheme, and returns the new coordinates at the new time 
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h  , x + k1 * h       )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + k2)*h/2

   return _t, _x, h



# Kutta's method (also known as a three-stage Runge-Kutta method, 
#                 3rd order accurate)
def rk3(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # Kutta scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h/2, x + k1*h/2       )
   k3 = f(t + h  , x - k1*h + 2*k2*h)
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + 4*k2 + k3)*h/6

   return _t, _x, h



# "The" Runge-Kutta method (also known as a four-step Runge-Kutta
#                           method, 4th order accurate)
def rk4(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # RK4 scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h/2, x + k1*h/2       )
   k3 = f(t + h/2, x + k2*h/2       )
   k4 = f(t + h  , x + k3*h         )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6

   return _t, _x, h
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Methods with adaptive stepsize   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# The Cash-Karp method (5th order method with 4th order
#                       interpolation, adaptive timestep)
def rkck(t,           # Current time
         x,           # Coordinates, as an array
         h,           # Time step
         f,           # Function handle for the derivatives (RHS),
                      # function signature: f = f(t, x)
         atol = 1e-6, # Absolute tolerance level (optional)
         rtol = 1e-9  # Relative tolerance level (optional)
        ):
   # This function attempts a single time step forwards, using the 
   # Cash-Karp adaptive timestep integrator scheme. If the new step is
   # not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./5.
   c3 = 3./10.
   c4 = 3./5.
   c5 = 1.
   c6 = 7./8.

   # Matrix elements
   a21 = 1./5.
   a31 = 3./40.
   a32 = 9./40.
   a41 = 3./10.
   a42 = -9./10.
   a43 = 6./5.
   a51 = -11./54.
   a52 = 5./2.
   a53 = -70./27.
   a54 = 35./27.
   a61 = 1631./55296.
   a62 = 175./512.
   a63 = 575./13824.
   a64 = 44275./110592.
   a65 = 253./4096.

   # Fourth-order weights
   b41 = 2825./27648.
   b42 = 0.
   b43 = 18575./48384.
   b44 = 13525./55296.
   b45 = 277./14336.
   b46 = 1./4.

   # Fifth-order weights
   b51 = 37./378.
   b52 = 0.
   b53 = 250./621.
   b54 = 125./594.
   b55 = 0.
   b56 = 512./1771.

   # Find "slopes"
   k1 = f(t       , x                                                       )
   k2 = f(t + c2*h, x + a21*h*k1                                            )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                                 )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3                      )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4           )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 4th order, with 5th order interpolation, hence:
   q = 5.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use fourth order result as next position
       _x = x_4
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h

# The Runge-Kutta-Fehlberg method (also known as the Fehlberg method,
#                                  4th order method with 5th order
#                                  interpolation, adaptive timestep)
def rkf(t,           # Current time
        x,           # Coordinates, as an array
        h,           # Time step
        f,           # Function handle for the derivatives (RHS),
                     # function signature: f = f(t, x)
        atol = 1e-6, # Absolute tolerance level (optional)
        rtol = 1e-9  # Relative tolerance level (optional)
        ):
   # This function attempts a single time step forwards, using the 
   # Fehlberg adaptive timestep integrator scheme. If the new step is
   # not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./4.
   c3 = 3./8.
   c4 = 12./13.
   c5 = 1.
   c6 = 1./2.

   # Matrix elements
   a21 = 1./4.
   a31 = 3./32.
   a32 = 9./32.
   a41 = 1932./2197.
   a42 = -7200./2197.
   a43 = 7296./2197.
   a51 = 439./216.
   a52 = -8.
   a53 = 3680./513
   a54 = -845./4104.
   a61 = -8./27.
   a62 = 2.
   a63 = -3544./2565.
   a64 = 1859./4104.
   a65 = -11./40.

   # Fourth-order weights
   b41 = 25./216.
   b42 = 0.
   b43 = 1408./2565.
   b44 = 2197./4104.
   b45 = -1./5.
   b46 = 0.

   # Fifth-order weights
   b51 = 16./135.
   b52 = 0.
   b53 = 6656./12825.
   b54 = 28561./56430.
   b55 = -9./50.
   b56 = 2./55.

   # Find "slopes"
   k1 = f(t       , x                                                       )
   k2 = f(t + c2*h, x + a21*h*k1                                            )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                                 )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3                      )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4           )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 4th order, with 5th order interpolation, hence:
   q = 4.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use fourth order result as next position
       _x = x_4
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h


# "The" Dormand-Prince method (5th order method with 4th order
#                              interpolation, adaptive timestep)
def rkdp54(t,           # Current time
           x,           # Coordinates, as an array
           h,           # Time step
           f,           # Function handle for the derivatives (RHS),
                        # function signature: f = f(t, x)
           atol = 1e-6, # Absolute tolerance level (optional)
           rtol = 1e-9  # Relative tolerance level (optional)
          ):
   # This function attempts a single time step forwards, using the 
   # Dormand-Prince adaptive timestep integrator scheme. If the new
   # step is not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./5.
   c3 = 3./10.
   c4 = 4./5.
   c5 = 8./9.
   c6 = 1.
   c7 = 1.

   # Matrix elements
   a21 = 1./5.
   a31 = 3./40.
   a32 = 9./40.
   a41 = 44./45.
   a42 = -56./15.
   a43 = 32./9.
   a51 = 19372./6561.
   a52 = -25350./2187.
   a53 = 64448./6561.
   a54 = -212./729.
   a61 = 9017./3168.
   a62 = -335./33.
   a63 = 46732./5247.
   a64 = 49./176.
   a65 = -5103./18656.
   a71 = 35./384.
   a72 = 0.
   a73 = 500./1113.
   a74 = 125./192.
   a75 = -2187./6784.
   a76 = 11./84.

   # Fourth-order weights
   b41 = 5179./57600.
   b42 = 0.
   b43 = 7571./16695.
   b44 = 393./640.
   b45 = -92097./339200.
   b46 = 187./2100.
   b47 = 1./40.

   # Fifth-order weights
   b51 = 35./384.
   b52 = 0.
   b53 = 500./1113.
   b54 = 125./192.
   b55 = -2187./6784.
   b56 = 11./84.
   b57 = 0.

   # Find "slopes"
   k1 = f(t       , x                                                                  )
   k2 = f(t + c2*h, x + a21*h*k1                                                       )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                                            )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3                                 )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4                      )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5           )
   k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4 + a75*h*k5 + a76*h*k6)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6 + b47*k7)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use fifth order result as next position
       _x = x_5
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h

# The Dormand-Prince 8(7) method (8th order method with 7th order
#                                 interpolation, adaptive timestep)
def rkdp87(t,           # Current time
            x,           # Coordinates, as an array
            h,           # Time step
            f,           # Function handle for the derivatives (RHS),
                         # function signature: f = f(t, x)
            atol = 1e-6, # Absolute tolerance level (optional)
            rtol = 1e-9  # Relative tolerance level (optional)
           ):
   # This function attempts a single time step forwards, using the 
   # Dormand-Prince 8(7) adaptive timestep integrator scheme. If the 
   # new step is not accepted, the time and coordinates are not
   # updated.

   # Nodes
   c2  = 1./18.
   c3  = 1./12.
   c4  = 1./8.
   c5  = 5./16.
   c6  = 3./8.
   c7  = 59./400.
   c8  = 93./200.
   c9  = 5490023248./9719169821.
   c10 = 13./20.
   c11 = 30992876149296355./33518267164510641.
   c12 = 1.
   c13 = 1.

   # Matrix elements
   a21   = 1./18.
   a31   = 1./48.
   a32   = 1./16.
   a41   = 1./32.
   a42   = 0.
   a43   = 3./32.
   a51   = 5./16.
   a52   = 0.
   a53   = -75./64.
   a54   = 75./64.
   a61   = 3./80.
   a62   = 0.
   a63   = 0.
   a64   = 3./16.
   a65   = 3./20.
   a71   = 215595617/4500000000
   a72   = 0.
   a73   = 0.
   a74   = 202047683./1800000000.
   a75   = -28693883./1125000000.
   a76   = 23124283./1800000000.
   a81   = 14873762658037143./879168438156250000.
   a82   = 0.
   a83   = 0.
   a84   = 3467633544794897./8940695981250000.
   a85   = 1474287494383247./40978189914062500.
   a86   = 26709270507070017./135600555715625000.
   a87   = -14591655588284./84484570233063.
   a91   = 7586331039021946882049083502441337664277676907617750536566352./109794461601491217860220353338581031394059220336451160078730445.
   a92   = 0.
   a93   = 0.
   a94   = -236057339412812449835946465344221735535939129430991059693568./372184615598275314780407977418918750488336340123563254504171.
   a95   = -3299739166368883603096250588167927276977533790499480498577408./20470153857905142312922438758040531276858498706795978997729405.
   a96   = 4695919603694846215470554638065271273971468502369170235542016./33868800019443053645017125945121606294438606951244256159879561.
   a97   = 291851811898394201384602939640627532330843113837053004434432000000./310174233778061645620360730197195350622945922304711702829528117367.
   a98   = 6992959981041103840944260661352231159203510904000000./ 33042342481018810238716485165383193327572243242031481.
   a101  = 99299034813490800741867453179778547./540971123539151162906952826011200000.
   a102  = 0.
   a103  = 0.
   a104  = -2493835259080554724582./1010153717930905426875.
   a105  = -48550347897506146536052./166675363458599395434375.
   a106  = -24871192635697392099560348960246./939492072180864357472739828818125.
   a107  = 478776089216929482237673925052922000./168119099731629344552415590032785027.
   a108  = 6560308981643238155096750./23314158982833116227901307.
   a109  = 1586281686644478270321241459439899956623408540189275177./12818966182821619734532382093543907143647820508227904000.
   a111  = -102116003386322998978127600084904875522141269364791505043913504184525097818434721165778087547359160299919872547571820573487921693./8401671738537636244051928845472275456111820610996845586362991556941300701548488408298927732714675061003289762042774165805944028.
   a112  = 0.
   a113  = 0.
   a114  = 338590872606752219742507143357021902717271169524361004010718467428498066558752974165816979255870352236800./20308212073515087965058545521329962060416948491603802421256875704911573108931922671691153944392874968051.
   a115  = 68189290605616416787948548385820859588684790288743680764422049661712817412461535969920258787664375619072./74463444269555322538548000244876527554862144469213942211275210918009101399417049796200897796107208216187.
   a116  = -1734282043732424474072631498514610096486835338935534079058145376760622893524503274680375038942168945756187943481380463560951840./286345537377499805912462279621622489249909215975695809863482134802066603511244489020404711919081949534640172065152437496912477.
   a117  = -3399549280223124443696423490103003766707892326374755946138975000967466690241111348721006509128775254952212682658842765965521154240000000./212424385105117691648087703103838079790425456287912424851546922389328485500145214289225448961304538830766072442444722564103495915888123.
   a118  = 14452808190943733856347403293564049428070036006455540637351575894308889412108389906599600485253194980566957563315340127500000./973298753951638431793701721528200883789914680313298926814615071301495341142665245758696799918623095581715765886887649741383.
   a119  = -847205714160239289113307424793539077951658318917591980262304042838612275700008766016957700930195545053374220841398660187944621107065829310608865394026418258355./63358704383980726998416112830322706485300332630289060627019459285960825979588560697460438306253611095891491565590971432387489415884103732012574255897878321824.
   a1110 = 115188988949323598098458035263894669359112068207548636038131244599058496172710646622536373145562218909633738697549245770000./22435701423704647109276644681016984863989966659062291511947760084943925084166270812354794844590216383205333034660617626349.
   a121  = 21969012306961489525323859125985377266525845354279828748./84868015648089839210997460517819380601933600521692915045.
   a122  = 0.
   a123  = 0.
   a124  = -2291872762438069505504./480025046760766258851.
   a125  = -3829018311866050387904./8800459190614048078935.
   a126  = -607977714773374460437401016185253441418120832060126402968./199370728929424959394190105343852509479613745231838418799.
   a127  = 5302029233035772894614097632213626682295966947853615180783170000000./950538766256052885387161080614691196420735587733978871061913292363.
   a128  = 102968047255116137164987219663037502898143843145000000./16726911019578511096352500731821705820659977305290973.
   a129  = -111383789341965407321602142444917514115800834690201329379027449761759895100011973929185171163615./22003454775272439861723739055800175619777853128055268766511800511549546753240522083740083243539.
   a1210 = 44737471541467333111555512048686345065750./20391511842264262870398286145868618178341.
   a1211 = 596546910748352988538198147432444829112451075399436970876618894337461087953328002664759407401623072330633057948252./4431076125983762085449284205348478790535717302043416234911901479328512794465980800998816354448181196721636373483787.
   a131  = 1066221205855832326088695778460159015192405644968016897066521076847764032613686056268693633./1296431693610525557488309197474904206216262654240544950471874305723890174339356551609704000.
   a132  = 0.
   a133  = 0.
   a134  = -1335791413506612664643690684478806471077526746614666064./114574907798601779179110271814903983120429559544320175.
   a135  = -1591415543044168099882026495959288688569084060473110176./2100539976307699284950354983273239690541208591645869875.
   a136  = 33975758488532631832742416857645572913178866704247539610423012370193845167470455176890924./47586856225469573819304596274208152402640120925455970356063642741972959597009066064956075.
   a137  = 12176653428667113090492984656207574633063967759246601254930448409444470870786024235115138527800000./1008353786145118968620988891518234034224047994442049071310258686840184337101721351612973016221399.
   a138  = -339784374935367314296824613776444883113869450234942131172912300100535979345925250000./159698690787587746004588725210359673189662237866695585709500421500486548151424426361.
   a139  = 4955095692700499418628052380948016677978733013841365878109775677669056866398110949788869771135857671298802131693154421086808143./2489789885462873158531234022579722982784822257458164105126884288597324542930882581099522281388970940826324647386340365850671680.
   a1310 = -563115171027780776675066866318087406247194110301648522108648094708415./2403532595444498372383116767918060257292523183751650851596520916634577.
   a1311 = 147332487580158450887955957061658718012538967463083369806963200702426559434915876714751833908862217396388157664714990174448521780809./837599084085749358149340415048050308970085851893614803629073546048735327947816070400330404870816820234727495143522673498826476267825.
   a1312 = 0.

   # Seventh-order weights
   b71  = 7136040226482108704342809557217./241464102842794736092004001974880.
   b72  = 0.
   b73  = 0.
   b74  = 0.
   b75  = 0.
   b76  = -15349154422148033115423212285265536./18524062462099973621193571994058285.
   b77  = 45434521806506196832804182374790400000000./145978635195580057402851847985603569106229.
   b78  = 365696286946774693155766999232150000000./148214481030059176862554298041717674741.
   b79  = -836336669851503831866889530158468123932231502753408325817124013619515886965077571./328368994730082689886153304749497093954319862912916225944630536728837081959128864.
   b710 = 294694385044387823293019951454286000./204145803180236295718144364590673643.
   b711 = 1759482754698187564675489259591170188433054767657805212470918093603353527288272972728828708146708084742711724049636./22155380629918810427246421026742393952678586510217081174559507396642563972329904004994081772240905983608181867418935.
   b712 = 2./45.
   b713 = 0.

   # Eighth-order weights
   b81  = 212810988215683677989664967567559./5097575504458999984164528930580800.
   b82  = 0.
   b83  = 0.
   b84  = 0.
   b85  = 0.
   b86  = -570667999368605802515460802224128./10291145812277763122885317774476825.
   b87  = 3970894643399159150754126826496000000000000./16592904867230933191457493387696939021741363.
   b88  = 177094288219480472437690862000000000000./251729356670100506734814442705774463449.
   b89  = -66822609448295850920212176513645119787713273203022994500406050793972052314809461629969645683./ 87952305220338336969447643899150816363456821562985998778022435070001091778042097545895594560.
   b810 = 314652731163869955629145958568800000./476340207420551356675670184044905167.
   b811 = 177014954088789647707522848990757432519504314686067075784476503038212450536095365316360385634933688213244039743969578872631174179769./1119019983628991838522384101261104859676427163726922121733732080377576616485631933067985100908132443862205090961383250990215178108200.
   b812 = -454665916000392064556420344242099./1909482158429176288068071462671400.
   b813 = 1./4.


   # Find "slopes"
   k1  = f(t        , x                                                                                                                                                      )
   k2  = f(t + c2*h , x + a21*h*k1                                                                                                                                           )
   k3  = f(t + c3*h , x + a31*h*k1 + a32*h*k2                                                                                                                                )
   k4  = f(t + c4*h , x + a41*h*k1 + a42*h*k2 + a43*h*k3                                                                                                                     )
   k5  = f(t + c5*h , x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4                                                                                                          )
   k6  = f(t + c6*h , x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5                                                                                               )
   k7  = f(t + c7*h , x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4 + a75*h*k5 + a76*h*k6                                                                                    )
   k8  = f(t + c8*h , x + a81*h*k1 + a82*h*k2 + a83*h*k3 + a84*h*k4 + a85*h*k5 + a86*h*k6 + a87*h*k7                                                                         )
   k9  = f(t + c9*h , x + a91*h*k1 + a92*h*k2 + a93*h*k3 + a94*h*k4 + a95*h*k5 + a96*h*k6 + a97*h*k7 + a98*h*k8                                                              )
   k10 = f(t + c10*h, x + a101*h*k1 + a102*h*k2 + a103*h*k3 + a104*h*k4 + a105*h*k5 + a106*h*k6 + a107*h*k7 + a108*h*k8 + a109*h*k9                                          )
   k11 = f(t + c11*h, x + a111*h*k1 + a112*h*k2 + a113*h*k3 + a114*h*k4 + a115*h*k5 + a116*h*k6 + a117*h*k7 + a118*h*k8 + a119*h*k9 + a1110*h*k10                            )
   k12 = f(t + c12*h, x + a121*h*k1 + a122*h*k2 + a123*h*k3 + a124*h*k4 + a125*h*k5 + a126*h*k6 + a127*h*k7 + a128*h*k8 + a129*h*k9 + a1210*h*k10 + a1211*h*k11              )
   k13 = f(t + c13*h, x + a131*h*k1 + a132*h*k2 + a133*h*k3 + a134*h*k4 + a135*h*k5 + a136*h*k6 + a137*h*k7 + a138*h*k8 + a139*h*k9 + a1310*h*k10 + a1311*h*k11 + a1312*h*k12)

   # Find seventh and eighth order prediction of new point
   x_7 = x + h*(b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6 + b77*k7
                + b78*k8 + b79*k9 + b710*k10 + b711*k11 + b712*k12 + b713*k13)
   x_8 = x + h*(b81*k1 + b82*k2 + b83*k3 + b84*k4 + b85*k5 + b86*k6 + b87*k7
                + b88*k8 + b89*k9 + b810*k10 + b811*k11 + b812*k12 + b813*k13)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 8th order, with 7th order interpolation, hence:
   q = 7.
   
   sc = atol + np.maximum(np.abs(x_7), np.abs(x_8)) * rtol
   err = np.amax(np.sqrt((x_7-x_8)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use fifth order result as next position
       _x = x_8
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h

# The (third order) Bogacki-Shampine method (3rd order method with 2nd
#                                            order interpolation,
#                                            adaptive timestep)

def rkbs32(t,           # Current time
           x,           # Coordinates, as an array
           h,           # Time step
           f,           # Function handle for the derivatives (RHS),
                         # function signature: f = f(t, x)
           atol = 1e-6, # Absolute tolerance level (optional)
           rtol = 1e-9  # Relative tolerance level (optional)
          ):
   # This function attempts a single time step forwards, using the 
   # Bogacki-Shampine 3(2) adaptive timestep integrator scheme. If the
   # new step is not accepted, the time and coordinates are not
   # updated.

   # Nodes
   c2 = 1./2.
   c3 = 3./4.
   c4 = 1.

   # Matrix elements
   a21 = 1./2.
   a31 = 0.
   a32 = 3./4.
   a41 = 2./9.
   a42 = 1./3.
   a43 = 4./9.

   # Second order weights
   b21 = 7./24.
   b22 = 1./4.
   b23 = 1./3.
   b24 = 1./8.

   # Third order weights
   b31 = 2./9.
   b32 = 1./3.
   b33 = 4./9.
   b34 = 0.

   # Find "slopes"
   k1 = f(t       , x                                 )
   k2 = f(t + c2*h, x + a21*h*k1                      )
   k3 = f(t + c2*h, x + a31*h*k1 + a32*h*k2           )
   k4 = f(t + c3*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3)

   # Find second and third order prediction of new point
   x_2 = x + h*(b21*k1 + b22*k2 + b23*k3 + b24*k4)
   x_3 = x + h*(b31*k1 + b32*k2 + b33*k3 + b34*k4)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 3rd order, with 2nd order interpolation, hence:
   q = 2.
   
   sc = atol + np.maximum(np.abs(x_2), np.abs(x_3)) * rtol
   err = np.amax(np.sqrt((x_2-x_3)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use third order result as next position
       _x = x_3
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h
   
# The (fifth order) Bogacki-Shamine method (5th order method with 4th
#                                            order interpolation,
#                                            adaptive timestep)
#
# For reference, see:
#  Bogacki, P., Shampine, L.F. (1996):
#                  "An efficient Runge-Kutta (4,5) pair",
#                  Computers Math. Applic., Vol. 32, no. 6, pp. 15-28

def rkbs54(t,           # Current time
           x,           # Coordinates, as an array
           h,           # Time step
           f,           # Function handle for the derivatives (RHS),
                        # function signature: f = f(t, x)
           atol = 1e-6, # Absolute tolerance level (optional)
           rtol = 1e-9  # Relative tolerance level (optional)
          ):
   # This function attempts a single time step forwards, using the 
   # Bogacki-Shampine 5(4) adaptive timestep integrator scheme. If the 
   # new step is not accepted, the time and coordinates are not
   # updated.

   # Nodes
   c2 = 1./6.
   c3 = 2./9.
   c4 = 3./7.
   c5 = 2./3.
   c6 = 3./4.
   c7 = 1.
   c8 = 1.

   # Matrix elements
   a21 = 1./6.
   a31 = 2./27.
   a32 = 4./27.
   a41 = 183./1372.
   a42 = -162./343.
   a43 = 1053./1372.
   a51 = 68./297.
   a52 = -4./11.
   a53 = 42./143.
   a54 = 1960./3861.
   a61 = 597./22528.
   a62 = 81./352.
   a63 = 63099./585728.
   a64 = 58653./366080.
   a65 = 4617./20480.
   a71 = 174197./959244.
   a72 = -30942./79937.
   a73 = 8152137./19744439.
   a74 = 666106./1039181.
   a75 = -29421./29068.
   a76 = 482048./414219.
   a81 = 587./8064.
   a82 = 0.
   a83 = 4440339./15491840.
   a84 = 24353./124800.
   a85 = 387./44800.
   a86 = 2152./5985.
   a87 = 7267./94080.

   # First of the fourth-order weights
   b41 = 6059./80640.
   b42 = 0.
   b43 = 8559189./30983680.
   b44 = 26411./124800.
   b45 = -927./89600.
   b46 = 443./1197.
   b47 = 7267./94080.
   b48 = 0.
   
   # Second of the fourth-order weights
   _b41 = 2479./34992.
   _b42 = 0.
   _b43 = 123./416.
   _b44 = 612941./3411720.
   _b45 = 43./1440.
   _b46 = 2272./6561.
   _b47 = 79937./1113912.
   _b48 = 3293./556956.

   # Fifth-order weights
   b51 = 587./8064.
   b52 = 0.
   b53 = 4440339./15491840.
   b54 = 24353./124800.
   b55 = 387./44800.
   b56 = 2152./5985.
   b57 = 7267./94080.
   b58 = 0.

   # Find "slopes"
   k1 = f(t       , x                                                                             )
   k2 = f(t + c2*h, x + a21*h*k1                                                                  )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                                                       )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3                                            )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4                                 )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5                      )
   k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4 + a75*h*k5 + a76*h*k6           )
   k8 = f(t + c8*h, x + a81*h*k1 + a82*h*k2 + a83*h*k3 + a84*h*k4 + a85*h*k5 + a86*h*k6 + a87*h*k7)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6 + b47*k7 + b48*k8         )
   _x_4 = x + h*(_b41*k1 + _b42*k2 + _b43*k3 + _b44*k4 + _b45*k5 + _b46*k6 + _b47*k7 + _b48*k8)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7 + b58*k8         )

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 5.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(_x_4)) * rtol
   err = np.amax(np.sqrt((x_4-_x_4)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # First trial step accepted.
       sc = atol + np.maximum(np.abs(_x_4), np.abs(x_5)) * rtol
       err = np.amax(np.sqrt((_x_4-x_5)**2)/sc)
       if err <= 1.:
           # Step is accepted, use fifth order result as next position
           _x = x_5
           _t = t + h
           # Refining h:
           # Should err happen to be 0, the optimal h is infinity.
           # We set an upper limit to get sensible behaviour:
           if err == 0.:
               h_opt = 10
           else:
               h_opt = h * (1./err) ** (1./(q + 1.))
           _h = max(maxfac * h, fac * h_opt)
       else:
            # Step is rejected, position and time not updated.
            _x = x
            _t = t
            # Refining h:
            h_opt = h * (1./err) ** (1./(q + 1.))
            _h = fac * h_opt
   else:
       # First trial step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h, based on first trial step:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h