module common_parameters
implicit none
  integer, parameter  :: wp = selected_real_kind(8)
  real(wp), parameter :: atol_default = 1._wp / (10**6)
  real(wp), parameter :: rtol_default = 1._wp / (10**9)
  real(wp), parameter :: fac = 8._wp/10
  real(wp), parameter :: maxfac = 2._wp
end module common_parameters

module single_step
  use common_parameters, only : wp
  implicit none
contains
  subroutine euler(t, x, h, f, t_, x_, h_)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!

    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current position
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New position
                              &h_            ! New time step

    external                :: f             ! External subroutine,
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp     
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative
                            !------------------------------------!
                            !--------------"Slopes"--------------!
                            !------------------------------------!
    real(wp)                :: k1
    
    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!

    ! Find "slopes"
    call f(t      , x              , k1)

    ! Find new time level
    t_ = t + h

    ! Find estimate for pos. at new time level
    x_ = x + k1*h    

    ! This is a fixed timestep method, hence
    h_ = h
  end subroutine euler

  subroutine rk2(t, x, h, f, t_, x_, h_)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!

    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current position
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New position
                              &h_            ! New time step

    external                :: f             ! External subroutine, 
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp             
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative
 
                            !------------------------------------!
                            !--------------"Slopes"--------------!
                            !------------------------------------!
    real(wp)                :: k1,&
                               &k2
    
    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!


    ! Find "slopes"
    call f(t      , x              , k1)
    call f(t + h  , x + h*k1       , k2)
    
    ! Find new time level
    t_ = t + h

    ! Find estimate for pos. at new time level
    x_ = x + (k1 + k2) * h/2

    ! This is a fixed timestep method, hence
    h_ = h
  end subroutine rk2

  subroutine rk3(t, x, h, f, t_, x_, h_)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!

    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current position
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New position
                              &h_            ! New time step

    external                :: f             ! External subroutine
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp     
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative

                            !------------------------------------!
                            !--------------"Slopes"--------------!
                            !------------------------------------!
    real(wp)                :: k1,&
                              &k2,&
                              &k3
    
    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!


    ! Find "slopes"
    call f(t      , x                , k1)
    call f(t + h/2, x + h*k1/2       , k2)
    call f(t + h  , x - h*k1 + 2*h*k2, k3)
    
    ! Find new time level
    t_ = t + h

    ! Find estimate for pos. at new time level
    x_ = x + (k1 + 4*k2 + k3) * h/6

    ! This is a fixed timestep method, hence
    h_ = h
  end subroutine rk3

  subroutine rk4(t, x, h, f, t_, x_, h_)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!

    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current pos. 
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New pos.
                              &h_            ! New time step

    external                :: f             ! External subroutine
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp             
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative

                            !------------------------------------!
                            !--------------"Slopes"--------------!
                            !------------------------------------!
    real(wp)                :: k1,&
                              &k2,&
                              &k3,&
                              &k4
    
    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!


    ! Find "slopes"
    call f(t      , x                , k1)
    call f(t + h/2, x + h*k1/2       , k2)
    call f(t + h/2, x + h*k2/2       , k3)
    call f(t + h  , x + h*k3         , k4)
    
    ! Find new time level
    t_ = t + h

    ! Find estimate for pos. at new time level
    x_ = x + (k1 + 2*k2 + 2*k3 + k4) * h/6

    ! This is a fixed timestep method, hence
    h_ = h
  end subroutine rk4
  
end module single_step  


module adaptive_step
  use common_parameters, only : wp, atol_default, rtol_default, &
                               &fac, maxfac
  implicit none
contains
  subroutine rkdp54(t, x, h, f, t_, x_, h_, atol_opt, rtol_opt)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!
    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current pos. 
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New position
                              &h_            ! New time step

    real(wp), intent(in), &
         &    optional      :: atol_opt, &   ! Absolute tolerance
                              &rtol_opt      ! Relative tolerance

    real(wp)                :: atol, &       ! Absolute tolerance
                               &rtol         ! Relative tolerance

                              !------------------------------------!
                              !----------------Nodes---------------!
                              !------------------------------------!
    real(wp), parameter     :: c2 = 1._wp/5, &
                              &c3 = 3._wp/10, &
                              &c4 = 4._wp/5, &
                              &c5 = 8._wp/9, &
                              &c6 = 1._wp, &
                              &c7 = 1._wp

                              !------------------------------------!
                              !-----------Matrix elements----------!
                              !------------------------------------!
    real(wp), parameter     :: a21 = 1._wp/5, &
                              &a31 = 3._wp/40, &
                              &a32 = 9._wp/40, &
                              &a41 = 44._wp/45, &
                              &a42 = -56._wp/15, &
                              &a43 = 32._wp/9, &
                              &a51 = 19372._wp/6561, &
                              &a52 = -25350._wp/2187, &
                              &a53 = 64448._wp/6561, &
                              &a54 = -212._wp/729, &
                              &a61 = 9017._wp/3168, &
                              &a62 = -335._wp/33, &
                              &a63 = 46732._wp/5247, &
                              &a64 = 49._wp/176, &
                              &a65 = -5103._wp/18656, &
                              &a71 = 35._wp/384, &
                              &a72 = 0._wp, &
                              &a73 = 500._wp/1113, &
                              &a74 = 125._wp/192, &
                              &a75 = -2187._wp/6784, &
                              &a76 = 11._wp/84

                              !------------------------------------!
                              !--------Fourth order weights--------!
                              !------------------------------------!
    real(wp), parameter     :: b41 = 5179._wp/57600, &
                              &b42 = 0._wp, &
                              &b43 = 7571._wp/16695, &
                              &b44 = 393._wp/640, &
                              &b45 = -92097._wp/339200, &
                              &b46 = 187._wp/2100, &
                              &b47 = 1._wp/40
                              
                              !------------------------------------!
                              !---------Fifth order weights--------!
                              !------------------------------------!
    real(wp), parameter     :: b51 = 35._wp/384, &
                              &b52 = 0._wp, &
                              &b53 = 500._wp/1113, &
                              &b54 = 125._wp/192, &
                              &b55 = -2187._wp/6784, &
                              &b56 = 11._wp/84, &
                              &b57 = 0._wp

                              !------------------------------------!
                              !--------------"Slopes"--------------!
                              !------------------------------------!
    real(wp)                :: k1, &
                              &k2, &
                              &k3, &
                              &k4, &
                              &k5, &
                              &k6, &
                              &k7

    real(wp)                :: x_4, &        ! Fourth order trial 
                                             ! step coordinate
                              &x_5           ! Fifth order trial 
                                             ! step coordinate

    real(wp), parameter     :: q = 4._wp     ! Interpolant order, 
                                             ! used to update
                                             ! time step

    real(wp)                :: sc, &         ! Used to estimate
                              &err           ! the error of the
                                             ! trial step 



    external                :: f             ! External subroutine
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp             
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative

    !--------------------------------------------------------------!
    !-----------------Handling Optional arguments------------------!
    !--------------------------------------------------------------!

    if(present(atol_opt)) then
       atol = atol_opt
    else
       atol = atol_default
    endif

    if(present(rtol_opt)) then
       rtol = rtol_opt
    else
       rtol = rtol_default
    endif

    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!

    ! Find "slopes"
    call f(t     , x                                              , k1)
    call f(t+c2*h, x+h*a21*k1                                     , k2)
    call f(t+c3*h, x+h*(a31*k1+a32*k2)                            , k3)
    call f(t+c4*h, x+h*(a41*k1+a42*k2+a43*k3)                     , k4)
    call f(t+c5*h, x+h*(a51*k1+a52*k2+a53*k3+a54*k4)              , k5)
    call f(t+c6*h, x+h*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)       , k6)
    call f(t+c7*h, x+h*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6), k7)


    ! Find fourth and fifth order predictions of new point
    x_4 = x + h*(b41*k1+b42*k2+b43*k3+b44*k4+b45*k5+b46*k6+b47*k7)
    x_5 = x + h*(b51*k1+b52*k2+b53*k3+b54*k4+b55*k5+b56*k6+b57*k7)

    ! Implementing error check and variable stepsize roughly as in
    ! Hairer, Nørsett and Wanner: "Solving ordinary differential
    !                              equations I -- nonstiff problems",
    !                              pages 167 and 168 in the 2008 ed.

    sc = atol + max(abs(x_4), abs(x_5)) * rtol
    err = (abs(x_4-x_5)/sc)

    if(err .LE. 1) then
       ! Step is accepted, use fifth order result as new position
       x_ = x_5
       t_ = t + h
       ! Refining h:
       !             Should err happen to be 0, the optimal h is
       !             infinity. We set an upper limit to get sensible
       !             behaviour
       if(err .EQ. 0._wp) then
          h_ = 10._wp
       else
          h_ = h * (1._wp/err)**(1._wp/(q + 1._wp))
       endif
       h_ = max(maxfac*h, fac*h_)
    else
       ! Step is rejected, position and time not updated
       x_ = x
       t_ = t
       ! Refining h:
       h_ = fac * h * (1._wp/err)**(1._wp/(q + 1._wp))
    endif

  end subroutine rkdp54

  subroutine rkbs32(t, x, h, f, t_, x_, h_, atol_opt, rtol_opt)
    implicit none
    !--------------------------------------------------------------!
    !-------------------------Declarations-------------------------!
    !--------------------------------------------------------------!
    real(wp), intent(in)    :: t, &          ! Current time 
                              &x, &          ! Current pos. 
                              &h             ! Current timestep 

    real(wp), intent(out)   :: t_, &         ! New time level
                              &x_, &         ! New position
                              &h_            ! New time step

    real(wp), intent(in), &
         &    optional      :: atol_opt, &   ! Absolute tolerance
                              &rtol_opt      ! Relative tolerance

    real(wp)                :: atol, &       ! Absolute tolerance
                               &rtol         ! Relative tolerance

                              !------------------------------------!
                              !----------------Nodes---------------!
                              !------------------------------------!
    real(wp), parameter     :: c2 = 1._wp/2, &
                              &c3 = 3._wp/4, &
                              &c4 = 1._wp

                              !------------------------------------!
                              !-----------Matrix elements----------!
                              !------------------------------------!
    real(wp), parameter     :: a21 = 1._wp/2, &
                              &a31 = 0._wp, &
                              &a32 = 3._wp/4, &
                              &a41 = 2._wp/9, &
                              &a42 = 1._wp/3, &
                              &a43 = 4._wp/9

                              !------------------------------------!
                              !--------Second order weights--------!
                              !------------------------------------!
    real(wp), parameter     :: b21 = 7._wp/24, &
                              &b22 = 1._wp/4, &
                              &b23 = 1._wp/3, &
                              &b24 = 1._wp/8
                              
                              !------------------------------------!
                              !---------Third order weights--------!
                              !------------------------------------!
    real(wp), parameter     :: b31 = 2._wp/9, &
                              &b32 = 1._wp/3, &
                              &b33 = 4._wp/9, &
                              &b34 = 0._wp

                              !------------------------------------!
                              !--------------"Slopes"--------------!
                              !------------------------------------!
    real(wp)                :: k1, &
                              &k2, &
                              &k3, &
                              &k4

    real(wp)                :: x_2, &        ! Second order trial 
                                             ! step coordinate
                              &x_3           ! Third order trial 
                                             ! step coordinate

    real(wp), parameter     :: q = 2._wp     ! Interpolant order, 
                                             ! used to update
                                             ! time step

    real(wp)                :: sc, &         ! Used to estimate
                              &err           ! the error of the
                                             ! trial step 



    external                :: f             ! External subroutine
                                             ! returning the
                                             ! derivatives (RHS)

    interface derivative                     ! Interfacing needed for
       subroutine f(t, x, deriv)             ! use with e.g. f2py
         use common_parameters, only :wp             
         implicit none
         real(wp), intent(in)  :: t, x
         real(wp), intent(out) :: deriv
       end subroutine f
    end interface derivative

    !--------------------------------------------------------------!
    !-----------------Handling Optional arguments------------------!
    !--------------------------------------------------------------!

    if(present(atol_opt)) then
       atol = atol_opt
    else
       atol = atol_default
    endif

    if(present(rtol_opt)) then
       rtol = rtol_opt
    else
       rtol = rtol_default
    endif

    !--------------------------------------------------------------!
    !--------------------------Calculations------------------------!
    !--------------------------------------------------------------!

    ! Find "slopes"
    call f(t     , x                                              , k1)
    call f(t+c2*h, x+h*a21*k1                                     , k2)
    call f(t+c3*h, x+h*(a31*k1+a32*k2)                            , k3)
    call f(t+c4*h, x+h*(a41*k1+a42*k2+a43*k3)                     , k4)


    ! Find second and third order predictions of new point
    x_2 = x + h*(b21*k1+b22*k2+b23*k3+b24*k4)
    x_3 = x + h*(b31*k1+b32*k2+b33*k3+b34*k4)

    ! Implementing error check and variable stepsize roughly as in
    ! Hairer, Nørsett and Wanner: "Solving ordinary differential
    !                              equations I -- nonstiff problems",
    !                              pages 167 and 168 in the 2008 ed.

    sc = atol + max(abs(x_2), abs(x_3)) * rtol
    err = (abs(x_2-x_3)/sc)

    if(err .LE. 1) then
       ! Step is accepted, use third order result as new position
       x_ = x_3
       t_ = t + h
       ! Refining h:
       !             Should err happen to be 0, the optimal h is
       !             infinity. We set an upper limit to get sensible
       !             behaviour
       if(err .EQ. 0._wp) then
          h_ = 10._wp
       else
          h_ = h * (1._wp/err)**(1._wp/(q + 1._wp))
       endif
       h_ = max(maxfac*h, fac*h_)
    else
       ! Step is rejected, position and time not updated
       x_ = x
       t_ = t
       ! Refining h:
       h_ = fac * h * (1._wp/err)**(1._wp/(q + 1._wp))
    endif

  end subroutine rkbs32
  
end module adaptive_step

