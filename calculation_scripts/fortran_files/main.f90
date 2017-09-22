program main
  use common_parameters, only : wp, atol_default, rtol_default, &
       & fac, maxfac
  use adaptive_step, only : rkdp54
  use test, only : rhs
  implicit none


  
  real(wp) :: t, x, h

  t = 0._wp
  x = 0._wp
  h = 1._wp/10
  write(*, '(A, A, F7.6, A, F7.6, A, F7.6)') 'Step 0:', '    t = ', t, ',    x = ', x, ',    h = ', h
  call rkdp54(t, x, h, rhs, t, x, h)
  write(*, '(A, A, F7.6, A, F7.6, A, F7.6)') 'Step 1:', '    t = ', t, ',    x = ', x, ',    h = ', h 
  
end program main
