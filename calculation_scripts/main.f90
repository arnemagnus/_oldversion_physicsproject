program main
  use common_parameters, only : wp, atol_default, rtol_default, &
       & fac, maxfac
  use adaptive_step, only : rkbs32
  use test, only : rhs
  implicit none


  
  real(wp) :: t, x, h
  write(*, *) maxfac
  t = 0._wp
  x = 0._wp
  h = 1._wp/10
  write(*, *) t, x, h
  call rkbs32(t, x, h, rhs, t, x, h)
  write(*, *) t, x, h
  
end program main
