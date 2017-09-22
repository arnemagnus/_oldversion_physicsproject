
module test  
use common_parameters, only : wp
contains
  subroutine rhs(t, x, deriv)
    implicit none
    real(wp), intent(in) :: t, x
    real(wp), intent(out) :: deriv
    deriv = 1 / (1 + t**2)
  end subroutine rhs
end module test

