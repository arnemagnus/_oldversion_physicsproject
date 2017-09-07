(TeX-add-style-hook
 "entry1"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (LaTeX-add-labels
    "eq:analytical_field_stream_function"
    "eq:analytical_velocity_field_parameter_f"
    "eq:analytical_velocity_field_parameter_a"
    "eq:analytical_velocity_field_parameter_b"
    "tab:euler-lyapunov-error-rk4-ref"
    "tab:heun-lyapunov-error-rk4-ref"
    "tab:kutta-lyapunov-error-rk4-ref"
    "tab:RK4-lyapunov-error-rk4-ref"))
 :latex)

