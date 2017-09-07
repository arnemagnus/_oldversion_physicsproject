(TeX-add-style-hook
 "journal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("xcolor" "usenames" "dvipsnames") ("layaureo" "big") ("natbib" "round")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "fontspec"
    "xunicode"
    "xltxtra"
    "url"
    "parskip"
    "color"
    "graphicx"
    "xcolor"
    "layaureo"
    "hyperref"
    "natbib"
    "tikz"
    "amsmath"
    "bm")
   (TeX-add-symbols
    '("vect" 1))
   (LaTeX-add-bibliographies
    "mybibliography")
   (LaTeX-add-xcolor-definecolors
    "linkcolour"))
 :latex)

