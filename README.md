# patternMorph
This repository is meant for code used to model a range of patterning and morphogenetic phenomena in biology. 

To run this code you need to download and install deal.ii from the website: www.dealii.org. Then clone this repo, and add a CMakeLists.txt in your local patternMorph/ folder. You can simply copy CMakeLists.txt from any of the deal.ii example folders. For instance, if you copied CMakeLists.txt from the example1 folder, modify the following line in CMakeLists.txt:

SET(TARGET "example1") --> SET(TARGET "diffusionAndMechanics10")

Then run the following commands:
>cmake CMakeLists.txt
>make

Please consider citing the following paper:
@article{Garikipati2016,
title   = {Perspectives on the mathematics of biological patterning and morphogenesis},
author  = {Garikipati, K.},
journal = {Journal of the Mechanics and Physics of Solids},
note    = {to appear},
doi     = {10.1016/j.jmps.2016.11.013},
archivePrefix = "arXiv",
eprint  = {1610.02584},
primaryClass = "q-bio",
Year    = {2016}
}

