---
title: 'MODE.behave: A Python Package for Estimating Discrete Choice Models'
tags:
  - Python
  - discrete choice
  - mixed logit
authors:
  - name: Julian Paul Reul
    orcid: 0000-0002-2786-0233
    corresponding: true
    affiliation: "1, 2"
  - name: Thomas Grube
    orcid: 0000-0002-0605-8356
    affiliation: 1
  - name: Jochen Linßen
    orcid: 0000-0003-2805-2307
    affiliation: 1
  - name: Detlef Stolten
    orcid: 0000-0002-1671-3262
    affiliation: "1, 2"
affiliations:
 - name: Institute for Techno-economic Systems Analysis (IEK-3), Forschungszentrum Jülich GmbH, Germany
   index: 1
 - name: Chair for Fuel Cells, RWTH Aachen University, Germany
   index: 2

date: 09 August 2022
bibliography: paper.bib
---

# Summary

The analysis of choice behavior is an important element in economic research as 
well an in related fields, such as the social sciences or civil engineering. 
Discrete choice theory is the mathematical foundation for the analysis of 
individual and aggregate choice behavior, which became widely established 
since the publication of seminal theoretical works in the 1970s in the context 
of transport-related research questions 
[@BenAkiva1973; @BenAkiva1985; @Train1985; @McFadden1976]. 
Examples of typical choice situations are the choice of an households’ energy 
supplier, the purchase of a new car or the study of political sentiment. 
In recent years, a new modeling approach in the field of discrete choice theory 
became popular – the mixed logit model [@Train2009]. Conventional discrete 
choice models have limited capabilities in describing the heterogeneity of 
choice preferences within a base population. I.e., divergent choice behavior of
different individuals or consumer groups can only be studied to a limited 
degree. Mixed logit overcomes this deficiency and allows for the analysis of 
preference distributions across the base population.

# Statement of need

MO|DE.behave is a python-based software package which enables the researcher to
estimate a specific type of mixed logit model. – The mixed logit model with 
nonparametric distributions [@Train2008; @Train2016]. The advantage of this 
model type is, that any preference distribution and any type of conventional 
discrete choice model can be approximated as closely as one pleases, only 
limited by computational complexity [@McFadden2000]. The software enables the 
use of GPU-hardware to accelerate the estimation process of such mixed logit 
models. Furthermore, discrete choice simulation methods for transportation 
research are included to enrich the software package for this specific 
community. MO|DE.behave complements already available python 
[@Arteaga2022; @Bierlaire2020; @Brathwaite2020] and 
R [@Croissant2020; @Hess2021; @Molloy2021] software packages for the estimation
of discrete choice models. However, it is the first to provide functionality 
for the estimation of mixed logit models with nonparametric distributions and 
additionally incorporates ready-to-apply simulation methods for the specific 
field of transportation research. Until now, MO|DE.behave has only been used 
internally at the Institute for Techno-economic Systems Analysis (IEK-3) at the
research center Jülich, Germany. Primary applications took place in the field 
of transportation research [DOI: Behavior change pre-print]. The publication 
of MO|DE.behave aims to ease the application of mixed logit models, especially 
with nonparametric design, for any student with an interest in choice modeling.

# Acknowledgements

The contributions to this paper are listed according to CRediT-taxonomy.

Julian Reul: Methodology, software, validation, formal analysis, data curation, 
writing – original draft.

Dr. Thomas Grube: Conceptualization, writing – review and editing, supervision, 
project administration, funding acquisition.
 
Prof. Dr. Jochen Linßen: Conceptualization, writing – review and editing, 
supervision, project administration, funding acquisition.

Prof. Dr. Detlef Stolten: Conceptualization, supervision, 
project administration, funding acquisition.

The authors declare that they have no known competing financial interests or 
personal relationships that could have appeared to influence the work reported 
in this paper.

Funding: This work was supported by the Helmholtz Association of German Research Centers.

# References