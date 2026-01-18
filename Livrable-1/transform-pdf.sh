#! /bin/bash
pandoc cahier-des-charges.md -o cahier-des-charges.pdf \
  --template=mytemplate.tex \
  --pdf-engine=xelatex \
  --toc \
  -V coverpage=true

  pandoc plan-de-developpement.md -o plan-de-developpement.pdf \
  --template=mytemplate.tex \
  --pdf-engine=xelatex \
  --toc \
  -V coverpage=true