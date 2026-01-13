#! /bin/bash
pandoc cahier-des-charges.md -o cahier-des-charges.pdf \
  --template=mytemplate.tex \
  --pdf-engine=xelatex \
  --toc \
  -V coverpage=true