#! /bin/bash
pandoc rapport-final.md -o rapport-final.pdf \
  --template=mytemplate.tex \
  --pdf-engine=xelatex \
  --toc \
  --citeproc \
  -V coverpage=true
